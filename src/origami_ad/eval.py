"""Evaluate FoldingAgent runs against PurelandFold ground truth (T1, T6).

For a run directory produced through ``origami_ad.run_agent`` (or FoldingAgent directly):

1. map each solved frame to its PurelandFold step (``keyframes_meta.json`` next to the
   manifest; frame n of a PurelandFold run is step n);
2. convert the checkpoint's ``raw_representation`` to a crease-pattern FOLD dict;
3. compile prediction and ground truth with Flat-Folder (CV);
4. compute TSS / GS / CS / FFS and CPD per step.

Per-sequence numbers follow the paper: CV is averaged over all solved steps, the
similarity metrics and CPD are reported at the final solved step (``*_final``) and also
averaged over steps (``*_mean``). Completion, cost and runtime come from the run log.

Outputs: ``results/<condition>/metrics/<run>.json`` per run and, via ``summarize``, a
``results/summary.csv`` / ``summary.md`` table with PurelandFold vs YouTube per sequence
and averaged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from origami_ad.dataset import get_sequence
from origami_ad.metrics import fold_utils as fu
from origami_ad.metrics.cpd import crease_pattern_dissimilarity
from origami_ad.metrics.flatfolder import compile_many
from origami_ad.metrics.origamispace import all_similarities
from origami_ad.paths import RESULTS_DIR

METRIC_KEYS = ("cv", "tss", "gs", "cs", "ffs", "cpd")


def load_checkpoints(run_dir: Path) -> dict[int, dict[str, Any]]:
    path = run_dir / "checkpoints.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {int(k): v["raw_representation"] for k, v in data.items() if v.get("raw_representation")}


def frame_to_step_map(run_dir: Path, manifest: Path | None) -> dict[int, int | None]:
    """1-based agent frame number -> PurelandFold step (None for extra steps)."""
    meta_path = (manifest.parent / "keyframes_meta.json") if manifest else None
    if meta_path and meta_path.exists():
        metas = json.loads(meta_path.read_text())
        return {i + 1: m.get("step") for i, m in enumerate(metas)}
    # Fallback: PurelandFold layout where frame n is step n.
    frames = (
        json.loads((run_dir / "frames.json").read_text())
        if (run_dir / "frames.json").exists()
        else []
    )
    return {i + 1: i + 1 for i in range(len(frames))}


def dtw_align(cost: np.ndarray) -> list[tuple[int, int]]:
    """Monotone alignment path through a (frames x steps) cost matrix, both ends anchored."""
    n, m = cost.shape
    acc = np.full((n, m), np.inf)
    acc[0, 0] = cost[0, 0]
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                continue
            best = min(
                acc[i - 1, j] if i > 0 else np.inf,
                acc[i, j - 1] if j > 0 else np.inf,
                acc[i - 1, j - 1] if i > 0 and j > 0 else np.inf,
            )
            acc[i, j] = cost[i, j] + best
    i, j = n - 1, m - 1
    path = [(i, j)]
    while (i, j) != (0, 0):
        cands = []
        if i > 0 and j > 0:
            cands.append((acc[i - 1, j - 1], i - 1, j - 1))
        if i > 0:
            cands.append((acc[i - 1, j], i - 1, j))
        if j > 0:
            cands.append((acc[i, j - 1], i, j - 1))
        _, i, j = min(cands)
        path.append((i, j))
    return path[::-1]


def auto_align_frames(preds: list[dict[str, Any]], gt_steps: dict[int, Any]) -> dict[int, int]:
    """frame position (0-based) -> GT step, by DTW on CPD between predicted and GT patterns.

    Used when keyframes came from automatic keyframing and carry no step labels. Each
    predicted frame takes the GT step of lowest cost among those the path pairs it with.
    """
    steps = sorted(gt_steps)
    cost = np.array(
        [[crease_pattern_dissimilarity(pr, gt_steps[s].fold).total for s in steps] for pr in preds]
    )
    path = dtw_align(cost)
    best: dict[int, tuple[float, int]] = {}
    for i, j in path:
        c = cost[i, j]
        if i not in best or c < best[i][0]:
            best[i] = (c, steps[j])
    return {i: s for i, (_, s) in best.items()}


def evaluate_run(
    run_dir: str | Path,
    *,
    sequence: str | None = None,
    manifest: str | Path | None = None,
    run_log: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    meta = (
        json.loads((run_dir / "metadata.json").read_text())
        if (run_dir / "metadata.json").exists()
        else {}
    )
    if run_log:
        sequence = sequence or run_log["sequence"]
        manifest = manifest or run_log.get("manifest")
    manifest = (
        Path(manifest) if manifest else (Path(meta["keyframes"]) if meta.get("keyframes") else None)
    )
    if sequence is None:
        sequence = meta.get("sequence", "").split("__")[0]
    gt_steps = {s.step: s for s in get_sequence(sequence)}

    checkpoints = load_checkpoints(run_dir)
    f2s = frame_to_step_map(run_dir, manifest)

    rows: list[dict[str, Any]] = []
    preds: list[dict[str, Any]] = []
    refs: list[dict[str, Any]] = []
    conv_errors: dict[int, str] = {}
    empty = {
        "vertices_coords": [],
        "edges_vertices": [],
        "edges_assignment": [],
        "faces_vertices": [],
    }
    converted: dict[int, dict[str, Any]] = {}
    for frame_n in sorted(checkpoints):
        try:
            converted[frame_n] = fu.cp_from_raw_representation(checkpoints[frame_n])
        except Exception as exc:  # the checkpoint could not be rebuilt in the simulator
            conv_errors[frame_n] = f"{type(exc).__name__}: {exc}"
            converted[frame_n] = dict(empty)
    # Automatic keyframes carry no step labels: align checkpoints to GT steps by DTW on CPD.
    auto_mode = bool(converted) and all(f2s.get(n) is None for n in converted)
    if auto_mode:
        order = sorted(converted)
        mapping = auto_align_frames([converted[n] for n in order], gt_steps)
        f2s = {n: mapping.get(i) for i, n in enumerate(order)}
    for frame_n in sorted(converted):
        step = f2s.get(frame_n)
        if step is None or step not in gt_steps:
            continue
        preds.append(converted[frame_n])
        refs.append(gt_steps[step].fold)
        rows.append({"frame": frame_n, "step": step})

    ff_pred = compile_many(preds) if preds else []
    ff_ref = compile_many(refs) if refs else []
    for row, pred, ref, fp, fr in zip(rows, preds, refs, ff_pred, ff_ref, strict=True):
        row["cv"] = int(bool(fp.get("compilation_success")))
        row["cv_reason"] = fp.get("reason", "")
        row["cv_ref"] = int(bool(fr.get("compilation_success")))
        row.update(all_similarities(pred, ref, fp, fr))
        row.update(crease_pattern_dissimilarity(pred, ref).as_dict())
        if row["frame"] in conv_errors:
            row["conversion_error"] = conv_errors[row["frame"]]

    n_steps = len(gt_steps)
    aligned_steps = sorted({s for s in f2s.values() if s is not None})
    if auto_mode:
        aligned_steps = sorted(gt_steps)  # nothing was aligned by hand; coverage is of all GT steps
    result: dict[str, Any] = {
        "run_dir": str(run_dir),
        "sequence": sequence,
        "condition": (run_log or {}).get("condition"),
        "video_id": (run_log or {}).get("video_id"),
        "model": meta.get("model"),
        "stop_reason": meta.get("stop_reason"),
        "num_gt_steps": n_steps,
        "num_aligned_steps": len(aligned_steps),
        "num_solved_steps": len(rows),
        "alignment": "auto_dtw" if auto_mode else "manual",
        "completed": bool(meta.get("stop_reason") == "all_frames_solved"),
        "completion_rate": (len(rows) / len(aligned_steps)) if aligned_steps else 0.0,
        "coverage_of_gt": (len(rows) / n_steps) if n_steps else 0.0,
        "wall_time_s": (run_log or {}).get("wall_time_s", meta.get("elapsed_seconds")),
        "cost_usd_estimate": (run_log or {}).get("cost_usd_estimate"),
        "tokens": meta.get("tokens"),
        "tool_calls_total": (run_log or {}).get("tool_calls_total"),
        "per_step": rows,
    }
    if rows:
        df = pd.DataFrame(rows)
        last = rows[-1]
        result["cv"] = float(df["cv"].mean())
        for k in ("tss", "gs", "cs", "ffs", "cpd"):
            result[f"{k}_final"] = float(last[k])
            result[f"{k}_mean"] = float(df[k].mean())
        result["final_step"] = last["step"]
    return result


def write_metrics(result: dict[str, Any], *, results_root: Path = RESULTS_DIR) -> Path:
    cond = result.get("condition") or "unknown"
    out_dir = results_root / cond / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (Path(result["run_dir"]).name + ".json")
    path.write_text(json.dumps(result, indent=2) + "\n")
    return path


def collect_metrics(results_root: Path = RESULTS_DIR) -> pd.DataFrame:
    recs = []
    for path in sorted(results_root.glob("*/metrics/*.json")):
        r = json.loads(path.read_text())
        r.pop("per_step", None)
        r.pop("tokens", None)
        recs.append(r)
    return pd.DataFrame(recs)


COMPARE_COLS = [
    "completed",
    "completion_rate",
    "cv",
    "tss_final",
    "gs_final",
    "cs_final",
    "ffs_final",
    "cpd_final",
    "wall_time_s",
    "cost_usd_estimate",
    "tool_calls_total",
]


def summarize(results_root: Path = RESULTS_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-run table, PurelandFold-vs-YouTube table per sequence + average)."""
    df = collect_metrics(results_root)
    if df.empty:
        return df, df
    df["completed"] = df["completed"].astype(float)
    keep = [c for c in COMPARE_COLS if c in df.columns]
    per = df.groupby(["sequence", "condition"])[keep].mean(numeric_only=True)
    avg = df.groupby("condition")[keep].mean(numeric_only=True)
    avg.index = pd.MultiIndex.from_product(
        [["__average__"], avg.index], names=["sequence", "condition"]
    )
    table = pd.concat([per, avg])
    (results_root / "summary.csv").write_text(table.to_csv())
    try:
        md = table.round(3).to_markdown()
    except ImportError:  # tabulate missing: plain text is fine
        md = table.round(3).to_string()
    (results_root / "summary.md").write_text(md + "\n")
    return df, table
