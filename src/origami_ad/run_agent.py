"""Run FoldingAgent on PurelandFold keyframes or on aligned YouTube keyframes (T1, T6).

FoldingAgent is driven through its public ``run_agent(manifest, backend=...)``. A run
needs a directory laid out as::

    <dir>/keyframes.json      {"name", "frames", "front_color", "back_color", "keyframes": [idx...]}
    <frames dir>/000012.jpg   frames named by global index (the manifest's ``frames`` key)

* PurelandFold: frames are the dataset keyframes, indexed by step number. There are no
  in-between frames, so the agent's ``observe_movement`` tool degrades to a two-frame
  filmstrip (it needs >= 2 frames in range, which the two keyframes provide).
* YouTube (Condition A): frames are the fixed-fps extraction from ``origami_ad.video``;
  keyframe indices come from the manual alignment timestamps, so ``observe_movement``
  sees real intermediate frames.

Every run writes a JSON log under ``results/<condition>/logs/`` with model, sequence,
video id, keyframe count, tool-call counts, tokens, estimated cost, wall time and
success flags, alongside FoldingAgent's own run directory under ``results/<condition>/runs/``.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from origami_ad.align import Alignment
from origami_ad.dataset import get_sequence
from origami_ad.paths import ANNOTATIONS_DIR, DATA_DIR, RESULTS_DIR, load_env
from origami_ad.video import DEFAULT_FPS, download, extract_frames, timestamp_to_index, video_dir

KEYFRAMES_ROOT = DATA_DIR / "keyframes"
PAPER_COLORS_JSON = ANNOTATIONS_DIR / "paper_colors.json"
PRICING_JSON = ANNOTATIONS_DIR / "pricing.json"
DEFAULT_MODEL = "gemini-3.1-pro-preview"  # the paper's backbone


@dataclass
class KeyframeMeta:
    """How a manifest's frame indices map back to PurelandFold steps."""

    frame_index: int
    step: int | None  # None for extra steps with no PurelandFold counterpart
    timestamp_s: float | None = None
    notes: str = ""


@dataclass
class RunSpec:
    sequence: str
    condition: str  # "purelandfold" | "youtube_A_manual" | "youtube_B_hamer"
    manifest: Path
    keyframes: list[KeyframeMeta]
    video_id: str | None = None
    model: str = DEFAULT_MODEL
    extra: dict[str, Any] = field(default_factory=dict)


# ── Paper colours ─────────────────────────────────────────────────────────────


def paper_colors(sequence: str) -> tuple[str, str]:
    """(front, back) colour names for a PurelandFold sequence, from paper_colors.json."""
    table: dict[str, Any] = {}
    if PAPER_COLORS_JSON.exists():
        table = json.loads(PAPER_COLORS_JSON.read_text())
    entry = table.get(sequence) or table.get("_default", {})
    return entry.get("front_color", "white"), entry.get("back_color", "lightblue")


# ── Manifest builders ─────────────────────────────────────────────────────────


def _write_manifest(
    directory: Path,
    *,
    name: str,
    frames_rel: str,
    indices: list[int],
    front_color: str,
    back_color: str,
    metas: list[KeyframeMeta],
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "keyframes.json"
    manifest.write_text(
        json.dumps(
            {
                "name": name,
                "frames": frames_rel,
                "front_color": front_color,
                "back_color": back_color,
                "keyframes": indices,
            },
            indent=2,
        )
        + "\n"
    )
    (directory / "keyframes_meta.json").write_text(
        json.dumps([asdict(m) for m in metas], indent=2) + "\n"
    )
    return manifest


def build_purelandfold_manifest(sequence: str, *, root: Path = KEYFRAMES_ROOT) -> RunSpec:
    """Dataset keyframes -> ``data/keyframes/purelandfold/<sequence>/`` (gitignored)."""
    directory = root / "purelandfold" / sequence
    frames_dir = directory / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    metas: list[KeyframeMeta] = []
    indices: list[int] = []
    for step in get_sequence(sequence):
        path = frames_dir / f"{step.step:06d}.jpg"
        if not path.exists():
            step.image.convert("RGB").save(path, quality=92)
        indices.append(step.step)
        metas.append(KeyframeMeta(frame_index=step.step, step=step.step))
    front, back = paper_colors(sequence)
    manifest = _write_manifest(
        directory,
        name=sequence,
        frames_rel="frames",
        indices=indices,
        front_color=front,
        back_color=back,
        metas=metas,
    )
    return RunSpec(sequence=sequence, condition="purelandfold", manifest=manifest, keyframes=metas)


def build_youtube_manifest(
    alignment: Alignment,
    *,
    fps: float = DEFAULT_FPS,
    root: Path = KEYFRAMES_ROOT,
    include_extra_steps: bool = False,
    condition: str = "youtube_A_manual",
) -> RunSpec:
    """Aligned YouTube timestamps -> manifest over the fixed-fps frame dump.

    Keyframes with ``timestamp_s < 0`` (not yet annotated) are skipped. Extra steps are
    excluded by default so every agent frame maps to a PurelandFold step.
    """
    vid = alignment.video_id
    download(vid)
    extract_frames(vid, fps=fps)
    directory = root / "youtube" / vid
    directory.mkdir(parents=True, exist_ok=True)
    frames_rel = os.path.relpath((video_dir(vid) / "frames").resolve(), directory.resolve())

    items: list[tuple[float, int | None, str]] = [
        (k.timestamp_s, k.step, k.notes) for k in alignment.keyframes if k.timestamp_s >= 0
    ]
    if include_extra_steps:
        items += [(e.timestamp_s, None, e.notes) for e in alignment.extra_steps]
    items.sort(key=lambda t: t[0])

    metas: list[KeyframeMeta] = []
    indices: list[int] = []
    for ts, step, notes in items:
        idx = timestamp_to_index(ts, fps)
        if indices and idx <= indices[-1]:
            idx = indices[-1] + 1  # keep indices strictly increasing at low fps
        indices.append(idx)
        metas.append(KeyframeMeta(frame_index=idx, step=step, timestamp_s=ts, notes=notes))
    if not indices:
        raise ValueError(f"alignment {vid} has no annotated keyframes")

    seq_front, seq_back = paper_colors(alignment.sequence)
    manifest = _write_manifest(
        directory,
        name=f"{alignment.sequence}__{vid}",
        frames_rel=frames_rel,
        indices=indices,
        front_color=alignment.front_color or seq_front,
        back_color=alignment.back_color or seq_back,
        metas=metas,
    )
    return RunSpec(
        sequence=alignment.sequence,
        condition=condition,
        manifest=manifest,
        keyframes=metas,
        video_id=vid,
        extra={"fps": fps, "design_match": alignment.design_match},
    )


def build_auto_manifest(
    video_id: str,
    sequence: str,
    *,
    root: Path = KEYFRAMES_ROOT,
    params: Any | None = None,
    front_color: str | None = None,
    back_color: str | None = None,
    condition: str = "youtube_B_auto",
) -> RunSpec:
    """Automatic keyframes (Condition B): stillness + hand mask, no manual timestamps.

    Frames are extracted at the analysis rate so every proposal maps to an exact frame.
    Steps are unknown here (``KeyframeMeta.step = None``); ``eval`` aligns the agent's
    checkpoints to PurelandFold steps afterwards by dynamic time warping on CPD.
    """
    from origami_ad.keyframing import KeyframeParams, analyse_video, contact_sheet

    p = params or KeyframeParams()
    path = download(video_id)
    analysis = analyse_video(path, p)
    fps = analysis.sample_fps
    extract_frames(video_id, fps=fps)
    directory = root / "youtube_auto" / video_id
    directory.mkdir(parents=True, exist_ok=True)
    analysis.save(directory / "keyframes_auto.json")
    contact_sheet(path, analysis.timestamps, directory / "keyframes_auto.jpg")
    frames_rel = os.path.relpath((video_dir(video_id) / "frames").resolve(), directory.resolve())

    metas: list[KeyframeMeta] = []
    indices: list[int] = []
    for prop in analysis.proposals:
        idx = timestamp_to_index(prop.timestamp_s, fps)
        if indices and idx <= indices[-1]:
            idx = indices[-1] + 1
        indices.append(idx)
        metas.append(KeyframeMeta(frame_index=idx, step=None, timestamp_s=prop.timestamp_s))
    if not indices:
        raise ValueError(f"no keyframes proposed for {video_id}")

    seq_front, seq_back = paper_colors(sequence)
    manifest = _write_manifest(
        directory,
        name=f"{sequence}__{video_id}__auto",
        frames_rel=frames_rel,
        indices=indices,
        front_color=front_color or seq_front,
        back_color=back_color or seq_back,
        metas=metas,
    )
    return RunSpec(
        sequence=sequence,
        condition=condition,
        manifest=manifest,
        keyframes=metas,
        video_id=video_id,
        extra={"fps": fps, "keyframing": "auto", "params": analysis.params},
    )


# ── Running ───────────────────────────────────────────────────────────────────


def make_backend(model: str):
    """A FoldingAgent backend for *model*; provider follows the model id prefix."""
    from foldingagent.backends.base import _provider_for_model
    from foldingagent.main import AnthropicBackend, GeminiBackend, OpenAIBackend

    provider = _provider_for_model(model)
    if provider == "anthropic":
        return AnthropicBackend(model=model)
    if provider == "openai":
        return OpenAIBackend(model=model)
    return GeminiBackend(model=model)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> tuple[float | None, str]:
    """USD estimate from ``pricing.json`` (longest matching model prefix), or None."""
    if not PRICING_JSON.exists():
        return None, "no pricing.json"
    table = json.loads(PRICING_JSON.read_text())
    best = None
    for prefix, p in table.items():
        if prefix.startswith("_"):
            continue
        if model.startswith(prefix) and (best is None or len(prefix) > len(best[0])):
            best = (prefix, p)
    if best is None or best[1].get("input_per_m") is None:
        return None, f"no pricing for {model}"
    p = best[1]
    cost = input_tokens / 1e6 * p["input_per_m"] + output_tokens / 1e6 * p["output_per_m"]
    return round(cost, 4), p.get("source", "pricing.json")


def _tool_call_counts(agent_log: Path) -> Counter:
    counts: Counter = Counter()
    if not agent_log.exists():
        return counts
    for line in agent_log.read_text().splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("event") == "tool_call":
            counts[ev.get("tool", "?")] += 1
    return counts


def summarize_run(run_dir: Path, spec: RunSpec, *, wall_s: float | None = None) -> dict[str, Any]:
    meta = (
        json.loads((run_dir / "metadata.json").read_text())
        if (run_dir / "metadata.json").exists()
        else {}
    )
    tokens = meta.get("tokens", {})
    tool_counts = _tool_call_counts(run_dir / "logs" / "agent_log.jsonl")
    frames_solved = meta.get("frames_solved", [])
    n_frames = len(spec.keyframes)
    cost, cost_source = estimate_cost(spec.model, tokens.get("input", 0), tokens.get("output", 0))
    return {
        "run_dir": str(run_dir),
        "condition": spec.condition,
        "sequence": spec.sequence,
        "video_id": spec.video_id,
        "model": spec.model,
        "manifest": str(spec.manifest),
        "num_keyframes": n_frames,
        "keyframes": [asdict(k) for k in spec.keyframes],
        "started": meta.get("started"),
        "finished": meta.get("finished"),
        "wall_time_s": wall_s if wall_s is not None else meta.get("elapsed_seconds"),
        "thinking_time_s": meta.get("thinking_time_seconds"),
        "iterations": meta.get("iterations"),
        "stop_reason": meta.get("stop_reason"),
        "frames_solved": frames_solved,
        "completed": len(frames_solved) == n_frames
        and meta.get("stop_reason") == "all_frames_solved",
        "completion_rate": (len(frames_solved) / n_frames) if n_frames else 0.0,
        "tool_calls_total": sum(tool_counts.values()),
        "tool_calls": dict(tool_counts),
        "tokens": tokens,
        "cost_usd_estimate": cost,
        "cost_source": cost_source,
        "config": meta.get("config"),
        "extra": spec.extra,
    }


def run(
    spec: RunSpec,
    *,
    results_root: Path = RESULTS_DIR,
    max_iterations: int | None = None,
    resume_run_dir: Path | None = None,
    backend: Any | None = None,
) -> dict[str, Any]:
    """Run FoldingAgent for *spec* and write the JSON log. Returns the log dict.

    *backend* overrides the model-derived backend (used by the scripted test backend).
    """
    load_env()
    from foldingagent import config as fa_config
    from foldingagent.main import run_agent as fa_run_agent

    runs_root = results_root / spec.condition / "runs"
    logs_root = results_root / spec.condition / "logs"
    runs_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    fa_config.OUT_DIR = str(runs_root)
    if max_iterations is not None:
        fa_config.MAX_ITERATIONS = max_iterations

    before = {p for p in runs_root.iterdir() if p.is_dir()}
    if backend is None:
        backend = make_backend(spec.model)
    else:
        spec.model = getattr(backend, "model", spec.model)
    t0 = time.time()
    try:
        fa_run_agent(spec.manifest, backend=backend, resume_run_dir=resume_run_dir)
    except KeyboardInterrupt:
        pass
    wall = time.time() - t0

    if resume_run_dir is not None:
        run_dir = Path(resume_run_dir)
    else:
        new = sorted(
            {p for p in runs_root.iterdir() if p.is_dir()} - before, key=lambda p: p.stat().st_mtime
        )
        if not new:
            raise RuntimeError(f"FoldingAgent produced no run directory under {runs_root}")
        run_dir = new[-1]

    summary = summarize_run(run_dir, spec, wall_s=round(wall, 1))
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{spec.sequence}" + (f"__{spec.video_id}" if spec.video_id else "")
    log_path = logs_root / f"{tag}__{spec.model}__{stamp}.json"
    log_path.write_text(json.dumps(summary, indent=2) + "\n")
    summary["log_path"] = str(log_path)
    return summary
