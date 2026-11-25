#!/usr/bin/env python3
"""T1: run FoldingAgent on PurelandFold sequences and evaluate.

Defaults to one short, one medium and one long sequence (yacht 5, ladybug 13, snake 21).
Needs the API key for the chosen model (see .env.example).

    python scripts/reproduce.py                          # gemini-3.1-pro-preview on the 3 defaults
    python scripts/reproduce.py --model claude-opus-5-5 --sequence yacht
    python scripts/reproduce.py --max-iterations 50 --sequence yacht   # cheap smoke run
"""

from __future__ import annotations

import argparse
import json

import _common  # noqa: F401
from origami_ad.describe import describe_run
from origami_ad.eval import evaluate_run, summarize, write_metrics
from origami_ad.run_agent import DEFAULT_MODEL, build_purelandfold_manifest, run

DEFAULT_SEQUENCES = ["yacht", "ladybug", "snake"]


def _parse_steps(text: str) -> list[int]:
    """'1-5,7,9-10' -> [1, 2, 3, 4, 5, 7, 9, 10]."""
    out: list[int] = []
    for part in text.split(","):
        lo, _, hi = part.strip().partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return sorted(set(out))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--sequence", action="append")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-iterations", type=int, default=None)
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument(
        "--steps",
        help="keyframe subset, e.g. 1-5,7-12,14,16,17 (results go to results/<--condition>/)",
    )
    p.add_argument("--condition", help="results folder for a --steps subset run")
    a = p.parse_args()
    steps = _parse_steps(a.steps) if a.steps else None
    if steps and not a.condition:
        p.error("--steps needs --condition so subset runs stay apart from full benchmark runs")
    for seq in a.sequence or DEFAULT_SEQUENCES:
        spec = build_purelandfold_manifest(
            seq, steps=steps, condition=a.condition or "purelandfold"
        )
        spec.model = a.model
        print(f"\n=== {seq}: {len(spec.keyframes)} keyframes, model {a.model} ===")
        log = run(spec, max_iterations=a.max_iterations)
        print(
            json.dumps(
                {
                    k: log[k]
                    for k in (
                        "stop_reason",
                        "frames_solved",
                        "wall_time_s",
                        "tokens",
                        "cost_usd_estimate",
                    )
                },
                indent=1,
            )
        )
        if not a.skip_eval:
            print(describe_run(log["run_dir"]))
            res = evaluate_run(log["run_dir"], run_log=log)
            path = write_metrics(res)
            print(
                f"metrics -> {path}: CV={res.get('cv')} "
                f"TSS={res.get('tss_final')} CPD={res.get('cpd_final')}"
            )
    _, table = summarize()
    if not table.empty:
        print(table.round(3).to_string())


if __name__ == "__main__":
    main()
