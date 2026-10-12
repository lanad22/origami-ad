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


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--sequence", action="append")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-iterations", type=int, default=None)
    p.add_argument("--skip-eval", action="store_true")
    a = p.parse_args()
    for seq in a.sequence or DEFAULT_SEQUENCES:
        spec = build_purelandfold_manifest(seq)
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
