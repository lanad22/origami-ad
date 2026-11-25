#!/usr/bin/env python3
"""T6: run FoldingAgent on YouTube keyframes and evaluate.

Condition B, the default (automatic keyframes, no timestamps; steps aligned by DTW in eval):
    python scripts/run_youtube.py VIDEO_ID SEQUENCE [VIDEO_ID SEQUENCE ...]
    python scripts/run_youtube.py VIDEO_ID SEQUENCE --keyframes-only
    python scripts/run_youtube.py VIDEO_ID SEQUENCE --keyframer grid        # previous default
    python scripts/run_youtube.py VIDEO_ID SEQUENCE --keyframer stillness   # no API
    python scripts/run_youtube.py --matches [--sequence yacht]   # every exact match

Condition A (manual timestamps in data/annotations/alignment/, ablation only):
    python scripts/run_youtube.py --condition A VIDEO_ID [VIDEO_ID ...] [--fps 2]
    python scripts/run_youtube.py --condition A --all
"""

from __future__ import annotations

import argparse
import json

import _common  # noqa: F401
from origami_ad.align import Alignment, list_alignments
from origami_ad.audit.search import load_matches
from origami_ad.describe import describe_run
from origami_ad.eval import evaluate_run, summarize, write_metrics
from origami_ad.run_agent import (
    DEFAULT_MODEL,
    KEYFRAMERS,
    build_auto_manifest,
    build_youtube_manifest,
    run,
)
from origami_ad.video import DEFAULT_FPS


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("video_ids", nargs="*")
    p.add_argument("--all", action="store_true")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--fps", type=float, default=DEFAULT_FPS)
    p.add_argument("--max-iterations", type=int, default=None)
    p.add_argument("--condition", choices=["A", "B"], default="B")
    p.add_argument(
        "--matches", action="store_true", help="condition B: run every exact match in matches.csv"
    )
    p.add_argument(
        "--sequence", action="append", help="with --matches: restrict to these sequences"
    )
    p.add_argument("--include-extra-steps", action="store_true")
    p.add_argument(
        "--keyframes-only",
        action="store_true",
        help="condition B: propose keyframes, skip the agent",
    )
    p.add_argument(
        "--keyframer",
        choices=KEYFRAMERS,
        default=KEYFRAMERS[0],
        help="condition B: pairwise review (default), grid review or stillness",
    )
    a = p.parse_args()

    if a.condition == "B":
        if a.matches:
            pairs = [
                (m["video_id"], m["sequence"])
                for m in load_matches()
                if not a.sequence or m["sequence"] in a.sequence
            ]
            if not pairs:
                raise SystemExit("no exact matches in data/annotations/matches.csv yet")
        elif not a.video_ids or len(a.video_ids) % 2:
            raise SystemExit(
                "condition B takes VIDEO_ID SEQUENCE pairs, e.g. abc123 ladybug, or --matches"
            )
        else:
            pairs = list(zip(a.video_ids[::2], a.video_ids[1::2], strict=True))
        for vid, seq in pairs:
            spec = build_auto_manifest(vid, seq, keyframer=a.keyframer)
            spec.model = a.model
            sheet = spec.manifest.parent / "keyframes_auto.jpg"
            print(f"\n=== {seq} / {vid}: {len(spec.keyframes)} auto keyframes (sheet: {sheet}) ===")
            if a.keyframes_only:
                continue
            log = run(spec, max_iterations=a.max_iterations)
            keys = ("stop_reason", "frames_solved", "wall_time_s", "cost_usd_estimate")
            print(json.dumps({k: log[k] for k in keys}, indent=1))
            print(describe_run(log["run_dir"]))
            res = evaluate_run(log["run_dir"], run_log=log)
            print(f"metrics -> {write_metrics(res)}")
        _, table = summarize()
        if not table.empty:
            print(table.round(3).to_string())
        return

    als = list_alignments() if a.all else [Alignment.load(v) for v in a.video_ids]
    als = [al for al in als if any(k.timestamp_s >= 0 for k in al.keyframes)]
    if not als:
        raise SystemExit("no alignments with timestamps; see scripts/alignment.py")
    for al in als:
        spec = build_youtube_manifest(al, fps=a.fps, include_extra_steps=a.include_extra_steps)
        spec.model = a.model
        print(f"\n=== {al.sequence} / {al.video_id}: {len(spec.keyframes)} keyframes ===")
        log = run(spec, max_iterations=a.max_iterations)
        print(
            json.dumps(
                {
                    k: log[k]
                    for k in ("stop_reason", "frames_solved", "wall_time_s", "cost_usd_estimate")
                },
                indent=1,
            )
        )
        print(describe_run(log["run_dir"]))
        res = evaluate_run(log["run_dir"], run_log=log)
        print(f"metrics -> {write_metrics(res)}")
    _, table = summarize()
    if not table.empty:
        print(table.round(3).to_string())


if __name__ == "__main__":
    main()
