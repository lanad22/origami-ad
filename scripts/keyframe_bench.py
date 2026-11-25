#!/usr/bin/env python3
"""Compare keyframing methods on the hand-labelled tutorials (keyframe_labels.json).

python scripts/keyframe_bench.py                         # all methods, all labelled videos
python scripts/keyframe_bench.py --method stillness      # no API calls
python scripts/keyframe_bench.py --method pairwise --split extra --sheets   # heart only
python scripts/keyframe_bench.py --split test --sheets   # also save contact sheets

Methods: stillness (keyframing), and keyframe_select with each review pass: pick (pick pass
only), grid (previous default) and pairwise (default). Writes
results/keyframing/<method>.json (timestamps + per-video scores; for pairwise also every
verdict and reason). Contact sheets go to data/keyframes/bench/<method>/<video>.jpg.
keyframe_select calls Claude only for images it has not seen before (responses are cached,
and pick / grid / pairwise share the pick pass): about $0.10 per new video for the pick
pass, $0.15-0.25 more for the pairwise review.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import _common  # noqa: F401
from origami_ad.keyframe_bench import load_labels, score, totals
from origami_ad.keyframe_select import SelectParams, select_keyframes
from origami_ad.keyframing import analyse_video, contact_sheet
from origami_ad.paths import DATA_DIR, RESULTS_DIR
from origami_ad.run_agent import estimate_cost
from origami_ad.video import download

METHODS = {
    "stillness": lambda path: analyse_video(path),
    "pick": lambda path: select_keyframes(path, SelectParams(review="none")),
    "grid": lambda path: select_keyframes(path, SelectParams(review="grid")),
    "pairwise": lambda path: select_keyframes(path, SelectParams(review="pairwise")),
}
SPLITS = ("dev", "test", "extra")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--method", action="append", choices=list(METHODS))
    p.add_argument("--split", choices=SPLITS)
    p.add_argument("--video", action="append")
    p.add_argument("--sheets", action="store_true")
    a = p.parse_args()

    labels = load_labels()
    vids = [
        v
        for v, lab in labels.items()
        if (not a.split or lab["split"] == a.split) and (not a.video or v in a.video)
    ]
    out_dir = RESULTS_DIR / "keyframing"
    out_dir.mkdir(parents=True, exist_ok=True)
    for method in a.method or list(METHODS):
        out = out_dir / f"{method}.json"
        rows = json.loads(out.read_text()) if out.exists() else {}  # keep other videos' rows
        scores, tokens = {}, [0, 0]
        for vid in vids:
            path = download(vid)
            an = METHODS[method](path)
            sel = an.params.get("selection", {})
            tokens[0] += sel.get("new_input_tokens", 0)
            tokens[1] += sel.get("new_output_tokens", 0)
            s = scores[vid] = score(an.timestamps, labels[vid])
            rows[vid] = {"split": labels[vid]["split"], "timestamps": an.timestamps}
            if "pairwise" in sel:
                rows[vid] |= {"picked_s": sel["picked_s"], "pairwise": sel["pairwise"]}
            rows[vid] |= asdict(s)
            print(
                f"{method:9s} {vid} [{labels[vid]['split']}] {s.hits}/{s.required} states, "
                f"{s.bad} bad  {s.detail}  missed {s.missed}"
            )
            if a.sheets:
                sheet = DATA_DIR / "keyframes" / "bench" / method / f"{vid}.jpg"
                contact_sheet(path, an.timestamps, sheet)
        for split in SPLITS:
            part = {v: s for v, s in scores.items() if labels[v]["split"] == split}
            if part:
                t = totals(part)
                print(
                    f"{method:9s} {split:5s}: recall {t['hits']}/{t['required']} = "
                    f"{t['recall']:.2f}, bad picks {t['bad']}, precision {t['precision']:.2f}, "
                    f"F1 {t['f1']:.3f}"
                )
        if any(tokens):
            cost, _ = estimate_cost(SelectParams().model, *tokens)
            print(f"{method:9s} new tokens: {tokens[0]} in / {tokens[1]} out, ~${cost}")
        out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
