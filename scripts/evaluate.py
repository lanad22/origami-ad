#!/usr/bin/env python3
"""Re-evaluate existing run directories and rebuild results/summary.{csv,md}.

python scripts/evaluate.py                       # every run log under results/*/logs
python scripts/evaluate.py results/purelandfold/runs/run_..._yacht --sequence yacht
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _common  # noqa: F401
from origami_ad.eval import evaluate_run, summarize, write_metrics
from origami_ad.paths import RESULTS_DIR


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("run_dir", nargs="?")
    p.add_argument("--sequence")
    a = p.parse_args()
    if a.run_dir:
        res = evaluate_run(a.run_dir, sequence=a.sequence)
        res["condition"] = res.get("condition") or Path(a.run_dir).resolve().parent.parent.name
        print(write_metrics(res))
    else:
        for log_path in sorted(RESULTS_DIR.glob("*/logs/*.json")):
            log = json.loads(log_path.read_text())
            if not Path(log["run_dir"]).exists():
                continue
            res = evaluate_run(log["run_dir"], run_log=log)
            print(write_metrics(res))
    _, table = summarize()
    print(table.round(3).to_string() if not table.empty else "no metrics yet")


if __name__ == "__main__":
    main()
