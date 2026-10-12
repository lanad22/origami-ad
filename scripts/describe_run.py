#!/usr/bin/env python3
"""Print a run's fold program as plain-language steps (and save description.md next to it).

python scripts/describe_run.py results/purelandfold/runs/run_..._yacht
python scripts/describe_run.py --latest            # most recent run under results/
"""

from __future__ import annotations

import argparse

import _common  # noqa: F401
from origami_ad.describe import describe_run
from origami_ad.paths import RESULTS_DIR


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("run_dir", nargs="?")
    p.add_argument("--latest", action="store_true")
    a = p.parse_args()
    if a.latest or not a.run_dir:
        runs = sorted(RESULTS_DIR.glob("*/runs/run_*"), key=lambda d: d.stat().st_mtime)
        if not runs:
            raise SystemExit("no runs under results/")
        a.run_dir = str(runs[-1])
    print(a.run_dir)
    print(describe_run(a.run_dir))


if __name__ == "__main__":
    main()
