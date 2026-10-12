#!/usr/bin/env python3
"""T2: YouTube coverage audit. Needs YOUTUBE_API_KEY in .env.

python scripts/coverage_audit.py                 # all 27 sequences (~8,100 quota units)
python scripts/coverage_audit.py --sequence ladybug --sequence heart
python scripts/coverage_audit.py --summary       # just print the per-sequence table
"""

from __future__ import annotations

import argparse

import _common  # noqa: F401
from origami_ad.audit.search import (
    AUDIT_CSV,
    MATCHES_CSV,
    load_audit,
    print_summary,
    run_audit,
    write_matches,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--sequence", action="append", help="restrict to these sequences (repeatable)")
    p.add_argument("--max-results", type=int, default=10)
    p.add_argument(
        "--summary", action="store_true", help="only print the summary of the existing CSV"
    )
    a = p.parse_args()
    if a.summary:
        print_summary(load_audit())
        return
    rows = run_audit(a.sequence, max_results=a.max_results)
    print(f"wrote {AUDIT_CSV} ({len(rows)} rows)")
    print_summary(rows)


if __name__ == "__main__":
    main()
