"""Shared argparse helpers for the thin CLI entry points."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
