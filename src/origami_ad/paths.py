"""Repository paths and environment loading, shared by every module."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
PURELANDFOLD_DIR = DATA_DIR / "purelandfold"
YOUTUBE_DIR = DATA_DIR / "youtube"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
ALIGNMENT_DIR = ANNOTATIONS_DIR / "alignment"
RESULTS_DIR = REPO_ROOT / "results"
EXTERNAL_DIR = REPO_ROOT / "external"
FOLDINGAGENT_DIR = EXTERNAL_DIR / "FoldingAgent"
FLATFOLDER_DIR = EXTERNAL_DIR / "flat-folder"
SCRIPTS_DIR = REPO_ROOT / "scripts"


def load_env() -> None:
    """Load ``.env`` from the repo root and mirror GEMINI_API_KEY to GOOGLE_API_KEY.

    FoldingAgent reads ``GOOGLE_API_KEY`` for gemini models; our ``.env.example``
    offers ``GEMINI_API_KEY`` as well, so either name works.
    """
    load_dotenv(REPO_ROOT / ".env")
    if os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]


def require_env(name: str) -> str:
    load_env()
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and fill it in (never commit .env)."
        )
    return value
