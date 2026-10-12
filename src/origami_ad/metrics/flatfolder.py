"""Compilation Validity (CV): run Flat-Folder headlessly on FOLD dicts.

Flat-Folder (Ku 2022) is a JavaScript crease-pattern solver. ``scripts/flatfolder_compile.mjs``
imports its modules from the ``external/flat-folder`` submodule under Node and reproduces
its batch pipeline. Results are cached by content hash under ``data/cache/flatfolder``.

``CV1(state) = 1`` iff ``compilation_success`` is true.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from origami_ad.paths import DATA_DIR, SCRIPTS_DIR

COMPILE_SCRIPT = SCRIPTS_DIR / "flatfolder_compile.mjs"
CACHE_DIR = DATA_DIR / "cache" / "flatfolder"
DEFAULT_STATE_LIMIT = 1000

_CANONICAL_KEYS = ("vertices_coords", "edges_vertices", "edges_assignment", "faces_vertices")


def _canonical(fold: dict[str, Any]) -> dict[str, Any]:
    """Only the fields Flat-Folder reads, so caching ignores metadata."""
    return {k: fold[k] for k in _CANONICAL_KEYS if k in fold}


def _digest(fold: dict[str, Any], limit: int) -> str:
    text = json.dumps(
        {"fold": _canonical(fold), "limit": limit}, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha1(text.encode()).hexdigest()


def node_available() -> bool:
    return shutil.which("node") is not None


def compile_many(
    folds: list[dict[str, Any]],
    *,
    limit: int = DEFAULT_STATE_LIMIT,
    timeout_s: float = 600.0,
    use_cache: bool = True,
) -> list[dict[str, Any]]:
    """Compile several states in one Node process. Order is preserved."""
    results: list[dict[str, Any] | None] = [None] * len(folds)
    todo: list[int] = []
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    digests = [_digest(f, limit) for f in folds]
    for i, d in enumerate(digests):
        cached = CACHE_DIR / f"{d}.json"
        if use_cache and cached.exists():
            results[i] = json.loads(cached.read_text())
        else:
            todo.append(i)
    if todo:
        if not node_available():
            raise RuntimeError("node is required for Flat-Folder compilation (not found on PATH)")
        with tempfile.TemporaryDirectory(prefix="flatfolder_") as tmp:
            tmpdir = Path(tmp)
            inputs = []
            for i in todo:
                p = tmpdir / f"{i:05d}.fold"
                p.write_text(json.dumps(_canonical(folds[i])))
                inputs.append(str(p))
            out = tmpdir / "out.json"
            manifest = tmpdir / "manifest.json"
            manifest.write_text(json.dumps({"inputs": inputs, "output": str(out)}))
            proc = subprocess.run(
                ["node", str(COMPILE_SCRIPT), "--limit", str(limit), str(manifest)],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            if proc.returncode != 0 or not out.exists():
                raise RuntimeError(f"flatfolder_compile.mjs failed: {proc.stderr[-2000:]}")
            recs = json.loads(out.read_text())
        for i, rec in zip(todo, recs, strict=True):
            rec.pop("input", None)
            results[i] = rec
            (CACHE_DIR / f"{digests[i]}.json").write_text(json.dumps(rec))
    return [r for r in results if r is not None]


def compile_fold(fold: dict[str, Any], **kw: Any) -> dict[str, Any]:
    return compile_many([fold], **kw)[0]


def compilation_validity(result: dict[str, Any]) -> int:
    return int(bool(result.get("compilation_success")))
