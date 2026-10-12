"""PurelandFold access: download/cache the HF dataset, list sequences, fetch states.

The dataset has one row per keyframe with columns ``sequence``, ``step``, ``image``,
``cp.svg``, ``cp.fold`` (FOLD JSON of the crease pattern with ``faceOrders``) and
``flat_folder`` (the authors' Flat-Folder compile output for that state).

Run ``python -m origami_ad.dataset`` to print every sequence with its step count.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from PIL import Image

from origami_ad.paths import PURELANDFOLD_DIR, load_env

HF_DATASET_ID = "mayaweiz/PurelandFold"


@dataclass(frozen=True)
class Step:
    """One keyframe of a PurelandFold sequence."""

    sequence: str
    step: int
    row_index: int

    @property
    def image(self) -> Image.Image:
        return _dataset()[self.row_index]["image"]

    @property
    def fold(self) -> dict[str, Any]:
        return json.loads(_dataset()[self.row_index]["cp.fold"])

    @property
    def svg(self) -> str:
        return _dataset()[self.row_index]["cp.svg"]

    @property
    def flat_folder(self) -> dict[str, Any]:
        return json.loads(_dataset()[self.row_index]["flat_folder"])


@lru_cache(maxsize=1)
def _dataset():
    load_env()
    from datasets import load_dataset

    PURELANDFOLD_DIR.mkdir(parents=True, exist_ok=True)
    return load_dataset(HF_DATASET_ID, split="train", cache_dir=str(PURELANDFOLD_DIR / "hf_cache"))


@lru_cache(maxsize=1)
def _index() -> dict[str, list[Step]]:
    ds = _dataset()
    by_seq: dict[str, list[Step]] = {}
    for i, (seq, step) in enumerate(zip(ds["sequence"], ds["step"], strict=True)):
        by_seq.setdefault(seq, []).append(Step(seq, int(step), i))
    for steps in by_seq.values():
        steps.sort(key=lambda s: s.step)
    return by_seq


def list_sequences() -> list[str]:
    """Sequence names in dataset order."""
    return list(_index().keys())


def get_sequence(name: str) -> list[Step]:
    """All steps of a sequence, sorted by step number."""
    try:
        return list(_index()[name])
    except KeyError as exc:
        raise KeyError(f"Unknown sequence {name!r}; known: {list_sequences()}") from exc


def get_step(name: str, step: int) -> Step:
    for s in get_sequence(name):
        if s.step == step:
            return s
    raise KeyError(f"{name} has no step {step}")


def get_state(name: str, step: int) -> dict[str, Any]:
    """Parsed ``cp.fold`` for one keyframe."""
    return get_step(name, step).fold


def num_steps(name: str) -> int:
    return len(get_sequence(name))


def display_name(name: str) -> str:
    """``gift_card_holder`` -> ``gift card holder``; version suffixes are dropped."""
    words = [w for w in name.split("_") if not (w.startswith("v") and w[1:].isdigit())]
    return " ".join(words)


def main() -> None:
    seqs = list_sequences()
    total = 0
    for name in seqs:
        n = num_steps(name)
        total += n
        print(f"{name:20s} {n:3d} steps")
    print(f"{len(seqs)} sequences, {total} keyframes")


if __name__ == "__main__":
    main()
