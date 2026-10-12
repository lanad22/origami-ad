"""Shared opening steps across PurelandFold sequences (T4).

Each state's crease pattern is reduced to a canonical signature that is invariant to
the square's 8 symmetries (rotations by 90 degrees and reflections) and to turning the
paper over (which swaps mountain and valley). Two states share a *base* when their
signatures are equal. Coordinates are snapped to a 0.01 grid before hashing, which
absorbs the ~1e-6 numerical noise in the dataset; this is a tolerance by quantisation,
so a crease sitting exactly on a snapping boundary could in principle split a base.

Output ``data/annotations/bases.json``::

    {"bases": [{"base_id": "base_003", "canonical_state": {...}, "num_sequences": 4,
                "sequences": [{"name": "ladybug", "last_step": 3, "steps": [3]}, ...]}],
     "prefix_lengths": {"ladybug|cat": 2, ...}}

``prefix_lengths`` is the number of leading steps two sequences share step for step; it
is what tells us how far a YouTube video of a model *outside* PurelandFold can be scored
against a known sequence when it opens with the same base.
"""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
from typing import Any

import numpy as np

from origami_ad.dataset import get_sequence, list_sequences
from origami_ad.metrics import fold_utils as fu
from origami_ad.paths import ANNOTATIONS_DIR

BASES_JSON = ANNOTATIONS_DIR / "bases.json"
GRID = 0.01

_SYMMETRIES = [
    lambda x, y: (x, y),
    lambda x, y: (1 - y, x),
    lambda x, y: (1 - x, 1 - y),
    lambda x, y: (y, 1 - x),
    lambda x, y: (1 - x, y),
    lambda x, y: (x, 1 - y),
    lambda x, y: (y, x),
    lambda x, y: (1 - y, 1 - x),
]
_SWAP = {"M": "V", "V": "M"}


def _snap(v: float) -> float:
    return round(round(float(v), 4) / GRID) * GRID


def segments(
    fold: dict[str, Any], *, include_boundary: bool = False
) -> list[tuple[float, float, float, float, str]]:
    v = fu.vertices(fold)
    out = []
    for (a, b), t in zip(fu.edges(fold), fu.assignments(fold), strict=True):
        if t == "B" and not include_boundary:
            continue
        if t == "U":
            continue
        out.append((float(v[a][0]), float(v[a][1]), float(v[b][0]), float(v[b][1]), t))
    return out


def _signature_under(segs, sym, swap: bool) -> tuple:
    items = []
    for x1, y1, x2, y2, t in segs:
        p = tuple(_snap(c) for c in sym(x1, y1))
        q = tuple(_snap(c) for c in sym(x2, y2))
        if q < p:
            p, q = q, p
        tt = _SWAP.get(t, t) if swap else t
        items.append((*p, *q, tt))
    return tuple(sorted(set(items)))


def canonical_signature(fold: dict[str, Any]) -> tuple:
    """Lexicographically smallest signature over symmetries x {paper flip}."""
    segs = segments(fold)
    best = None
    for sym in _SYMMETRIES:
        for swap in (False, True):
            sig = _signature_under(segs, sym, swap)
            if best is None or sig < best:
                best = sig
    return best or ()


def signature_key(sig: tuple) -> str:
    return (
        "|".join(",".join(f"{c:.2f}" if isinstance(c, float) else str(c) for c in s) for s in sig)
        or "<square>"
    )


def compute_bases(sequences: list[str] | None = None) -> dict[str, Any]:
    sequences = sequences or list_sequences()
    sigs: dict[str, list[str]] = {}
    representative: dict[str, tuple[str, int]] = {}
    by_key: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for name in sequences:
        keys = []
        for step in get_sequence(name):
            key = signature_key(canonical_signature(step.fold))
            keys.append(key)
            by_key[key][name].append(step.step)
            representative.setdefault(key, (name, step.step))
        sigs[name] = keys

    bases = []
    shared = [(k, s) for k, s in by_key.items() if len(s) >= 2]
    shared.sort(key=lambda ks: (-len(ks[1]), min(min(v) for v in ks[1].values())))
    for i, (key, seqs) in enumerate(shared):
        rep_name, rep_step = representative[key]
        rep_fold = next(s for s in get_sequence(rep_name) if s.step == rep_step).fold
        bases.append(
            {
                "base_id": f"base_{i:03d}",
                "signature": key,
                "canonical_state": {
                    "representative": {"sequence": rep_name, "step": rep_step},
                    "segments": [list(s) for s in segments(rep_fold)],
                    "num_creases": len(segments(rep_fold)),
                },
                "num_sequences": len(seqs),
                "sequences": [
                    {"name": n, "last_step": max(steps), "steps": sorted(steps)}
                    for n, steps in sorted(seqs.items())
                ],
            }
        )

    prefix: dict[str, int] = {}
    for a, b in combinations(sequences, 2):
        n = 0
        for ka, kb in zip(sigs[a], sigs[b], strict=False):
            if ka != kb:
                break
            n += 1
        prefix[f"{a}|{b}"] = n

    return {
        "grid": GRID,
        "num_sequences": len(sequences),
        "bases": bases,
        "prefix_lengths": prefix,
        "step_signatures": sigs,
    }


def save_bases(data: dict[str, Any], path=BASES_JSON) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1) + "\n")


def print_summary(data: dict[str, Any]) -> None:
    print(f"{len(data['bases'])} shared states across {data['num_sequences']} sequences")
    for b in data["bases"][:25]:
        rep = b["canonical_state"]["representative"]
        who = ", ".join(f"{s['name']}@{s['last_step']}" for s in b["sequences"])
        print(
            f"  {b['base_id']}  creases={b['canonical_state']['num_creases']:2d}  "
            f"rep={rep['sequence']}#{rep['step']}  n={b['num_sequences']}: {who}"
        )
    lengths = np.array(list(data["prefix_lengths"].values()))
    print(
        f"pairs sharing >=2 opening steps: {(lengths >= 2).sum()} / {len(lengths)}; "
        f">=3: {(lengths >= 3).sum()}; max {lengths.max()}"
    )


if __name__ == "__main__":
    d = compute_bases()
    save_bases(d)
    print_summary(d)
    print(f"wrote {BASES_JSON}")
