"""Score automatic keyframes against hand-labelled fold states.

``data/annotations/keyframe_labels.json`` lists, per video, the *required* windows in
which each fold state is visible (finished, before the next manipulation) and *optional*
windows (re-shows, colouring, in-hand shaping). An optional window may name the required
state it re-shows. A proposed timestamp is

* ``H`` a hit: the first pick inside a required window (``h``: via a linked re-show);
* ``D`` a duplicate: another pick of a state already hit, or a second pick in one
  optional window;
* ``o`` neutral: the first pick in an unlinked optional window;
* ``X`` an extra: outside every window (mid-fold frame, title card, end screen).

Recall = hits / required states, precision = hits / (hits + extras + duplicates).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from origami_ad.paths import ANNOTATIONS_DIR

LABELS_JSON = ANNOTATIONS_DIR / "keyframe_labels.json"


def load_labels(path: Path = LABELS_JSON) -> dict[str, dict[str, Any]]:
    return json.loads(path.read_text())["videos"]


@dataclass
class Score:
    n: int
    required: int
    hits: int
    extras: int
    duplicates: int
    detail: str
    missed: list[int]

    @property
    def bad(self) -> int:
        return self.extras + self.duplicates


def score(timestamps: list[float], label: dict[str, Any], tol: float = 0.5) -> Score:
    req = [(a - tol, b + tol) for a, b in label["required"]]
    opt = [(o[0] - tol, o[1] + tol, o[2] - 1 if len(o) > 2 else None) for o in label["optional"]]
    hit: set[int] = set()
    used_opt: set[int] = set()
    extras = dups = 0
    detail = []
    for t in timestamps:
        in_req = [i for i, (a, b) in enumerate(req) if a <= t <= b]
        in_opt = [i for i, (a, b, _) in enumerate(opt) if a <= t <= b]
        new = [i for i in in_req if i not in hit]
        link = [opt[i][2] for i in in_opt if opt[i][2] is not None]
        if new:
            hit.add(new[0])
            detail.append("H")
        elif in_req:
            dups += 1
            detail.append("D")
        elif link and link[0] not in hit:
            hit.add(link[0])
            detail.append("h")
        elif link:
            dups += 1
            detail.append("D")
        elif in_opt and in_opt[0] not in used_opt:
            used_opt.add(in_opt[0])
            detail.append("o")
        elif in_opt:
            dups += 1
            detail.append("D")
        else:
            extras += 1
            detail.append("X")
    return Score(
        n=len(timestamps),
        required=len(req),
        hits=len(hit),
        extras=extras,
        duplicates=dups,
        detail="".join(detail),
        missed=[i + 1 for i in range(len(req)) if i not in hit],
    )


def totals(scores: dict[str, Score]) -> dict[str, float]:
    hits = sum(s.hits for s in scores.values())
    req = sum(s.required for s in scores.values())
    bad = sum(s.bad for s in scores.values())
    recall = hits / req if req else 0.0
    precision = hits / (hits + bad) if hits + bad else 0.0
    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    return {
        "hits": hits,
        "required": req,
        "bad": bad,
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }
