"""Alignment files: YouTube timestamps -> PurelandFold steps (T5).

One JSON per video at ``data/annotations/alignment/{video_id}.json``::

    {
      "video_id": "...", "sequence": "ladybug", "design_match": "exact",
      "front_color": "red", "back_color": "white",
      "keyframes": [{"step": 1, "timestamp_s": 4.2, "notes": ""}, ...],
      "skipped_steps": [], "extra_steps": [{"after_step": 5, "timestamp_s": 40.0, "notes": ""}]
    }

Keyframe timestamps are entered manually for now (Condition A). ``skipped_steps``
lists PurelandFold steps the video never shows; ``extra_steps`` lists folds the video
performs that have no PurelandFold counterpart. Steps in ``skipped_steps`` are left
unaligned rather than merged into a multi-action transition (open question in CLAUDE.md;
``eval`` reports them separately).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from origami_ad.paths import ALIGNMENT_DIR


@dataclass
class Keyframe:
    step: int
    timestamp_s: float
    notes: str = ""


@dataclass
class ExtraStep:
    after_step: int
    timestamp_s: float
    notes: str = ""


@dataclass
class Alignment:
    video_id: str
    sequence: str
    design_match: str = "exact"
    front_color: str | None = None
    back_color: str | None = None
    keyframes: list[Keyframe] = field(default_factory=list)
    skipped_steps: list[int] = field(default_factory=list)
    extra_steps: list[ExtraStep] = field(default_factory=list)
    notes: str = ""

    @property
    def path(self) -> Path:
        return ALIGNMENT_DIR / f"{self.video_id}.json"

    def validate(self) -> list[str]:
        problems: list[str] = []
        ts = [k.timestamp_s for k in self.keyframes]
        if ts != sorted(ts):
            problems.append("keyframe timestamps are not increasing")
        steps = [k.step for k in self.keyframes]
        if len(set(steps)) != len(steps):
            problems.append("duplicate steps in keyframes")
        if set(steps) & set(self.skipped_steps):
            problems.append("a step is both aligned and skipped")
        return problems

    def save(self) -> Path:
        problems = self.validate()
        if problems:
            raise ValueError(f"{self.video_id}: " + "; ".join(problems))
        ALIGNMENT_DIR.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(asdict(self), indent=2) + "\n")
        return self.path

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Alignment:
        return cls(
            video_id=d["video_id"],
            sequence=d["sequence"],
            design_match=d.get("design_match", "exact"),
            front_color=d.get("front_color"),
            back_color=d.get("back_color"),
            keyframes=[Keyframe(**k) for k in d.get("keyframes", [])],
            skipped_steps=list(d.get("skipped_steps", [])),
            extra_steps=[ExtraStep(**e) for e in d.get("extra_steps", [])],
            notes=d.get("notes", ""),
        )

    @classmethod
    def load(cls, video_id: str) -> Alignment:
        path = ALIGNMENT_DIR / f"{video_id}.json"
        return cls.from_dict(json.loads(path.read_text()))


def list_alignments() -> list[Alignment]:
    if not ALIGNMENT_DIR.is_dir():
        return []
    return [
        Alignment.from_dict(json.loads(p.read_text())) for p in sorted(ALIGNMENT_DIR.glob("*.json"))
    ]


def new_alignment_template(video_id: str, sequence: str, num_steps: int) -> Alignment:
    """A skeleton with every step listed and timestamps set to -1 to be filled in."""
    return Alignment(
        video_id=video_id,
        sequence=sequence,
        keyframes=[Keyframe(step=s, timestamp_s=-1.0) for s in range(1, num_steps + 1)],
    )
