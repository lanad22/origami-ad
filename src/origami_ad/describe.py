"""Turn a FoldingAgent fold program into plain-language steps.

Reads a run's ``checkpoints.json``, replays the operations in the simulator so every
vertex can be named by where it currently sits on the paper (``top-left corner``,
``midpoint of the bottom edge``, ``the point 41% along the edge from ... to ...``), and
writes one sentence per keyframe transition. This is the first, rule-based draft of the
audio description; it describes the *program*, not the pixels.

Coordinates follow the simulator: the paper starts as the unit square with x to the
right and y up, so vertex 1 (0,1) is the top-left corner.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

LANDMARKS = {
    (0.0, 1.0): "top-left corner",
    (1.0, 1.0): "top-right corner",
    (1.0, 0.0): "bottom-right corner",
    (0.0, 0.0): "bottom-left corner",
    (0.5, 1.0): "midpoint of the top edge",
    (0.5, 0.0): "midpoint of the bottom edge",
    (0.0, 0.5): "midpoint of the left edge",
    (1.0, 0.5): "midpoint of the right edge",
    (0.5, 0.5): "centre of the paper",
}


def _pos_name(pos, tol: float = 0.02) -> str:
    x, y = float(pos[0]), float(pos[1])
    for (lx, ly), name in LANDMARKS.items():
        if abs(x - lx) < tol and abs(y - ly) < tol:
            return name
    return f"the point at ({x:.2f}, {y:.2f})"


def _the(name: str) -> str:
    """Lettered points take no article: ``point A`` vs ``the top-left corner``."""
    return name if name.startswith("point ") else f"the {name}"


def _pct(p: float) -> str:
    return f"{round(p * 100)}%"


class _Replayer:
    """Replays operations in the simulator; names vertices by current position."""

    def __init__(self) -> None:
        from origami.origami import Origami

        self.o = Origami()
        self.ok = True
        self.marks: dict[int, str] = {}  # vertex id -> short name given when it was marked
        self._next_mark = 0

    def name(self, vid: int) -> str:
        if vid in self.marks:
            return self.marks[vid]
        if not self.ok:
            return f"vertex {vid}"
        try:
            return _pos_name(self.o.vertices[vid].pos)
        except Exception:
            return f"vertex {vid}"

    def edge_name(self, a: int, b: int) -> str:
        """``the bottom edge, starting from ...`` when both ends are corners on one side."""
        na, nb = self.name(a), self.name(b)
        sides = {
            ("top-left corner", "top-right corner"): "top",
            ("bottom-left corner", "bottom-right corner"): "bottom",
            ("top-left corner", "bottom-left corner"): "left",
            ("top-right corner", "bottom-right corner"): "right",
        }
        for (c1, c2), side in sides.items():
            if {na, nb} == {c1, c2}:
                return f"the {side} edge, starting from the {na}"
        return f"the line from the {na} to the {nb}"

    def apply(self, fn: str, p: dict[str, Any]) -> str:
        """Describe *then* apply one operation. Returns the sentence."""
        text = ""
        try:
            if fn == "flip":
                axis = p.get("axis", "x")
                text = "Flip the paper over " + (
                    "left to right." if axis == "x" else "top to bottom."
                )
                if self.ok:
                    self.o.flip(axis)
            elif fn == "rotate":
                ang = float(p.get("angle", p.get("degrees", 0)))
                way = "clockwise" if ang > 0 else "anticlockwise"
                text = f"Rotate the paper {abs(ang):g} degrees {way}."
                if self.ok:
                    self.o.rotate(ang)
            elif fn == "add_vertex":
                a, b = (int(v) for v in p["edge"])
                pos = float(p["position"])
                label = "point " + "ABCDEFGHJKLMNPQRSTUVWXYZ"[self._next_mark % 23]
                self._next_mark += 1
                text = f"Mark {label} {_pct(pos)} of the way along {self.edge_name(a, b)}."
                if self.ok:
                    new_id = self.o.add_vertex((a, b), pos)
                    self.marks[new_id] = label
            elif fn == "fold":
                a, b = (int(v) for v in p["edge"])
                d = int(p.get("direction", 1))
                na, nb = self.name(a), self.name(b)
                side = self._moving_side(a, b, d)
                text = (
                    f"Fold along the line from {_the(na)} to {_the(nb)}, "
                    f"bringing the {side} side over."
                )
                if self.ok:
                    self.o.fold((a, b), d)
            elif fn == "unfold":
                a, b = (int(v) for v in p["edge"])
                text = f"Unfold the crease between {_the(self.name(a))} and {_the(self.name(b))}."
                if self.ok:
                    self.o.unfold((a, b))
            else:
                text = f"{fn}({p})"
        except Exception as exc:
            self.ok = False
            text = (text or f"{fn}({p})") + f"  [replay stopped: {type(exc).__name__}]"
        return text

    def _moving_side(self, a: int, b: int, d: int) -> str:
        try:
            pa, pb = self.o.vertices[a].pos, self.o.vertices[b].pos
            vertical = abs(float(pa[0]) - float(pb[0])) < 1e-6
        except Exception:
            vertical = False
        if vertical:
            return "right" if d > 0 else "left"
        return "upper" if d > 0 else "lower"


# ── Paper state in words ──────────────────────────────────────────────────────

_FRACTIONS = [
    (1.0, "the full square"),
    (0.75, "three quarters of the square"),
    (2 / 3, "two thirds of the square"),
    (0.5, "half the square"),
    (3 / 8, "three eighths of the square"),
    (1 / 3, "a third of the square"),
    (0.25, "a quarter of the square"),
    (1 / 8, "an eighth of the square"),
]


def _size_words(area: float) -> str:
    for f, words in _FRACTIONS:
        if abs(area - f) < 0.035:
            return words
    return f"about {round(area * 100)}% of the square"


def _shape_words(poly) -> str:
    """Name a simplified outline polygon: triangle, square, rectangle, kite, pentagon..."""
    ring = list(poly.simplify(0.015).exterior.coords)[:-1]
    n = len(ring)
    pts = [np.array(p) for p in ring]
    sides = [np.linalg.norm(pts[(i + 1) % n] - pts[i]) for i in range(n)]
    angles = []
    for i in range(n):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
        u, v = a - b, c - b
        cosang = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
        angles.append(math.degrees(math.acos(max(-1.0, min(1.0, cosang)))))
    right = [abs(x - 90) < 6 for x in angles]
    if n == 3:
        return "a right triangle" if any(right) else "a triangle"
    if n == 4:
        eq = max(sides) - min(sides) < 0.06
        if all(right):
            return "a square" if eq else "a rectangle"
        if eq:
            return "a diamond"
        if (
            abs(sides[0] - sides[1]) < 0.05
            and abs(sides[2] - sides[3]) < 0.05
            or (abs(sides[1] - sides[2]) < 0.05 and abs(sides[3] - sides[0]) < 0.05)
        ):
            return "a kite shape"
        return "a four-sided shape"
    names = {5: "a pentagon", 6: "a hexagon", 7: "a seven-sided shape", 8: "an eight-sided shape"}
    return names.get(n, f"a {n}-sided shape")


def _long_edge_words(poly) -> str:
    ring = list(poly.simplify(0.015).exterior.coords)[:-1]
    n = len(ring)
    best, best_len = None, -1.0
    for i in range(n):
        a, b = np.array(ring[i]), np.array(ring[(i + 1) % n])
        if np.linalg.norm(b - a) > best_len:
            best_len, best = float(np.linalg.norm(b - a)), (a, b)
    a, b = best
    cx, cy = poly.centroid.x, poly.centroid.y
    mx, my = (a + b) / 2
    d = b - a
    ang = abs(math.degrees(math.atan2(d[1], d[0]))) % 180
    if ang < 20 or ang > 160:
        return "its longest edge along the " + ("bottom" if my < cy else "top")
    if 70 < ang < 110:
        return "its longest edge on the " + ("left" if mx < cx else "right")
    return "its longest edge running diagonally"


def describe_state(
    raw: dict[str, Any], front_color: str = "front", back_color: str = "back"
) -> str:
    """One sentence on what the paper looks like now, from the folded geometry."""
    try:
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
    except ImportError:  # pragma: no cover
        return ""
    verts = {int(k): np.array(v, dtype=float) for k, v in raw.get("vertices", {}).items()}
    faces = {int(k): [int(i) for i in v] for k, v in raw.get("faces", {}).items()}
    orient = {int(k): int(v) for k, v in (raw.get("faces_orientations") or {}).items()}
    polys = {}
    for fid, vids in faces.items():
        if len(vids) >= 3:
            pg = Polygon([verts[v] for v in vids])
            if pg.is_valid and pg.area > 1e-9:
                polys[fid] = pg
    if not polys:
        return ""
    outline = unary_union(list(polys.values()))
    if outline.geom_type != "Polygon":
        outline = max(outline.geoms, key=lambda g: g.area)
    area = outline.area
    shape = _shape_words(outline)
    size = _size_words(area)

    # thickness: most faces stacked over any one point
    thickness = 1
    for pg in polys.values():
        pt = pg.representative_point()
        thickness = max(thickness, sum(1 for q in polys.values() if q.contains(pt)))

    # visible colour: walk layers from the top, count uncovered area per side
    order = [fid for level in raw.get("layers", []) for group in level for fid in group]
    order = [f for f in order if f in polys] or list(polys)
    remaining = outline
    seen = {0: 0.0, 1: 0.0}
    for fid in reversed(order):  # last level is the top
        vis = polys[fid].intersection(remaining)
        if not vis.is_empty:
            seen[orient.get(fid, 0)] += vis.area
            remaining = remaining.difference(polys[fid])
    total = seen[0] + seen[1] or 1.0
    f_share = seen[0] / total
    if f_share > 0.9:
        colour = f"{front_color} side up"
    elif f_share < 0.1:
        colour = f"{back_color} side up"
    else:
        major, minor = (front_color, back_color) if f_share >= 0.5 else (back_color, front_color)
        colour = f"mostly {major} with some {minor} showing"

    words = ["", "one", "two", "three", "four", "five", "six", "seven", "eight"]
    if thickness == 1:
        thick = "a single layer"
    elif thickness < len(words):
        thick = f"{words[thickness]} layers thick"
    else:
        thick = f"{thickness} layers thick"
    if shape == "a square" and abs(area - 1.0) < 0.035 and thickness == 1:
        return f"The paper is the flat square, {colour}."
    return f"The paper is now {shape}, {size}, {thick}, {colour}, with {_long_edge_words(outline)}."


def _ops(entry: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for op in entry.get("operations", []):
        fn = op.get("function") or op.get("name")
        params = op.get("params") or op.get("parameters") or {}
        out.append((fn, params))
    return out


def describe_checkpoints(
    checkpoints: dict[str, Any], keyframe_meta: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """One row per keyframe: frame, step (if known), timestamp (if known), sentences."""
    rep = _Replayer()
    rows = []
    prev_n = 0
    frames = sorted(checkpoints, key=int)
    # Checkpoint 1 is saved before the manifest colours are applied, so take the colours
    # from the latest checkpoint and use them for every frame.
    run_colors: dict[str, str] = {}
    for k in reversed(frames):
        c = checkpoints[k].get("paper_colors") or {}
        if c.get("front_color"):
            run_colors = c
            break
    for k in frames:
        ops = _ops(checkpoints[k])
        new = ops[prev_n:]
        prev_n = len(ops)
        sentences = [rep.apply(fn, p) for fn, p in new]
        if int(k) == 1 and not new:
            sentences = ["Start with the flat square."]
        elif not new:
            sentences = ["No change from the previous keyframe."]
        row = {"frame": int(k), "description": " ".join(sentences), "num_operations": len(new)}
        if keyframe_meta and int(k) - 1 < len(keyframe_meta):
            m = keyframe_meta[int(k) - 1]
            row["step"] = m.get("step")
            row["timestamp_s"] = m.get("timestamp_s")
        geom = checkpoints[k].get("raw_representation") or {}
        colors = run_colors or (checkpoints[k].get("paper_colors") or {})
        row["faces"] = len(geom.get("faces", {}))
        row["layers"] = len(geom.get("layers", []))
        row["state"] = describe_state(
            geom, colors.get("front_color", "front"), colors.get("back_color", "back")
        )
        rows.append(row)
    return rows


def describe_run(run_dir: str | Path, *, write: bool = True) -> str:
    run_dir = Path(run_dir)
    checkpoints = json.loads((run_dir / "checkpoints.json").read_text())
    meta: list[dict[str, Any]] | None = None
    md = run_dir / "metadata.json"
    if md.exists():
        kf = Path(json.loads(md.read_text()).get("keyframes", ""))
        meta_path = kf.parent / "keyframes_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
    rows = describe_checkpoints(checkpoints, meta)
    has_ts = any(r.get("timestamp_s") is not None for r in rows)
    lines = [
        "| frame | "
        + ("time | " if has_ts else "")
        + "what the agent concluded happened | what the paper looks like now |",
        "|---|" + ("---|" if has_ts else "") + "---|---|",
    ]
    for r in rows:
        ts = (
            f"{r['timestamp_s']:.1f}s | "
            if has_ts and r.get("timestamp_s") is not None
            else ("| " if has_ts else "")
        )
        lines.append(f"| {r['frame']} | {ts}{r['description']} | {r.get('state', '')} |")
    text = "\n".join(lines) + "\n"
    if write:
        (run_dir / "description.md").write_text(text)
        (run_dir / "description.json").write_text(json.dumps(rows, indent=1) + "\n")
    return text
