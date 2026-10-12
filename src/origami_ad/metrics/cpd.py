"""Crease-Pattern Dissimilarity (CPD), after Oh, Toussaint, Demaine & Demaine (2015).

Two crease patterns are compared as geometric graphs via two minimum-weight bipartite
matchings, one over edges and one over vertices; the dissimilarity is the sum of the
two matching costs. Arc weights are linear combinations of normalised features:

    E1 = |len1 - len2| / sqrt(2)
    E2 = smaller angle between the supporting lines / (2 pi)
    E3 = minimum Euclidean distance between the two segments / sqrt(2)
    E4 = 1 if the crease types differ else 0
    V1 = |deg1 - deg2|
    V2 = distance between the vertices / sqrt(2)

Default weights are the paper's "UP" set (wv1, wv2, we1, we2, we3, we4) =
(0, 18.333, 51.285, 19.503, 0, 0), which the authors found the better perceptual
indicator; the "Bio" set is also provided.

Unequal element counts: the paper pads the cost matrix with dummy rows/columns "of very
high cost". We use a rectangular assignment (every element of the smaller set is matched)
and charge each unmatched element ``unmatched_cost``, which defaults to the maximum
possible arc weight (sum of the feature weights). That keeps the scale of the matched
part and makes the penalty configurable; it is an assumption, not the paper's constant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from origami_ad.metrics import fold_utils as fu

WEIGHTS_UP = {"wv1": 0.0, "wv2": 18.333, "we1": 51.285, "we2": 19.503, "we3": 0.0, "we4": 0.0}
WEIGHTS_BIO = {"wv1": 16.2, "wv2": 65.253, "we1": 80.906, "we2": 45.36, "we3": 13.744, "we4": 0.0}


@dataclass
class CPDResult:
    edge_cost: float
    vertex_cost: float
    n_edges: tuple[int, int]
    n_vertices: tuple[int, int]

    @property
    def total(self) -> float:
        return self.edge_cost + self.vertex_cost

    def as_dict(self) -> dict[str, Any]:
        return {
            "cpd": self.total,
            "cpd_edge": self.edge_cost,
            "cpd_vertex": self.vertex_cost,
            "n_edges": list(self.n_edges),
            "n_vertices": list(self.n_vertices),
        }


def _segment_distance(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> float:
    """Minimum distance between 2-D segments p1p2 and q1q2."""

    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_seg(a, b, c):
        return min(a[0], b[0]) - 1e-12 <= c[0] <= max(a[0], b[0]) + 1e-12 and (
            min(a[1], b[1]) - 1e-12 <= c[1] <= max(a[1], b[1]) + 1e-12
        )

    d1, d2 = orient(q1, q2, p1), orient(q1, q2, p2)
    d3, d4 = orient(p1, p2, q1), orient(p1, p2, q2)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)) and d1 * d2 < 0 and d3 * d4 < 0:
        return 0.0
    for a, b, c, d in ((q1, q2, p1, d1), (q1, q2, p2, d2), (p1, p2, q1, d3), (p1, p2, q2, d4)):
        if abs(d) < 1e-12 and on_seg(a, b, c):
            return 0.0

    def pt_seg(p, a, b):
        ab = b - a
        t = 0.0 if np.dot(ab, ab) == 0 else float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1))
        return float(np.linalg.norm(p - (a + t * ab)))

    return min(pt_seg(p1, q1, q2), pt_seg(p2, q1, q2), pt_seg(q1, p1, p2), pt_seg(q2, p1, p2))


def _edge_features(fold: dict[str, Any]):
    v = fu.vertices(fold)
    out = []
    for (a, b), t in zip(fu.edges(fold), fu.assignments(fold), strict=True):
        p, q = v[a], v[b]
        d = q - p
        out.append((p, q, float(np.linalg.norm(d)), math.atan2(d[1], d[0]) % math.pi, t))
    return out


def _line_angle(a1: float, a2: float) -> float:
    d = abs(a1 - a2) % math.pi
    return min(d, math.pi - d)


def edge_cost_matrix(f1: dict[str, Any], f2: dict[str, Any], w: dict[str, float]) -> np.ndarray:
    e1, e2 = _edge_features(f1), _edge_features(f2)
    c = np.zeros((len(e1), len(e2)))
    s2 = math.sqrt(2)
    for i, (p1, q1, l1, a1, t1) in enumerate(e1):
        for j, (p2, q2, l2, a2, t2) in enumerate(e2):
            cost = w["we1"] * abs(l1 - l2) / s2 + w["we2"] * _line_angle(a1, a2) / (2 * math.pi)
            if w["we3"]:
                cost += w["we3"] * _segment_distance(p1, q1, p2, q2) / s2
            if w["we4"]:
                cost += w["we4"] * (0.0 if t1 == t2 else 1.0)
            c[i, j] = cost
    return c


def vertex_cost_matrix(f1: dict[str, Any], f2: dict[str, Any], w: dict[str, float]) -> np.ndarray:
    v1, v2 = fu.vertices(f1), fu.vertices(f2)
    d1, d2 = np.array(fu.degrees(f1)), np.array(fu.degrees(f2))
    c = np.zeros((len(v1), len(v2)))
    if len(v1) and len(v2):
        dist = np.linalg.norm(v1[:, None, :] - v2[None, :, :], axis=2) / math.sqrt(2)
        c = w["wv1"] * np.abs(d1[:, None] - d2[None, :]) + w["wv2"] * dist
    return c


def _matching_cost(c: np.ndarray, unmatched_cost: float) -> float:
    n, m = c.shape
    if n == 0 or m == 0:
        return unmatched_cost * max(n, m)
    r, k = linear_sum_assignment(c)
    return float(c[r, k].sum()) + unmatched_cost * abs(n - m)


def crease_pattern_dissimilarity(
    pred: dict[str, Any],
    ref: dict[str, Any],
    *,
    weights: dict[str, float] | None = None,
    unmatched_cost: float | None = None,
) -> CPDResult:
    """CPD(pred, ref) = edge matching cost + vertex matching cost."""
    w = dict(WEIGHTS_UP if weights is None else weights)
    ce = edge_cost_matrix(pred, ref, w)
    cv = vertex_cost_matrix(pred, ref, w)
    pen_e = (
        sum(w[k] for k in ("we1", "we2", "we3", "we4"))
        if unmatched_cost is None
        else unmatched_cost
    )
    pen_v = sum(w[k] for k in ("wv1", "wv2")) if unmatched_cost is None else unmatched_cost
    return CPDResult(
        edge_cost=_matching_cost(ce, pen_e),
        vertex_cost=_matching_cost(cv, pen_v),
        n_edges=(ce.shape[0], ce.shape[1]),
        n_vertices=(cv.shape[0], cv.shape[1]),
    )
