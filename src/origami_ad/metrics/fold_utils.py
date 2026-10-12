"""Helpers on FOLD dicts: conversion from FoldingAgent output, graph/geometry features.

Conventions established against the simulator and the dataset:

* PurelandFold ``cp.fold`` stores the *crease pattern* (unit-square coordinates) with
  ``faceOrders`` giving the layer order of the folded state. Its ``file_classes`` says
  ``creasePattern`` and that is what it is.
* FoldingAgent checkpoints store ``raw_representation`` with *folded* vertex positions.
  ``Origami.from_representation(raw).export_cp()`` yields the same crease-pattern form
  as the dataset, so predictions are converted with :func:`cp_from_raw_representation`.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import networkx as nx
import numpy as np

CREASE_TYPES = ("M", "V", "B", "F", "U")


def cp_from_raw_representation(raw: dict[str, Any]) -> dict[str, Any]:
    """FoldingAgent ``raw_representation`` -> crease-pattern FOLD dict (dataset form)."""
    from origami.origami import Origami

    return Origami.from_representation(raw).export_cp()


def folded_from_raw_representation(raw: dict[str, Any]) -> dict[str, Any]:
    """FoldingAgent ``raw_representation`` -> folded-form FOLD dict (simulator positions)."""
    from origami.origami import Origami

    return Origami.from_representation(raw).export(FOLD=True)


def vertices(fold: dict[str, Any]) -> np.ndarray:
    v = np.asarray(fold.get("vertices_coords", []), dtype=float)
    if v.ndim != 2:
        return np.zeros((0, 2))
    return v


def edges(fold: dict[str, Any]) -> list[tuple[int, int]]:
    return [(int(a), int(b)) for a, b in fold.get("edges_vertices", [])]


def assignments(fold: dict[str, Any]) -> list[str]:
    ea = list(fold.get("edges_assignment", []))
    n = len(fold.get("edges_vertices", []))
    return (ea + ["U"] * n)[:n]


def faces(fold: dict[str, Any]) -> list[list[int]]:
    return [[int(i) for i in f] for f in fold.get("faces_vertices", [])]


def graph(fold: dict[str, Any]) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(len(vertices(fold))))
    for (a, b), t in zip(edges(fold), assignments(fold), strict=True):
        g.add_edge(a, b, assignment=t)
    return g


def degrees(fold: dict[str, Any]) -> list[int]:
    g = graph(fold)
    return [d for _, d in g.degree()]


def num_components(fold: dict[str, Any]) -> int:
    g = graph(fold)
    return nx.number_connected_components(g) if g.number_of_nodes() else 0


def face_sizes(fold: dict[str, Any]) -> list[int]:
    return [len(f) for f in faces(fold)]


def crease_counts(fold: dict[str, Any]) -> Counter:
    return Counter(assignments(fold))


def edge_lengths(fold: dict[str, Any]) -> np.ndarray:
    v = vertices(fold)
    return (
        np.array([np.linalg.norm(v[a] - v[b]) for a, b in edges(fold)]) if len(v) else np.zeros(0)
    )


def boundary_vertices(fold: dict[str, Any]) -> set[int]:
    out: set[int] = set()
    for (a, b), t in zip(edges(fold), assignments(fold), strict=True):
        if t == "B":
            out.update((a, b))
    return out


def face_centroids(fold: dict[str, Any]) -> np.ndarray:
    v = vertices(fold)
    fs = faces(fold)
    if not fs or not len(v):
        return np.zeros((0, 2))
    return np.array([v[f].mean(axis=0) for f in fs])


def face_areas(fold: dict[str, Any]) -> np.ndarray:
    v = vertices(fold)
    out = []
    for f in faces(fold):
        p = v[f]
        x, y = p[:, 0], p[:, 1]
        out.append(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    return np.array(out)


def face_order_map(fold: dict[str, Any]) -> dict[tuple[int, int], int]:
    """``faceOrders`` -> {(i, j) with i<j: +1 if j above i else -1}.

    Accepts both the dataset's symmetric list (``[i,j,s]`` and ``[j,i,-s]``) and the
    simulator's one-sided list. Entries with order 0 (coplanar / unknown) are dropped.
    Conflicting entries for a pair are dropped as well.
    """
    out: dict[tuple[int, int], int] = {}
    conflicts: set[tuple[int, int]] = set()
    for entry in fold.get("faceOrders", []) or []:
        if len(entry) < 3:
            continue
        i, j, s = int(entry[0]), int(entry[1]), int(np.sign(entry[2]))
        if s == 0 or i == j:
            continue
        key, val = ((i, j), s) if i < j else ((j, i), -s)
        if key in out and out[key] != val:
            conflicts.add(key)
        out[key] = val
    for k in conflicts:
        out.pop(k, None)
    return out


def _sorted_crease_angles(fold: dict[str, Any], vid: int, types: set[str]) -> list[float]:
    v = vertices(fold)
    angs = []
    for (a, b), t in zip(edges(fold), assignments(fold), strict=True):
        if t not in types:
            continue
        if a == vid:
            other = b
        elif b == vid:
            other = a
        else:
            continue
        d = v[other] - v[vid]
        angs.append(math.atan2(d[1], d[0]) % (2 * math.pi))
    return sorted(angs)


def maekawa_ok(fold: dict[str, Any], *, tol: int = 0) -> bool:
    """|M - V| == 2 at every interior vertex that carries at least one M/V crease."""
    boundary = boundary_vertices(fold)
    counts: dict[int, Counter] = {}
    for (a, b), t in zip(edges(fold), assignments(fold), strict=True):
        if t in ("M", "V"):
            for vid in (a, b):
                counts.setdefault(vid, Counter())[t] += 1
    for vid, c in counts.items():
        if vid in boundary:
            continue
        if abs(abs(c["M"] - c["V"]) - 2) > tol:
            return False
    return True


def kawasaki_ok(fold: dict[str, Any], *, tol_rad: float = 1e-3) -> bool:
    """Alternating angle sums around every interior crease vertex equal pi.

    Uses M/V/F creases (F creases still split the sector angles of the pattern) and
    skips boundary vertices and vertices with fewer than two creases.
    """
    boundary = boundary_vertices(fold)
    n = len(vertices(fold))
    for vid in range(n):
        if vid in boundary:
            continue
        angs = _sorted_crease_angles(fold, vid, {"M", "V", "F"})
        if len(angs) < 2:
            continue
        sectors = [(angs[(k + 1) % len(angs)] - angs[k]) % (2 * math.pi) for k in range(len(angs))]
        if len(sectors) % 2 == 1:
            return False
        if abs(sum(sectors[0::2]) - math.pi) > tol_rad:
            return False
    return True


def normalize_points(points: np.ndarray) -> np.ndarray:
    """Centre on the centroid and scale so the farthest point is at distance 1 (3-D)."""
    p = np.asarray(points, dtype=float)
    if p.size == 0:
        return np.zeros((0, 3))
    if p.shape[1] == 2:
        p = np.hstack([p, np.zeros((len(p), 1))])
    p = p - p.mean(axis=0)
    r = np.linalg.norm(p, axis=1).max()
    return p / r if r > 0 else p


def hausdorff(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.spatial.distance import directed_hausdorff

    if len(a) == 0 or len(b) == 0:
        return float("inf")
    return max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0])
