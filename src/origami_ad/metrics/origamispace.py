"""OrigamiSpace-style similarity metrics: TSS, GS, CS, FFS.

Reimplemented from the OrigamiSpace appendix (arXiv 2511.18450, App. D) as summarised
in the FoldingAgent paper (Sec. 5.2.2, Eqs. 4-7). Weights and penalty values follow the
papers; a few sub-scores are under-specified there, and our choices are documented
inline and flagged ``approx`` in the output so results are read as *comparable across
our own runs*, not as a bit-exact reproduction of the official evaluator:

* degree / face-size distributions: 1-D Wasserstein distance between the empirical
  distributions, divided by the larger maximum value so it lies in [0, 1];
* taco-taco / taco-tortilla / transitivity: only the counts from Flat-Folder are
  compared (Jaccard over constraint sets is undefined when face indices do not
  correspond between a prediction and the reference);
* layering consistency (LCR): faces are put in correspondence by Hungarian matching
  on crease-pattern centroids (+ area), then the fraction of face pairs whose
  above/below relation agrees is reported, up to a global flip (a model turned over
  is the same model; Flat-Folder's root-face choice can flip the whole stack);
* Hausdorff distances on folded points are taken as the minimum over the square's
  8 symmetries applied to the prediction, for the same root-face reason.

All functions take crease-pattern FOLD dicts (dataset form) plus the Flat-Folder
compile result for each (see ``origami_ad.metrics.flatfolder``).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import wasserstein_distance

from origami_ad.metrics import fold_utils as fu

PENALTY_GS = 0.2
PENALTY_CS = 0.2
PENALTY_FFS = 0.3
HAUSDORFF_K = 5.0


def _exp_rel(a: float, b: float) -> float:
    return 1.0 if a == b else math.exp(-abs(a - b) / max(1.0, min(a, b)))


def _w1_score(xs: list[int], ys: list[int]) -> float:
    if not xs or not ys:
        return 0.0 if (xs or ys) else 1.0
    scale = max(max(xs), max(ys), 1)
    return float(np.clip(1.0 - wasserstein_distance(xs, ys) / scale, 0.0, 1.0))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.5
    return float(np.dot(a, b) / (na * nb))


# ── TSS ──────────────────────────────────────────────────────────────────────


def topological_structure_similarity(pred: dict[str, Any], ref: dict[str, Any]) -> dict[str, float]:
    nv_p, nv_r = len(fu.vertices(pred)), len(fu.vertices(ref))
    if nv_p == nv_r:
        s_vertex = 1.0
    else:
        lo = max(min(nv_p, nv_r), 1)
        s_vertex = math.exp(-0.5 * abs(nv_p - nv_r) / lo)

    s_degree = _w1_score(fu.degrees(pred), fu.degrees(ref))
    cp, cr = fu.num_components(pred), fu.num_components(ref)
    s_conn = 1.0 if cp == cr else math.exp(-abs(cp - cr))
    s_edge = 0.7 * s_degree + 0.3 * s_conn

    fs_p, fs_r = fu.face_sizes(pred), fu.face_sizes(ref)
    s_fcount = _exp_rel(len(fs_p), len(fs_r))
    avg_p = float(np.mean(fs_p)) if fs_p else 0.0
    avg_r = float(np.mean(fs_r)) if fs_r else 0.0
    s_favg = _exp_rel(avg_p, avg_r)
    s_fdist = _w1_score(fs_p, fs_r)
    s_face = 0.3 * s_fcount + 0.3 * s_favg + 0.4 * s_fdist

    ap, ar = fu.assignments(pred), fu.assignments(ref)
    if not ap or not ar:
        s_crease = 0.2
    else:

        def prop(a, t):
            return sum(1 for x in a if x == t) / len(a)

        s_m = 1 - abs(prop(ap, "M") - prop(ar, "M"))
        s_v = 1 - abs(prop(ap, "V") - prop(ar, "V"))
        s_b = 1 - abs(prop(ap, "B") - prop(ar, "B"))
        p_l = min(len(ap), len(ar)) / max(len(ap), len(ar))
        s_crease = (0.4 * s_m + 0.4 * s_v + 0.2 * s_b) * p_l

    tss = 0.2 * s_vertex + 0.3 * s_edge + 0.3 * s_face + 0.2 * s_crease
    return {
        "tss": tss,
        "tss_vertex": s_vertex,
        "tss_edge": s_edge,
        "tss_face": s_face,
        "tss_crease": s_crease,
    }


# ── GS ───────────────────────────────────────────────────────────────────────


def _folded_points(ff: dict[str, Any] | None) -> np.ndarray:
    if not ff or not ff.get("compilation_success"):
        return np.zeros((0, 3))
    return fu.normalize_points(np.asarray(ff.get("Vf", []), dtype=float))


_D4 = [
    np.array([[1, 0], [0, 1]]),
    np.array([[0, -1], [1, 0]]),
    np.array([[-1, 0], [0, -1]]),
    np.array([[0, 1], [-1, 0]]),
    np.array([[-1, 0], [0, 1]]),
    np.array([[1, 0], [0, -1]]),
    np.array([[0, 1], [1, 0]]),
    np.array([[0, -1], [-1, 0]]),
]


def symmetric_hausdorff(pred_pts: np.ndarray, ref_pts: np.ndarray) -> float:
    """Bidirectional Hausdorff, minimised over the 8 planar symmetries of the prediction."""
    if len(pred_pts) == 0 or len(ref_pts) == 0:
        return float("inf")
    best = float("inf")
    for r in _D4:
        p = pred_pts.copy()
        p[:, :2] = p[:, :2] @ r.T
        best = min(best, fu.hausdorff(p, ref_pts))
    return best


def _dihedral_histogram(fold: dict[str, Any]) -> np.ndarray:
    """18 bins of 10 degrees over [0, 180]; flat-folded M/V creases fold to 180, F to 0."""
    hist = np.zeros(18)
    for t in fu.assignments(fold):
        if t in ("M", "V"):
            hist[17] += 1
        elif t == "F":
            hist[0] += 1
    return hist


def _bbox_ratios(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(3)
    ext = np.sort(points.max(axis=0) - points.min(axis=0))[::-1]
    return ext / ext[0] if ext[0] > 0 else np.zeros(3)


def geometric_similarity(
    pred: dict[str, Any],
    ref: dict[str, Any],
    ff_pred: dict[str, Any] | None,
    ff_ref: dict[str, Any] | None,
) -> dict[str, float]:
    if not (ff_pred and ff_pred.get("compilation_success")) or not (
        ff_ref and ff_ref.get("compilation_success")
    ):
        return {"gs": PENALTY_GS, "gs_point": 0.0, "gs_angle": 0.0, "gs_size": 0.0}
    pp, pr = _folded_points(ff_pred), _folded_points(ff_ref)
    d_h = symmetric_hausdorff(pp, pr)
    s_point = math.exp(-HAUSDORFF_K * d_h) if math.isfinite(d_h) else 0.0
    hp, hr = _dihedral_histogram(pred), _dihedral_histogram(ref)
    if hp.sum() == 0 and hr.sum() == 0:
        s_angle = 1.0  # neither state has a crease yet: identical, not unknown
    elif hp.sum() == 0 or hr.sum() == 0:
        s_angle = 0.0
    else:
        s_angle = _cosine(hp, hr)
    s_size = _cosine(_bbox_ratios(pp), _bbox_ratios(pr)) if len(pp) and len(pr) else 0.5
    gs = 0.4 * s_point + 0.3 * s_angle + 0.3 * s_size
    return {"gs": gs, "gs_point": s_point, "gs_angle": s_angle, "gs_size": s_size}


# ── CS ───────────────────────────────────────────────────────────────────────


def _count_score(n_p: float, n_r: float) -> float:
    if n_p == 0 and n_r == 0:
        return 1.0
    if n_p == 0 or n_r == 0:
        return 0.3
    return _exp_rel(n_p, n_r)


def constraint_satisfaction(
    pred: dict[str, Any],
    ref: dict[str, Any],
    ff_pred: dict[str, Any] | None,
    ff_ref: dict[str, Any] | None,
) -> dict[str, float]:
    ref_ok = bool(ff_ref and ff_ref.get("compilation_success"))
    pred_ok = bool(ff_pred and ff_pred.get("compilation_success"))
    if ref_ok and not pred_ok:
        return {"cs": PENALTY_CS, "cs_tt": 0.0, "cs_tto": 0.0, "cs_trans": 0.0, "cs_ff": 0.0}
    ff_pred = ff_pred or {}
    ff_ref = ff_ref or {}
    s_tt = _count_score(ff_pred.get("taco_taco", 0), ff_ref.get("taco_taco", 0))
    s_tto = _count_score(ff_pred.get("taco_tortilla", 0), ff_ref.get("taco_tortilla", 0))
    s_tr = _count_score(ff_pred.get("transitivity", 0), ff_ref.get("transitivity", 0))

    def thm(fn) -> float:
        return 0.2 if (fn(ref) and not fn(pred)) else 1.0

    s_ff = 0.5 * thm(fu.kawasaki_ok) + 0.5 * thm(fu.maekawa_ok)
    cs = 0.3 * s_tt + 0.3 * s_tto + 0.2 * s_tr + 0.2 * s_ff
    return {"cs": cs, "cs_tt": s_tt, "cs_tto": s_tto, "cs_trans": s_tr, "cs_ff": s_ff}


# ── FFS ──────────────────────────────────────────────────────────────────────


def match_faces(pred: dict[str, Any], ref: dict[str, Any]) -> dict[int, int]:
    """pred face index -> ref face index, by Hungarian matching on CP centroid + area."""
    cp, cr = fu.face_centroids(pred), fu.face_centroids(ref)
    if len(cp) == 0 or len(cr) == 0:
        return {}
    ap, ar = fu.face_areas(pred), fu.face_areas(ref)
    cost = np.linalg.norm(cp[:, None, :] - cr[None, :, :], axis=2) + np.abs(
        ap[:, None] - ar[None, :]
    )
    r, k = linear_sum_assignment(cost)
    return {int(i): int(j) for i, j in zip(r, k, strict=True)}


def layering_consistency(pred: dict[str, Any], ref: dict[str, Any]) -> float:
    """Fraction of comparable face pairs whose above/below relation agrees, up to a global flip.

    1.0 when neither state has any stacking (e.g. both flat), 0.0 when only one does.
    """
    m = match_faces(pred, ref)
    op, orf = fu.face_order_map(pred), fu.face_order_map(ref)
    if not op and not orf:
        return 1.0  # neither state has stacked faces: consistent
    if not op or not orf or not m:
        return 0.0
    agree = total = 0
    for (i, j), s in op.items():
        if i not in m or j not in m:
            continue
        a, b = m[i], m[j]
        key, s_ref_expected = ((a, b), s) if a < b else ((b, a), -s)
        if key not in orf:
            continue
        total += 1
        agree += int(orf[key] == s_ref_expected)
    if not total:
        return 0.5
    frac = agree / total
    return max(frac, 1.0 - frac)


def final_folded_state(
    pred: dict[str, Any],
    ref: dict[str, Any],
    ff_pred: dict[str, Any] | None,
    ff_ref: dict[str, Any] | None,
) -> dict[str, float]:
    pp, pr = _folded_points(ff_pred), _folded_points(ff_ref)
    if len(pp) == 0 or len(pr) == 0:
        return {"ffs": PENALTY_FFS, "ffs_shape": 0.0, "ffs_layer": 0.0}
    d_h = symmetric_hausdorff(pp, pr)
    s_shape = math.exp(-HAUSDORFF_K * d_h) if math.isfinite(d_h) else 0.0
    s_layer = layering_consistency(pred, ref)
    return {"ffs": 0.7 * s_shape + 0.3 * s_layer, "ffs_shape": s_shape, "ffs_layer": s_layer}


def all_similarities(
    pred: dict[str, Any],
    ref: dict[str, Any],
    ff_pred: dict[str, Any] | None,
    ff_ref: dict[str, Any] | None,
) -> dict[str, float]:
    out: dict[str, float] = {}
    out.update(topological_structure_similarity(pred, ref))
    out.update(geometric_similarity(pred, ref, ff_pred, ff_ref))
    out.update(constraint_satisfaction(pred, ref, ff_pred, ff_ref))
    out.update(final_folded_state(pred, ref, ff_pred, ff_ref))
    return out
