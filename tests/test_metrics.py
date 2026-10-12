import math

import numpy as np

from origami_ad.metrics import fold_utils as fu
from origami_ad.metrics.cpd import crease_pattern_dissimilarity
from origami_ad.metrics.origamispace import (
    all_similarities,
    layering_consistency,
    symmetric_hausdorff,
    topological_structure_similarity,
)

SQUARE = {
    "vertices_coords": [[0, 1], [1, 1], [1, 0], [0, 0]],
    "edges_vertices": [[0, 1], [1, 2], [2, 3], [3, 0]],
    "edges_assignment": ["B", "B", "B", "B"],
    "faces_vertices": [[0, 1, 2, 3]],
    "faceOrders": [],
}
DIAG = {
    "vertices_coords": [[0, 1], [1, 1], [1, 0], [0, 0]],
    "edges_vertices": [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]],
    "edges_assignment": ["B", "B", "B", "B", "V"],
    "faces_vertices": [[0, 1, 2], [0, 2, 3]],
    "faceOrders": [[0, 1, 1]],
}
DIAG_OTHER = {
    **DIAG,
    "edges_vertices": [[0, 1], [1, 2], [2, 3], [3, 0], [1, 3]],
    "faces_vertices": [[0, 1, 3], [1, 2, 3]],
}
FF_OK = {
    "compilation_success": True,
    "Vf": [[0, 1], [0, 0], [1, 0], [0, 0]],
    "taco_taco": 0,
    "taco_tortilla": 0,
    "transitivity": 0,
}


def test_cpd_identity_is_zero():
    assert crease_pattern_dissimilarity(DIAG, DIAG).total == 0


def test_cpd_symmetric_square_vs_diag_positive():
    r = crease_pattern_dissimilarity(SQUARE, DIAG)
    assert r.total > 0
    assert r.n_edges == (4, 5)


def test_cpd_symmetry_invariant_up_to_rotation():
    # the two diagonals are related by a rotation; CPD is not rotation invariant by design,
    # but both should be equidistant from the plain square
    a = crease_pattern_dissimilarity(SQUARE, DIAG).total
    b = crease_pattern_dissimilarity(SQUARE, DIAG_OTHER).total
    assert math.isclose(a, b, rel_tol=1e-9)


def test_tss_identity_and_degradation():
    assert math.isclose(topological_structure_similarity(DIAG, DIAG)["tss"], 1.0)
    assert topological_structure_similarity(SQUARE, DIAG)["tss"] < 1.0


def test_penalties_when_prediction_fails_to_compile():
    s = all_similarities(DIAG, DIAG, {"compilation_success": False}, FF_OK)
    assert s["gs"] == 0.2 and s["cs"] == 0.2 and s["ffs"] == 0.3


def test_identity_similarities_are_one():
    s = all_similarities(DIAG, DIAG, FF_OK, FF_OK)
    for k in ("tss", "gs", "cs", "ffs"):
        assert math.isclose(s[k], 1.0), k


def test_symmetric_hausdorff_handles_reflection():
    a = fu.normalize_points(np.array([[0, 0], [1, 0], [1, 0.3]]))
    b = a.copy()
    b[:, 0] *= -1
    assert symmetric_hausdorff(a, b) < 1e-9
    assert fu.hausdorff(a, b) > 0.1


def test_layering_consistency_up_to_global_flip():
    flipped = {**DIAG, "faceOrders": [[0, 1, -1]]}
    assert layering_consistency(DIAG, DIAG) == 1.0
    assert layering_consistency(flipped, DIAG) == 1.0


def test_face_order_map_accepts_symmetric_lists():
    m = fu.face_order_map({"faceOrders": [[0, 1, -1], [1, 0, 1], [0, 2, 0]]})
    assert m == {(0, 1): -1}


def test_maekawa_kawasaki_on_simple_patterns():
    assert fu.maekawa_ok(DIAG) and fu.kawasaki_ok(DIAG)
    bad = {
        "vertices_coords": [[0, 1], [1, 1], [1, 0], [0, 0], [0.5, 0.5]],
        "edges_vertices": [[0, 1], [1, 2], [2, 3], [3, 0], [4, 0], [4, 1], [4, 2]],
        "edges_assignment": ["B", "B", "B", "B", "M", "M", "M"],
        "faces_vertices": [],
    }
    assert not fu.maekawa_ok(bad)


def test_identical_flat_squares_score_one():
    ff = {
        "compilation_success": True,
        "Vf": SQUARE["vertices_coords"],
        "taco_taco": 0,
        "taco_tortilla": 0,
        "transitivity": 0,
    }
    s = all_similarities(SQUARE, SQUARE, ff, ff)
    for k in ("tss", "gs", "cs", "ffs"):
        assert math.isclose(s[k], 1.0), (k, s[k])
    assert layering_consistency(SQUARE, DIAG) == 0.0
