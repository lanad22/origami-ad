from origami_ad.bases import canonical_signature, signature_key

DIAG = {
    "vertices_coords": [[0, 1], [1, 1], [1, 0], [0, 0]],
    "edges_vertices": [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]],
    "edges_assignment": ["B", "B", "B", "B", "V"],
}
DIAG_ROT = {**DIAG, "edges_vertices": [[0, 1], [1, 2], [2, 3], [3, 0], [1, 3]]}
DIAG_MOUNTAIN = {**DIAG, "edges_assignment": ["B", "B", "B", "B", "M"]}
DIAG_NOISY = {**DIAG, "vertices_coords": [[0, 1], [1.0000006, 1], [1, -2.7e-7], [0, 0]]}
BOOK = {
    **DIAG,
    "vertices_coords": [[0, 1], [1, 1], [1, 0], [0, 0], [0.5, 1], [0.5, 0]],
    "edges_vertices": [[0, 4], [4, 1], [1, 2], [2, 5], [5, 3], [3, 0], [4, 5]],
    "edges_assignment": ["B", "B", "B", "B", "B", "B", "V"],
}


def test_rotation_and_flip_invariance():
    k = signature_key(canonical_signature(DIAG))
    assert k == signature_key(canonical_signature(DIAG_ROT))
    assert k == signature_key(canonical_signature(DIAG_MOUNTAIN))
    assert k == signature_key(canonical_signature(DIAG_NOISY))


def test_different_bases_differ():
    assert signature_key(canonical_signature(DIAG)) != signature_key(canonical_signature(BOOK))


def test_square_is_empty_signature():
    sq = {**DIAG, "edges_vertices": DIAG["edges_vertices"][:4], "edges_assignment": ["B"] * 4}
    assert signature_key(canonical_signature(sq)) == "<square>"
