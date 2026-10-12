from origami_ad.describe import describe_checkpoints


def _op(fn, **params):
    return {"function": fn, "params": params, "assign_to": None}


def test_describe_program_in_plain_words():
    ops = [
        _op("flip", axis="x"),
        _op("fold", edge=[1, 3], direction=1),
        _op("add_vertex", edge=[2, 3], position=0.41421356),
        _op("fold", edge=[1, 5], direction=-1),
    ]
    cps = {
        "1": {"operations": [], "raw_representation": {"faces": {"1": []}, "layers": [[[1]]]}},
        "2": {"operations": ops[:1], "raw_representation": {}},
        "3": {"operations": ops[:2], "raw_representation": {}},
        "4": {"operations": ops, "raw_representation": {}},
    }
    rows = describe_checkpoints(cps)
    assert rows[0]["description"] == "Start with the flat square."
    assert rows[1]["description"] == "Flip the paper over left to right."
    assert "top-right corner to the bottom-left corner" in rows[2]["description"]
    assert rows[3]["description"].startswith("Mark point A 41% of the way along the bottom edge")
    assert "to point A" in rows[3]["description"]


def test_describe_state_square_and_triangle():
    from origami.origami import Origami

    from origami_ad.describe import describe_state

    o = Origami(front_color="red", back_color="white")
    assert (
        describe_state(o.export(), "red", "white") == "The paper is the flat square, red side up."
    )
    o.fold((1, 3), 1)
    text = describe_state(o.export(), "red", "white")
    assert text.startswith("The paper is now a right triangle, half the square, two layers thick")
    assert "side up" in text
