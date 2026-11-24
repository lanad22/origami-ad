import json
from types import SimpleNamespace

import cv2
import numpy as np

from origami_ad.keyframe_bench import score, totals
from origami_ad.keyframe_select import (
    Picker,
    SelectParams,
    calm_candidates,
    parse_verdict,
    review_pairs,
    select_keyframes,
)
from origami_ad.keyframing import KeyframeAnalysis, KeyframeParams


def test_score_hits_duplicates_extras_and_linked_reshows():
    label = {"required": [[0, 2], [5, 7], [10, 12]], "optional": [[3, 4], [13, 15, 3]]}
    s = score([1.0, 1.5, 3.5, 3.8, 6.0, 8.5, 14.0], label)
    assert s.detail == "HDoDHXh"
    assert (s.hits, s.extras, s.duplicates, s.missed) == (3, 1, 2, [])
    s = score([11.0, 14.0], label)  # state 3 already hit, so the re-show is a duplicate
    assert s.detail == "HD" and s.missed == [1, 2]
    t = totals({"a": score([1.0, 6.0, 8.5], label)})
    assert (t["hits"], t["bad"]) == (2, 1)
    assert abs(t["recall"] - 2 / 3) < 1e-9 and abs(t["precision"] - 2 / 3) < 1e-9


def test_calm_candidates_take_the_calmest_sample_per_slot():
    motion = [0.5, 0.1, 0.4, 0.3, 0.9, 0.0, 0.2, 0.6]
    an = KeyframeAnalysis(
        video="x", duration_s=4.0, sample_fps=2.0, params={}, motion=motion,
        hands=[0] * 8, hand_cover=[0.0] * 8, still=[False] * 8,
    )
    # 1 s slots centred on 0, 1, 2, 3 s cover samples [0], [1, 2], [3, 4], [5, 6]
    assert calm_candidates(an, 1.0) == [0, 1, 3, 5]


def _video(path, fps=10, states=3, still_s=2.0):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    for s in range(states):
        frame = np.full((120, 160, 3), 200, np.uint8)
        cv2.rectangle(frame, (30 + 10 * s, 30), (120 - 10 * s, 90), (0, 0, 220), -1)
        for _ in range(int(still_s * fps)):
            out.write(frame)
    out.release()


class _FakeClient:
    """Keeps tile 1 of every pick grid, everything in the grid review, and calls every
    pairwise candidate a new state."""

    def __init__(self):
        self.prompts = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        prompt = kw["messages"][0]["content"][1]["text"]
        self.prompts.append(prompt)
        if prompt.startswith("These tiles"):
            reply = {"keep": [1]}
        elif prompt.startswith("These are frames"):
            reply = {"verdict": "new", "reason": "fake"}
        else:
            n = int(prompt.split()[1])
            reply = {"keep": list(range(1, n + 1))}
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text=json.dumps(reply))],
            usage=SimpleNamespace(input_tokens=100, output_tokens=10),
        )


def _params(**kw):
    return SelectParams(
        step_s=1.0,
        per_grid=2,
        signals=KeyframeParams(use_hands=False, intro_policy=None, outro_policy=None),
        **kw,
    )


def test_select_keyframes_grid_review_batches_and_caches(tmp_path):
    path = tmp_path / "v.mp4"
    _video(path)
    p = _params(review="grid")
    client = _FakeClient()
    an = select_keyframes(path, p, picker=Picker(p, cache_dir=tmp_path / "cache", client=client))
    sel = an.params["selection"]
    assert len(sel["candidates_s"]) == 6  # one per second of a 6 s video
    assert sel["picked_s"] == sel["candidates_s"][::2]  # tile 1 of each two-tile grid
    assert an.timestamps == sel["final_s"] == sel["picked_s"]
    assert sel["calls"] == 4 and sel["new_input_tokens"] == 400  # 3 pick grids + 1 review
    # a second run is served from the cache: same answer, no new calls
    again = Picker(p, cache_dir=tmp_path / "cache", client=client)
    an2 = select_keyframes(path, p, analysis=None, picker=again)
    assert an2.timestamps == an.timestamps and len(client.prompts) == 4
    assert an2.params["selection"]["new_input_tokens"] == 0


def test_parse_reply_tolerates_text_around_json():
    from origami_ad.keyframe_select import parse_reply

    assert parse_reply('{"keep": [9, 11], "why": {"9": "flat"}}')["keep"] == [9, 11]
    # trailing commentary containing a brace used to break the greedy regex
    assert parse_reply('```json\n{"keep": [2]}\n```\nNote: tile {3} is blurry.')["keep"] == [2]
    # malformed reason string: still recover the keep list
    bad = '{"keep": [9, 11, 18], "why": {"9": "6" x 6" sheet"}}'
    assert parse_reply(bad)["keep"] == [9, 11, 18]
    assert parse_reply("no json here") == {}


def test_select_keyframes_pairwise_is_default_and_logs_verdicts(tmp_path):
    path = tmp_path / "v.mp4"
    _video(path)
    p = _params()
    assert p.review == "pairwise"
    client = _FakeClient()
    an = select_keyframes(path, p, picker=Picker(p, cache_dir=tmp_path / "cache", client=client))
    sel = an.params["selection"]
    assert sel["review"] == "pairwise" and sel["picked_s"] == sel["candidates_s"][::2]
    assert an.timestamps == sel["picked_s"]  # the fake calls every pick "new"
    assert [r["action"] for r in sel["pairwise"]] == ["keep"] * 3
    assert sel["calls"] == 6  # 3 pick grids + 3 pair images
    assert len(sel["replies"]) == 3  # replies lists the grid calls only
    # cached: pick replies and verdicts are both reused
    again = Picker(p, cache_dir=tmp_path / "cache", client=client)
    assert select_keyframes(path, p, picker=again).timestamps == an.timestamps
    assert len(client.prompts) == 6


def test_unknown_review_is_rejected(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        select_keyframes(tmp_path / "v.mp4", _params(review="pairs"))


# ── pairwise review ───────────────────────────────────────────────────────────


class _ScriptedPicker:
    """Returns scripted verdicts in order; ignores the image."""

    def __init__(self, verdicts):
        self.verdicts = list(verdicts)

    def verdict(self, jpeg, prompt):
        return self.verdicts.pop(0)


def test_parse_verdict_tolerates_text_and_bad_json():
    assert parse_verdict('ok {"verdict": "new", "reason": "tip"} done')["verdict"] == "new"
    assert parse_verdict('{"verdict": "same", "reason": "he said "hi""}') == {"verdict": "same"}
    assert parse_verdict("no json here") == {}
    assert parse_verdict('{"verdict": "maybe"}') == {}


def test_review_pairs_keeps_only_new_and_swaps_clearer_same(monkeypatch):
    import origami_ad.keyframe_select as ks

    monkeypatch.setattr(ks, "pair_image", lambda *a, **k: b"")
    picker = _ScriptedPicker(
        [
            {"verdict": "new"},  # 1.0 kept (first state)
            {"verdict": "midfold"},  # 2.0 dropped
            {"verdict": "same", "clearer": "B"},  # 3.0 replaces 1.0
            {"verdict": "new"},  # 4.0 kept
            {"verdict": "same", "clearer": "A"},  # 5.0 dropped
        ]
    )
    kept, rows = review_pairs(None, [1.0, 2.0, 3.0, 4.0, 5.0], picker, SelectParams(), 10.0)
    assert kept == [3.0, 4.0]
    assert [r["action"] for r in rows] == ["keep", "drop", "replace previous", "keep", "drop"]


# ── keyframer choice in run_agent ─────────────────────────────────────────────


class _Called(Exception):
    pass


def _which_keyframer(monkeypatch, keyframer=None):
    """The keyframer build_auto_manifest runs (review name or "stillness"); stops there."""
    import origami_ad.keyframe_select as ks
    import origami_ad.keyframing as kf
    import origami_ad.run_agent as ra

    def select(path, params=None, **kw):
        raise _Called(params.review)

    def stillness(*a, **k):
        raise _Called("stillness")

    monkeypatch.setattr(ra, "download", lambda vid: "video.mp4")
    monkeypatch.setattr(ra, "load_env", lambda: None)
    monkeypatch.setattr(ks, "select_keyframes", select)
    monkeypatch.setattr(kf, "analyse_video", stillness)
    kwargs = {} if keyframer is None else {"keyframer": keyframer}
    try:
        ra.build_auto_manifest("vid", "heart", **kwargs)
    except _Called as e:
        return str(e)
    raise AssertionError("no keyframer was called")


def test_pairwise_is_the_default_keyframer(monkeypatch):
    from origami_ad.run_agent import KEYFRAMERS

    assert KEYFRAMERS[0] == "pairwise"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    assert _which_keyframer(monkeypatch) == "pairwise"
    assert _which_keyframer(monkeypatch, "grid") == "grid"


def test_no_api_key_falls_back_to_stillness(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert _which_keyframer(monkeypatch) == "stillness"
    assert _which_keyframer(monkeypatch, "grid") == "stillness"
