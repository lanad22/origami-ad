import cv2
import numpy as np
import pytest

from origami_ad.eval import auto_align_frames, dtw_align
from origami_ad.keyframing import KeyframeParams, analyse_video, hand_mask


def _synthetic_video(path, fps=10, states=4, still_s=2.0, move_s=1.0):
    """A grey 'paper' patch that jumps to a new shape `states` times, with noise-free rests."""
    w, h = 160, 120
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    rng = np.random.default_rng(0)
    for s in range(states):
        frame = np.full((h, w, 3), 200, np.uint8)
        cv2.rectangle(frame, (30 + 10 * s, 30), (120 - 10 * s, 90 + 5 * s), (0, 0, 220), -1)
        for _ in range(int(still_s * fps)):
            out.write(frame)
        if s < states - 1:
            for k in range(int(move_s * fps)):  # moving blob = activity between rests
                f = frame.copy()
                cv2.circle(f, (20 + 12 * k, 60), 14, (255, 255, 255), -1)
                f = np.clip(f.astype(int) + rng.integers(-3, 4, f.shape), 0, 255).astype(np.uint8)
                out.write(f)
    out.release()


def test_synthetic_video_yields_one_keyframe_per_rest(tmp_path):
    path = tmp_path / "synthetic.mp4"
    _synthetic_video(path)
    params = KeyframeParams(use_hands=False, intro_policy=None, outro_policy=None, sample_fps=5)
    an = analyse_video(path, params)
    assert len(an.proposals) == 4, [p.timestamp_s for p in an.proposals]
    ts = an.timestamps
    assert ts == sorted(ts)
    # one proposal inside each 2 s rest window (rests start at 0, 3, 6, 9 s)
    for i, t in enumerate(ts):
        assert 3 * i - 0.3 <= t <= 3 * i + 2.3, (i, t)


def test_hand_mask_covers_landmarks_and_forearm():
    pts = np.array([[0.5, 0.9]] + [[0.5 + 0.02 * i, 0.5 - 0.02 * i] for i in range(20)])
    m = hand_mask([pts], (100, 100), dilate_px=3)
    assert m[90, 50] and m[50, 50]
    assert m[99, 50]  # forearm continues below the wrist towards the frame edge
    assert not m[5, 5]


def test_dtw_align_diagonal_and_skips():
    cost = np.array([[0, 5, 9], [5, 0, 5], [9, 5, 0]], dtype=float)
    assert dtw_align(cost) == [(0, 0), (1, 1), (2, 2)]
    cost = np.array([[0, 5, 9], [9, 5, 0]], dtype=float)  # frame 1 skipped a GT step
    path = dtw_align(cost)
    assert path[0] == (0, 0) and path[-1] == (1, 2)


class _Step:
    def __init__(self, fold):
        self.fold = fold


def _cp(creases):
    v = [[0, 1], [1, 1], [1, 0], [0, 0]]
    e = [[0, 1], [1, 2], [2, 3], [3, 0]] + creases
    a = ["B"] * 4 + ["V"] * len(creases)
    return {"vertices_coords": v, "edges_vertices": e, "edges_assignment": a, "faces_vertices": []}


def test_auto_align_frames_maps_to_nearest_states():
    gt = {1: _Step(_cp([])), 2: _Step(_cp([[0, 2]])), 3: _Step(_cp([[0, 2], [1, 3]]))}
    preds = [_cp([]), _cp([[0, 2], [1, 3]])]  # the middle step was skipped by the video
    mapping = auto_align_frames(preds, gt)
    assert mapping == {0: 1, 1: 3}


@pytest.mark.slow
def test_mediapipe_available():
    from origami_ad.keyframing import HandSignal

    sig = HandSignal()
    assert sig.available, sig.error
