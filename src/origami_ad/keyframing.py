"""Automatic keyframe proposal from a tutorial video (Condition B).

A fold step in a tutorial ends with a rest: the paper stops changing and the hands
either leave or pause. We detect those rests from two cheap per-frame signals and take
one keyframe per rest:

* ``motion``  fraction of hand-free pixels that changed between consecutive downscaled
              grey frames, inside the paper region when one is given;
* ``hands``   MediaPipe Hands (2-D landmarks, CPU): a per-frame hand mask and how much of
              the paper region it covers.

Hands are masked out (dilated convex hull of the landmarks plus the forearm direction)
before any pixel comparison, so hands resting on or hovering over the paper do not count
as paper motion. A frame is *still* when the fraction of hand-free pixels that changed
since the previous sample is below ``motion_thresh``. Runs of still frames longer than
``min_still_s`` are rest segments; within each we pick the frame with the least hand
coverage, then the least motion. Consecutive proposals whose hand-free pixels differ by
less than ``dedupe_thresh`` show the same paper state and are merged, which removes
pauses where the narrator talks without folding. With ``intro_policy="first_hands"``
everything before the hands first appear (title cards, the finished model) is dropped and
that first appearance becomes keyframe 1.

HaMeR (3-D hand meshes) can replace the hand signal later behind the same
``HandSignal`` interface; keyframing itself does not need 3-D.

Public API::

    propose_keyframes(video_path) -> list[float]            # timestamps in seconds
    analyse_video(video_path, ...) -> KeyframeAnalysis       # signals + proposals + debug
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from origami_ad.paths import DATA_DIR

HAND_MODEL = DATA_DIR / "models" / "hand_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/"
    "float16/1/hand_landmarker.task"
)


@dataclass
class KeyframeParams:
    sample_fps: float = 5.0  # analysis rate
    width: int = 320  # analysis resolution
    #: a pixel counts as changed when its blurred grey value moves by more than this
    pixel_delta: int = 20
    #: paper is still when fewer than this fraction of hand-free pixels changed per sample
    motion_thresh: float = 0.004
    #: minimum rest length
    min_still_s: float = 0.4
    #: two rests show the same state when fewer than this fraction of hand-free pixels differ;
    #: the effective threshold is max(dedupe_thresh, dedupe_rel * typical fold change), where
    #: the typical fold change is the median of the largest ``dedupe_top_k`` changes in the video
    dedupe_thresh: float = 0.010
    dedupe_rel: float = 0.35
    dedupe_top_k: int = 8
    #: hand mask dilation, as a fraction of the analysis width
    hand_dilate: float = 0.05
    paper_region: tuple[float, float, float, float] | None = None  # x0,y0,x1,y1 in [0,1]
    max_hands: int = 2
    use_hands: bool = True
    #: "first_hands": drop everything before the hands first appear, then open with the
    #: calmest frame between that moment and the first rest when it still differs from the
    #: first rest (the flat square being put down); None keeps everything from t=0
    intro_policy: str | None = "first_hands"
    #: "one_after_hands": once the hands have left for good keep a single proposal (the
    #: finished model), dropping later presentation shots; None keeps them all
    outro_policy: str | None = "one_after_hands"


@dataclass
class Proposal:
    timestamp_s: float
    sample_index: int
    segment_start_s: float
    segment_end_s: float
    motion: float
    hands: int
    hand_cover: float
    change_from_prev: float = 0.0


@dataclass
class KeyframeAnalysis:
    video: str
    duration_s: float
    sample_fps: float
    params: dict[str, Any]
    motion: list[float]
    hands: list[int]
    hand_cover: list[float]
    still: list[bool]
    proposals: list[Proposal] = field(default_factory=list)

    @property
    def timestamps(self) -> list[float]:
        return [p.timestamp_s for p in self.proposals]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def load(cls, path: Path) -> KeyframeAnalysis:
        d = json.loads(path.read_text())
        d["proposals"] = [Proposal(**p) for p in d["proposals"]]
        return cls(**d)


# ── Hand signal ───────────────────────────────────────────────────────────────


class HandSignal:
    """Per-frame hand landmarks via MediaPipe Tasks. ``detect`` returns [] when unavailable."""

    def __init__(self, model_path: Path = HAND_MODEL, max_hands: int = 2) -> None:
        self.available = False
        self.error: str | None = None
        if not model_path.exists():
            _download_hand_model(model_path)
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mpt
            from mediapipe.tasks.python import vision

            self._mp = mp
            opts = vision.HandLandmarkerOptions(
                base_options=mpt.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=max_hands,
                min_hand_detection_confidence=0.4,
                min_hand_presence_confidence=0.4,
                min_tracking_confidence=0.4,
            )
            self._lm = vision.HandLandmarker.create_from_options(opts)
            self.available = True
        except Exception as exc:  # mediapipe missing or model unreadable
            self.error = f"{type(exc).__name__}: {exc}"

    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> list[np.ndarray]:
        """List of (21, 2) arrays of normalised landmark x,y per detected hand."""
        if not self.available:
            return []
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self._lm.detect_for_video(img, timestamp_ms)
        return [np.array([[p.x, p.y] for p in hand]) for hand in res.hand_landmarks]


def _download_hand_model(path: Path) -> None:
    import urllib.request

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(HAND_MODEL_URL, path)
    except Exception:
        pass


# ── Masks and differences ─────────────────────────────────────────────────────


def hand_mask(hands: list[np.ndarray], shape: tuple[int, int], dilate_px: int) -> np.ndarray:
    """Boolean mask of pixels covered by hands: dilated convex hull of each hand's landmarks."""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for hand in hands:
        pts = np.round(hand * [w, h]).astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 1)
        # the forearm continues past the wrist (landmark 0) away from the middle finger base (9)
        wrist, mid = pts[0].astype(float), pts[9].astype(float)
        direction = wrist - mid
        norm = np.linalg.norm(direction)
        if norm > 0:
            far = wrist + direction / norm * max(h, w)
            cv2.line(
                mask,
                tuple(wrist.astype(int)),
                tuple(far.astype(int)),
                1,
                thickness=max(1, dilate_px),
            )
    if dilate_px > 0 and mask.any():
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        mask = cv2.dilate(mask, k)
    return mask.astype(bool)


def region_mask(
    shape: tuple[int, int], region: tuple[float, float, float, float] | None
) -> np.ndarray:
    h, w = shape
    if region is None:
        return np.ones((h, w), dtype=bool)
    m = np.zeros((h, w), dtype=bool)
    x0, y0, x1, y1 = region
    m[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)] = True
    return m


def changed_fraction(a: np.ndarray, b: np.ndarray, valid: np.ndarray, pixel_delta: int) -> float:
    """Fraction of *valid* pixels whose grey value differs by more than ``pixel_delta``."""
    n = int(valid.sum())
    if n == 0:
        return 0.0
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16)) > pixel_delta
    return float((diff & valid).sum() / n)


# ── Analysis ──────────────────────────────────────────────────────────────────


@dataclass
class _Sample:
    gray: np.ndarray
    hands_mask: np.ndarray
    n_hands: int


def analyse_video(video_path: str | Path, params: KeyframeParams | None = None) -> KeyframeAnalysis:
    p = params or KeyframeParams()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = n_src / src_fps if n_src else 0.0
    step = max(1, int(round(src_fps / p.sample_fps)))
    sample_fps = src_fps / step
    dilate_px = int(round(p.hand_dilate * p.width))

    hands_sig = HandSignal(max_hands=p.max_hands) if p.use_hands else None

    samples: list[_Sample] = []
    motion: list[float] = []
    region: np.ndarray | None = None
    i = k = 0
    while True:
        if not cap.grab():
            break
        if i % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            h0, w0 = frame.shape[:2]
            small = cv2.resize(
                frame, (p.width, int(h0 * p.width / w0)), interpolation=cv2.INTER_AREA
            )
            gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            if region is None:
                region = region_mask(gray.shape, p.paper_region)
            hands = []
            if hands_sig is not None and hands_sig.available:
                hands = hands_sig.detect(
                    cv2.cvtColor(small, cv2.COLOR_BGR2RGB), int(k / sample_fps * 1000)
                )
            hm = (
                hand_mask(hands, gray.shape, dilate_px)
                if hands
                else np.zeros(gray.shape, dtype=bool)
            )
            cur = _Sample(gray=gray, hands_mask=hm, n_hands=len(hands))
            if samples:
                prev = samples[-1]
                valid = region & ~hm & ~prev.hands_mask
                motion.append(changed_fraction(gray, prev.gray, valid, p.pixel_delta))
            else:
                motion.append(0.0)
            samples.append(cur)
            k += 1
        i += 1
    cap.release()
    n = len(samples)
    if n == 0:
        raise RuntimeError(f"no frames decoded from {video_path}")
    assert region is not None

    m = np.array(motion)
    hv = np.array([s.n_hands for s in samples])
    region_n = max(1, int(region.sum()))
    hc = np.array([float((s.hands_mask & region).sum() / region_n) for s in samples])
    still = m < p.motion_thresh

    # Rest segments -> one candidate each
    min_len = max(1, int(round(p.min_still_s * sample_fps)))
    proposals: list[Proposal] = []
    start = None
    for t in range(n + 1):
        is_still = t < n and bool(still[t])
        if is_still and start is None:
            start = t
        elif not is_still and start is not None:
            if t - start >= min_len:
                seg = np.arange(start, t)
                best = int(seg[np.lexsort((m[seg], np.round(hc[seg], 2)))[0]])
                proposals.append(
                    Proposal(
                        timestamp_s=round(best / sample_fps, 3),
                        sample_index=best,
                        segment_start_s=round(start / sample_fps, 3),
                        segment_end_s=round((t - 1) / sample_fps, 3),
                        motion=float(m[best]),
                        hands=int(hv[best]),
                        hand_cover=float(hc[best]),
                    )
                )
            start = None

    def _make(idx: int) -> Proposal:
        return Proposal(
            timestamp_s=round(idx / sample_fps, 3),
            sample_index=idx,
            segment_start_s=round(idx / sample_fps, 3),
            segment_end_s=round(idx / sample_fps, 3),
            motion=float(m[idx]),
            hands=int(hv[idx]),
            hand_cover=float(hc[idx]),
        )

    def _change(i: int, j: int) -> float:
        a, b = samples[i], samples[j]
        valid = region & ~a.hands_mask & ~b.hands_mask
        return changed_fraction(a.gray, b.gray, valid, p.pixel_delta)

    # Intro: the paper usually arrives as a flat square in the hands
    if p.intro_policy == "first_hands" and hv.any():
        first = int(np.argmax(hv > 0))
        proposals = [pr for pr in proposals if pr.sample_index >= first]
        if proposals:
            lo = min(first + int(0.3 * sample_fps), proposals[0].sample_index)
            window = np.arange(lo, proposals[0].sample_index)
            if len(window):
                calm = int(window[np.argmin(m[window] + hc[window])])
                if _change(calm, proposals[0].sample_index) >= p.dedupe_thresh:
                    proposals.insert(0, _make(calm))
        else:
            proposals = [_make(first)]

    # Outro: after the hands leave for good, keep one shot of the finished model
    if p.outro_policy == "one_after_hands" and hv.any():
        last = int(np.max(np.nonzero(hv > 0)[0]))
        tail = [pr for pr in proposals if pr.sample_index > last]
        if len(tail) > 1:
            proposals = [pr for pr in proposals if pr.sample_index <= last] + tail[:1]

    # Adaptive dedupe threshold from this video's own fold changes
    changes = [
        _change(a.sample_index, b.sample_index)
        for a, b in zip(proposals, proposals[1:], strict=False)
    ]
    top = sorted(changes, reverse=True)[: p.dedupe_top_k]
    typical = float(np.median(top)) if top else 0.0
    dedupe = max(p.dedupe_thresh, p.dedupe_rel * typical)

    # Dedupe consecutive proposals that show the same paper state
    kept: list[Proposal] = []
    for pr in proposals:
        if kept:
            change = _change(kept[-1].sample_index, pr.sample_index)
            pr.change_from_prev = round(change, 4)
            if change < dedupe:
                if pr.hand_cover < kept[-1].hand_cover:
                    pr.change_from_prev = kept[-1].change_from_prev
                    kept[-1] = pr
                continue
        kept.append(pr)

    return KeyframeAnalysis(
        video=str(video_path),
        duration_s=round(duration, 2),
        sample_fps=sample_fps,
        params={
            **asdict(p),
            "dedupe_effective": round(dedupe, 4),
            "typical_fold_change": round(typical, 4),
        },
        motion=[round(x, 4) for x in motion],
        hands=[int(x) for x in hv],
        hand_cover=[round(float(x), 3) for x in hc],
        still=[bool(x) for x in still],
        proposals=kept,
    )


def propose_keyframes(video_path: str | Path, params: KeyframeParams | None = None) -> list[float]:
    """Candidate keyframe timestamps (seconds), strictly increasing."""
    return analyse_video(video_path, params).timestamps


def contact_sheet(
    video_path: str | Path, timestamps: list[float], out: Path, cols: int = 6
) -> Path:
    """Save the proposed keyframes side by side for a quick visual check."""
    from PIL import Image, ImageDraw

    cap = cv2.VideoCapture(str(video_path))
    w, h = 256, 144
    rows = (len(timestamps) + cols - 1) // cols
    sheet = Image.new("RGB", (w * cols, (h + 16) * max(rows, 1)), "white")
    d = ImageDraw.Draw(sheet)
    for i, ts in enumerate(timestamps):
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((w, h))
        r, c = divmod(i, cols)
        sheet.paste(img, (c * w, r * (h + 16) + 16))
        d.text((c * w + 2, r * (h + 16)), f"#{i + 1}  {ts:.1f}s", fill="black")
    cap.release()
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=85)
    return out
