"""Keyframes picked by a vision model from dense calm candidates (default keyframer).

``keyframing`` proposes one frame per *rest* (the paper stops changing). On our labelled
tutorials (``data/annotations/keyframe_labels.json``) a quarter of the finished fold states
never produce such a rest, and hands pausing mid-fold produce rests that are not states.
This module keeps ``keyframing``'s cheap per-frame signals but stops relying on rests:

1. **Candidates**: one frame every ``step_s`` seconds, each snapped to the calmest sample
   (least hand-free motion + hand cover) within its slot. At 1 s this covers every labelled
   state.
2. **Pick**: candidates go to a vision model in numbered grids, in time order. Tile 0 of
   each grid is the last keyframe already chosen, so the model can tell a new state from a
   re-show. It keeps one tile per new completed state (flat sheet, each finished fold or
   unfold, each flip, finished model) and skips mid-fold frames, title cards, rotations and
   colouring. This pass errs on keeping too much.
3. **Review** (``SelectParams.review``):

   * ``"pairwise"`` (default): each pick is judged against the last kept keyframe, side by
     side at ``pair_width`` px, with frames 1 s before and 1 s and 2 s after it. The model
     answers ``new`` / ``same`` / ``midfold``; only ``new`` is kept, and for ``same`` the
     clearer of the two frames stays. Strict on purpose: a wrong keyframe is worse for
     FoldingAgent than a missing one.
   * ``"grid"`` (previous default): all picks go back in one grid of up to 36 tiles and
     the model drops repeats and mid-fold frames. Small changes (one tip, one side) are
     hard to see at that size and get dropped as repeats.
   * ``"none"``: the pick pass alone.

Responses are cached under ``data/cache/keyframe_select`` keyed by image + prompt + model,
so re-running a video costs nothing. With Claude Opus 5.5 the pick pass costs about $0.04
per video minute and the pairwise review $0.15-0.25 per video. Results on the labelled
tutorials and a step-by-step map: ``results/keyframing/README.md``.

Public API::

    select_keyframes(video_path, params) -> KeyframeAnalysis   # same type as keyframing
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from origami_ad.keyframing import KeyframeAnalysis, KeyframeParams, Proposal, analyse_video
from origami_ad.paths import DATA_DIR, load_env

log = logging.getLogger(__name__)

CACHE_DIR = DATA_DIR / "cache" / "keyframe_select"
#: review pass options; the first is the default
REVIEWS = ("pairwise", "grid", "none")

PICK_PROMPT = """\
These tiles are frames from an origami tutorial video, in time order. Tile 0 (red border) \
is the keyframe already chosen just before this batch ("none" if we are at the start). \
Tiles 1-{n} are candidates about {step:g} s apart; each is labelled with its number and \
timestamp.

We need keyframes like a fold-by-fold diagram: one frame for each distinct paper state, \
taken after the fold is finished, paper flat and clearly visible.

Choose the tiles that show a NEW completed state compared with tile 0 and with the tiles \
you already chose in this batch. One tile per state: when several tiles show the same \
state, choose the one where the paper is most visible (hands moved away, paper flat, in \
focus).

Counts as a state: the flat sheet before the first fold; the result of each finished fold \
or unfold (a new crease line, a new outline, a new layer); the paper flipped over to the \
other side; the finished model.
Does not count: frames taken mid-fold (paper bent, lifted, half-folded, being creased); \
the same state moved, rotated or re-zoomed; title cards, text cards, end screens, a \
preview of the finished model shown before folding starts; drawing, colouring or gluing \
eyes on a finished shape; frames where hands hide most of the paper. If a state only \
appears in mid-fold frames, skip it.

Reply with JSON only: {{"keep": [tile numbers], "why": {{"<tile>": "<short reason>"}}}}"""

REVIEW_PROMPT = """\
These {n} tiles are the keyframes picked from an origami tutorial video, in time order, \
numbered and timestamped. They should form a fold-by-fold sequence: one frame per distinct \
paper state (the flat sheet, the result of each finished fold or unfold, each flip, the \
finished model), each taken after the fold is finished with the paper clearly visible.

Remove a tile when:
- it shows the same paper state as the tile kept before it (same outline, creases and \
visible side; moving, rotating or zooming does not make a new state). Of such a run keep \
only the tile where the paper is most visible;
- it is taken mid-fold (paper bent, lifted, half-folded), or hands hide most of the paper;
- it is not the folding at all: title or end cards, a preview of the finished model before \
folding starts, drawing, colouring or gluing eyes on a shape already shown.
Keep everything else; do not drop a real state.

Reply with JSON only: {{"keep": [tile numbers], "why_removed": {{"<tile>": "<short reason>"}}}}"""

JUDGE_PROMPT = """\
These are frames from an origami tutorial video. Top left, A: the last keyframe kept so far \
("none" at the start of the video). Top right, B: the candidate keyframe. Bottom row, small: \
the video 1 s before B, 1 s after B and 2 s after B, for context.

We are building a fold-by-fold diagram: one keyframe per completed paper state, taken after \
the fold is finished. A wrong keyframe (taken during a fold) is worse than a missing one.

Classify B:
- "new": B shows a completed paper state that differs from A: a new crease, a new outline, \
a new layer, or the other side of the paper facing up. Small changes count: one corner, one \
tip or one side folded is a new state. Hands touching or pressing an already finished fold \
are fine.
- "same": B shows the same paper state as A. Moving, rotating or zooming the paper does not \
make a new state.
- "midfold": B is taken during a fold (paper bent, lifted or half-folded; in the context \
frames right after B the same flap is still moving into place), or hands hide the paper so \
its state cannot be read, or B is not folding at all (title or end card, a preview of the \
finished model before folding starts, drawing or colouring). The NEXT fold starting right \
after B is normal and does not make B mid-fold.

Reply with JSON only: {{"verdict": "new" | "same" | "midfold", "clearer": "A" | "B", \
"reason": "<short reason>"}}  ("clearer" only matters for "same": which frame shows that \
state more clearly)."""


@dataclass
class SelectParams:
    model: str = "claude-opus-5-5"
    effort: str = "medium"
    #: one candidate per slot of this length, snapped to the calmest sample in the slot
    step_s: float = 1.0
    #: candidates per grid (tile 0, the previous keyframe, comes on top)
    per_grid: int = 23
    cols: int = 6
    tile_width: int = 250
    #: second pass over the picks: "pairwise" (default), "grid" or "none"
    review: str = REVIEWS[0]
    #: grid review: picks per grid
    review_max_tiles: int = 36
    #: pairwise review: size of A and B, size of the context frames, their offsets from B
    pair_width: int = 512
    context_width: int = 256
    context_offsets_s: tuple[float, ...] = (-1.0, 1.0, 2.0)
    #: signals for snapping; hands and stillness only, the rest/dedupe logic is unused
    signals: KeyframeParams = field(
        default_factory=lambda: KeyframeParams(intro_policy=None, outro_policy=None)
    )


# ── Candidates and images ─────────────────────────────────────────────────────


def calm_candidates(analysis: KeyframeAnalysis, step_s: float) -> list[int]:
    """Sample indices: per ``step_s`` slot, the one with least motion + hand cover."""
    m = np.asarray(analysis.motion) + np.asarray(analysis.hand_cover)
    fps, n = analysis.sample_fps, len(m)
    out: list[int] = []
    for t in np.arange(0.0, n / fps, step_s):
        lo = int(max(0, (t - step_s / 2) * fps))
        hi = max(lo + 1, int(min(n, (t + step_s / 2) * fps)))
        k = lo + int(np.argmin(m[lo:hi]))
        if not out or k > out[-1]:
            out.append(k)
    return out


def _frame(cap: cv2.VideoCapture, t: float, width: int) -> Image.Image:
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, f = cap.read()
    if not ok:
        return Image.new("RGB", (width, width * 9 // 16), "gray")
    h = int(f.shape[0] * width / f.shape[1])
    small = cv2.resize(f, (width, h), interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))


def _jpeg(sheet: Image.Image) -> bytes:
    buf = io.BytesIO()
    sheet.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def tile_grid(
    cap: cv2.VideoCapture,
    tiles: list[tuple[str, float | None]],
    cols: int,
    width: int,
    highlight: str | None = None,
) -> bytes:
    """JPEG of labelled tiles; a None timestamp draws an empty "none" tile."""
    ims = [_frame(cap, t, width) if t is not None else None for _, t in tiles]
    h = next((im.height for im in ims if im is not None), width * 9 // 16)
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * width, rows * (h + 22)), "white")
    d = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=18)
    for k, ((label, t), im) in enumerate(zip(tiles, ims, strict=True)):
        r, c = divmod(k, cols)
        x, y = c * width, r * (h + 22)
        if im is None:
            d.text((x + 10, y + 40), "none (start of video)", fill="black", font=font)
        else:
            sheet.paste(im, (x, y + 22))
        text = label if t is None else f"{label}   {t:.1f}s"
        d.text((x + 4, y + 2), text, fill="black", font=font)
        if label == highlight:
            d.rectangle([x, y + 22, x + width - 1, y + 21 + h], outline="red", width=4)
    return _jpeg(sheet)


def pair_image(
    cap: cv2.VideoCapture, ref: float | None, cand: float, p: SelectParams, duration: float
) -> bytes:
    """A | B on top at ``pair_width``; context frames around B below at ``context_width``."""
    font = ImageFont.load_default(size=20)
    a = _frame(cap, ref, p.pair_width) if ref is not None else None
    b = _frame(cap, cand, p.pair_width)
    ctx_t = [min(max(0.0, cand + o), max(0.0, duration - 0.05)) for o in p.context_offsets_s]
    ctx = [_frame(cap, t, p.context_width) for t in ctx_t]
    top_h, ctx_h, lab = b.height, ctx[0].height, 26
    w = 2 * p.pair_width
    sheet = Image.new("RGB", (w, lab + top_h + lab + ctx_h), "white")
    d = ImageDraw.Draw(sheet)
    d.text(
        (6, 3),
        "A  last keyframe" + (f"  {ref:.1f}s" if ref is not None else ""),
        fill="black",
        font=font,
    )
    if a is None:
        d.text((40, lab + top_h // 2), "none (start of video)", fill="black", font=font)
    else:
        sheet.paste(a, (0, lab))
    d.text((p.pair_width + 6, 3), f"B  candidate  {cand:.1f}s", fill="black", font=font)
    sheet.paste(b, (p.pair_width, lab))
    d.rectangle([p.pair_width, lab, w - 1, lab + top_h - 1], outline="red", width=4)
    y = lab + top_h
    for k, (o, im) in enumerate(zip(p.context_offsets_s, ctx, strict=True)):
        x = k * p.context_width
        d.text((x + 6, y + 3), f"B {o:+g}s", fill="black", font=font)
        sheet.paste(im, (x, y + lab))
    return _jpeg(sheet)


# ── Model calls ───────────────────────────────────────────────────────────────


class Picker:
    """Claude calls for every pass (cached): ``keep`` for grids, ``verdict`` for pairs."""

    def __init__(self, p: SelectParams, cache_dir: Path = CACHE_DIR, client: Any = None) -> None:
        self.p = p
        self.cache_dir = cache_dir
        self._client = client
        self.calls: list[dict[str, Any]] = []

    def client(self) -> Any:
        if self._client is None:
            import anthropic

            load_env()
            self._client = anthropic.Anthropic()
        return self._client

    def keep(self, jpeg: bytes, prompt: str, n_tiles: int) -> list[int]:
        """Tile numbers the model keeps from a grid (pick pass and grid review)."""
        key = hashlib.sha1(jpeg + prompt.encode() + self.p.model.encode()).hexdigest()[:20]
        data = self._cached(f"{key}.json", lambda: self._ask_keep(jpeg, prompt))
        return sorted({int(k) for k in data.get("keep", []) if 1 <= int(k) <= n_tiles})

    def verdict(self, jpeg: bytes, prompt: str) -> dict[str, Any]:
        """``new`` / ``same`` / ``midfold`` for one A | B image (pairwise review)."""
        # "review2" / "r2_" are the pairwise review's working name; kept so cached replies match
        key = hashlib.sha1(b"review2" + jpeg + prompt.encode() + self.p.model.encode())
        return self._cached(
            f"r2_{key.hexdigest()[:20]}.json", lambda: self._ask_verdict(jpeg, prompt)
        )

    def _cached(self, name: str, ask: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        path = self.cache_dir / name
        if path.exists():
            data, cached = json.loads(path.read_text()), True
        else:
            data, cached = ask(), False
            if "unparsed_reply" not in data:  # never cache a reply we could not read
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data))
        self.calls.append({"cache": path.name, "cached": cached, **data})
        return data

    def _create(self, jpeg: bytes, prompt: str, max_tokens: int) -> tuple[Any, dict[str, int]]:
        resp = self.client().beta.messages.create(
            model=self.p.model,
            max_tokens=max_tokens,
            # on a policy decline, re-run on a fallback model chosen by the server
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": self.p.effort},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64.standard_b64encode(jpeg).decode(),
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
        return resp, usage

    @staticmethod
    def _text(resp: Any) -> str:
        return "".join(b.text for b in resp.content if b.type == "text")

    def _ask_keep(self, jpeg: bytes, prompt: str) -> dict[str, Any]:
        resp, usage = self._create(jpeg, prompt, max_tokens=16000)
        if resp.stop_reason == "refusal":
            log.warning("model declined a grid; treating it as no keyframes")
            return {"keep": [], "stop_reason": "refusal", "usage": usage}
        text = self._text(resp)
        data = parse_reply(text)
        if "keep" not in data:
            log.warning("no JSON keep list in reply: %.200s", text)
            return {"keep": [], "unparsed_reply": text, "usage": usage}
        return {**data, "usage": usage}

    def _ask_verdict(self, jpeg: bytes, prompt: str) -> dict[str, Any]:
        resp, usage = self._create(jpeg, prompt, max_tokens=4000)
        if resp.stop_reason == "refusal":
            # strict review: a frame we could not judge is not kept
            return {"verdict": "midfold", "reason": "model declined", "usage": usage}
        text = self._text(resp)
        data = parse_verdict(text)
        if not data:
            log.warning("no verdict in reply: %.200s", text)
            return {"verdict": "midfold", "unparsed_reply": text, "usage": usage}
        return {**data, "usage": usage}


def parse_reply(text: str) -> dict[str, Any]:
    """The first JSON object holding ``keep`` in a reply, tolerating text around it.

    Falls back to pulling just the ``"keep": [...]`` list when the rest of the object is
    malformed (e.g. an unescaped quote inside a reason string).
    """
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("keep"), list):
            return obj
    m = re.search(r'"keep"\s*:\s*\[([\d\s,]*)\]', text)
    if m:
        return {"keep": [int(x) for x in re.findall(r"\d+", m.group(1))]}
    return {}


def parse_verdict(text: str) -> dict[str, Any]:
    """The first JSON object with a valid ``verdict``; {} when there is none."""
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("verdict") in {"new", "same", "midfold"}:
            return obj
    m = re.search(r'"verdict"\s*:\s*"(new|same|midfold)"', text)
    return {"verdict": m.group(1)} if m else {}


# ── Passes ────────────────────────────────────────────────────────────────────


def pick(cap: cv2.VideoCapture, times: list[float], picker: Picker) -> list[float]:
    """Pass 1: batches of candidates, each shown with the last keyframe kept so far."""
    p = picker.p
    kept: list[float] = []
    for b in range(0, len(times), p.per_grid):
        batch = times[b : b + p.per_grid]
        ref = kept[-1] if kept else None
        tiles = [("0", ref)] + [(str(i + 1), t) for i, t in enumerate(batch)]
        jpeg = tile_grid(cap, tiles, p.cols, p.tile_width, highlight="0")
        prompt = PICK_PROMPT.format(n=len(batch), step=p.step_s)
        kept += [batch[k - 1] for k in picker.keep(jpeg, prompt, len(batch))]
    return kept


def review(cap: cv2.VideoCapture, times: list[float], picker: Picker) -> list[float]:
    """Pass 2, grid: drop repeats and mid-fold frames. Long lists go in chunks that overlap
    by one tile (the last frame kept from the previous chunk), so a repeat across chunks is
    seen."""
    p = picker.p
    if len(times) <= 1:
        return times
    out: list[float] = []
    rest = list(times)
    while rest:
        carry = out[-1:]
        chunk = carry + rest[: p.review_max_tiles - len(carry)]
        rest = rest[p.review_max_tiles - len(carry) :]
        tiles = [(str(i + 1), t) for i, t in enumerate(chunk)]
        jpeg = tile_grid(cap, tiles, p.cols, p.tile_width)
        keep = picker.keep(jpeg, REVIEW_PROMPT.format(n=len(chunk)), len(chunk))
        out = out[: len(out) - len(carry)] + [chunk[k - 1] for k in keep]
    return out


def review_pairs(
    cap: cv2.VideoCapture, picks: list[float], picker: Picker, p: SelectParams, duration: float
) -> tuple[list[float], list[dict[str, Any]]]:
    """Pass 2, pairwise: keep a pick only when it is a new completed state relative to the
    last kept one. Returns the kept times and one log row per pick."""
    kept: list[float] = []
    log_rows: list[dict[str, Any]] = []
    for t in picks:
        ref = kept[-1] if kept else None
        v = picker.verdict(pair_image(cap, ref, t, p, duration), JUDGE_PROMPT)
        verdict = v.get("verdict")
        action = "drop"
        if verdict == "new":
            kept.append(t)
            action = "keep"
        elif verdict == "same" and ref is not None and v.get("clearer") == "B":
            kept[-1] = t
            action = "replace previous"
        log_rows.append(
            {"t": t, "ref": ref, "verdict": verdict, "action": action, "reason": v.get("reason")}
        )
    return kept, log_rows


def select_keyframes(
    video_path: str | Path,
    params: SelectParams | None = None,
    *,
    analysis: KeyframeAnalysis | None = None,
    picker: Picker | None = None,
) -> KeyframeAnalysis:
    """Keyframes for *video_path* as a ``KeyframeAnalysis`` (signals from ``keyframing``,
    proposals from the model). ``params["selection"]`` records candidates, picks, the review
    and usage; with the pairwise review, every verdict and reason under ``"pairwise"``."""
    p = params or SelectParams()
    if p.review not in REVIEWS:
        raise ValueError(f"review must be one of {REVIEWS}, not {p.review!r}")
    an = analysis or analyse_video(video_path, p.signals)
    picker = picker or Picker(p)
    fps = an.sample_fps
    cand = calm_candidates(an, p.step_s)
    times = [i / fps for i in cand]  # exact sample times, so seeks land on the analysed frame
    duration = len(an.motion) / fps
    verdicts: list[dict[str, Any]] = []
    cap = cv2.VideoCapture(str(video_path))
    try:
        first = pick(cap, times, picker)
        if p.review == "pairwise":
            final, verdicts = review_pairs(cap, first, picker, p, duration)
        elif p.review == "grid":
            final = review(cap, first, picker)
        else:
            final = first
    finally:
        cap.release()
    times, first, final = ([round(t, 3) for t in ts] for ts in (times, first, final))

    def proposal(t: float) -> Proposal:
        i = min(len(an.motion) - 1, int(round(t * fps)))
        return Proposal(
            timestamp_s=t,
            sample_index=i,
            segment_start_s=t,
            segment_end_s=t,
            motion=an.motion[i],
            hands=an.hands[i],
            hand_cover=an.hand_cover[i],
        )

    calls = picker.calls
    selection = {
        "method": "keyframe_select",
        **{k: v for k, v in asdict(p).items() if k != "signals"},
        "candidates_s": times,
        "picked_s": first,
        "final_s": final,
        "calls": len(calls),
        "new_input_tokens": sum(c["usage"]["input_tokens"] for c in calls if not c["cached"]),
        "new_output_tokens": sum(c["usage"]["output_tokens"] for c in calls if not c["cached"]),
        "replies": [
            {k: c.get(k) for k in ("cache", "cached", "keep", "why", "why_removed")}
            for c in calls
            if "keep" in c
        ],
    }
    if p.review == "pairwise":
        selection["pairwise"] = verdicts
    return replace(
        an, params={**an.params, "selection": selection}, proposals=[proposal(t) for t in final]
    )
