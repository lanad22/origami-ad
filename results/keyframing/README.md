# Keyframing methods: what was tried and how it did

Goal (plan step 3): from a YouTube tutorial, pick frames like PurelandFold's keyframes:
one frame per fold, taken after the fold is finished, paper visible.

**Decision (2026-11-22): `keyframe_select.py` with the pairwise review is the default
keyframer.** Claude picks one frame per new state from calm 1 s candidates, then checks
each pick against the last kept keyframe, with the frames just after it, and keeps it only
if it is a new finished state. On the held-out test videos it finds 19/24 states with 1 bad
pick (F1 0.86), against 17/24 with 3 bad (0.77) for the grid review (previous default)
and 14/24 with 17 bad (0.51) for stillness. On the heart video it finds
13/14 states with 2 bad picks.

```bash
python scripts/run_youtube.py VIDEO_ID SEQUENCE --keyframes-only        # pairwise (default)
python scripts/run_youtube.py VIDEO_ID SEQUENCE --keyframer grid        # previous default
python scripts/run_youtube.py VIDEO_ID SEQUENCE --keyframer stillness   # no API calls
```

`pairwise` and `grid` are the same module (`keyframe_select.py`) with a different review
pass (`SelectParams.review`); both need `ANTHROPIC_API_KEY`, and without it the stillness
method is used. `review="none"` keeps the pick pass alone (benchmark only).

## Pairwise review: how it works

All steps after the per-frame signals live in `keyframe_select.py`; the per-frame signals
come from `keyframing.py`, which is not modified. Example numbers are
from the heart video (Z9PmOw4eMEA, 147 s, 14 labelled states).

```
Pairwise keyframing   select_keyframes()   keyframe_select.py   (SelectParams.review="pairwise")
│
├─ 0. Input
│     YouTube video ── download()                          video.py
│
├─ 1. Per-frame signals (no API)        analyse_video()     keyframing.py
│     │   shared with the stillness method; only its signals are used,
│     │   its own stillness keyframes (proposals) are ignored
│     ├─ sample at 5 fps
│     ├─ MediaPipe hands → hand mask → hand_cover per frame
│     └─ motion of the hand-free image between frames
│
├─ 2. Calm candidates (no API)          calm_candidates()   keyframe_select.py
│     ├─ cut the video into 1 s slots                (SelectParams.step_s = 1.0)
│     └─ keep the calmest frame per slot (least motion + hand_cover)
│     heart: about 147 candidates
│
├─ 3. Pass 1: pick (Claude)             pick()              keyframe_select.py
│     ├─ grids of 23 candidates, 6 columns, 250 px tiles
│     ├─ tile 0 = last keyframe kept so far (so it knows the current state)
│     ├─ prompt: keep one frame per NEW finished state      (PICK_PROMPT)
│     └─ errs on keeping too much: repeats and mid-fold frames
│     heart: 18 picks, 13/14 states covered
│
├─ 4. Pass 2: pairwise review (Claude)  review_pairs()      keyframe_select.py
│     │   (review="grid" runs review() instead: all picks in one grid)
│     ├─ for each pick B, build one image             pair_image()
│     │   ├─ top:    A = last kept keyframe | B = candidate   (512 px each)
│     │   └─ bottom: B −1 s, B +1 s, B +2 s                   (256 px, context)
│     ├─ Claude classifies B                          JUDGE_PROMPT, Picker.verdict()
│     │   ├─ "new"     → keep B
│     │   ├─ "same"    → keep whichever of A / B is clearer
│     │   └─ "midfold" → drop (also: hands hide paper, title card, colouring)
│     ├─ rule: a bad keyframe is worse than a missing one → strict
│     └─ replies cached in data/cache/keyframe_select/r2_*.json
│     heart: 16 kept (2 picks swapped for a clearer frame), 13/14 found, 2 bad
│
├─ 5. Output                            KeyframeAnalysis    (type from keyframing.py)
│     ├─ proposals (timestamps) + params.selection
│     │    (candidates, picks, every verdict + reason)
│     └─ build_auto_manifest(keyframer="pairwise")          run_agent.py
│           → data/keyframes/youtube_auto/<VIDEO_ID>/
│             keyframes.json (FoldingAgent manifest), keyframes_auto.jpg, keyframes_auto.json
│
└─ How it is scored
      ├─ labels: data/annotations/keyframe_labels.json  (7 videos + heart as "extra")
      ├─ scorer: keyframe_bench.py (found / bad / F1)
      ├─ runner: scripts/keyframe_bench.py  (stillness, pick only, grid, pairwise)
      │          → results/keyframing/<method>.json, e.g. pairwise.json
      └─ result: test 19/24 found, 1 bad, F1 0.86
```

The pick pass is meant to over-select: it should not miss a state. The pairwise pass then
removes the excess one frame at a time. Two things make it better than the grid review
(up to 36 picks at 250 px in one image, where small changes such as one
tip or one side were dropped as repeats):

- each pick is seen at 512 px next to the last kept keyframe, so small changes are visible;
- the frames 1 s and 2 s after the pick show whether the paper was still moving, which
  catches mid-fold frames.

Cost: about $0.15–0.25 per video on top of the pick pass (about $0.04 per video minute),
against $1–5 for the FoldingAgent run that follows. Cached replies make reruns free.

### Heart video, frame by frame

State numbers are the 14 labelled states in `keyframe_labels.json`; the PurelandFold step
for each is in `data/annotations/heart_ad_mapping.json` (`keyframe_target_14`). Reasons
are Claude's, shortened; all of them are in `pairwise.json`.

| Kept (s) | Scored as | Pairwise verdict |
|---|---|---|
| – | state 1 (flat square, 7–9 s) **missed** | never picked: under a "Material required" caption |
| 10.0 | state 2 | new: flat square laid as a diamond |
| 16.8 | state 3 | new: folded in half into a triangle |
| 23.4 | state 4 | new: unfolded, diagonal crease visible |
| 35.6 | state 5 | new: folded in half along a diagonal again |
| 45.0 | state 6 | new: unfolded flat, crease lines visible |
| 53.2 | state 7 | new: folded in half again, hands only pressing |
| 64.2 | state 8 | new: top point folded down |
| 72.6 | state 9 | new: right side folded down |
| 78.8 | state 10 (early view) | new: flat diamond with a centre crease |
| 85.4 | **bad: repeat** of state 10 | new: bottom folded up, caption "Flip over the model" |
| 89.6 | state 11 | new: model flipped over, heart front facing up |
| 111.4 | state 12 | same as 99.6 s but clearer, so it replaced it |
| 116.8 | optional state (not counted) | new: side corners folded in |
| 126.2 | **bad: extra** | new: heart folded in half, unfolded again right after |
| 136.8 | state 13 | same as 133.8 s but clearer, so it replaced it |
| 146.2 | state 14 | new: finished heart |

Contact sheet of these frames (not committed, images stay local):
`python scripts/keyframe_bench.py --method pairwise --split extra --sheets`, written under
`data/keyframes/bench/pairwise/`.

### Heart video, all methods

Split `extra`, so dev/test totals below are unchanged. 147 s, 14 required states: the
website-style target (PurelandFold steps 1–5, 7–12, 14, 16, 17; see
`data/annotations/heart_ad_mapping.json`).

| Method | Found | Bad | Missed |
|---|---|---|---|
| Stillness | 7/14 | 10 | diamond creases, corner to centre, both sides, flip, both tips, side corners |
| Codec I-frames | 13/14 | 11 | 1 state |
| Grid review (previous default) | 10/14 | 1 | flat square, vertical half-fold, right side up, both top tips |
| **Pairwise review (default)** | **13/14** | **2** | flat square |

Pick pass and candidates are the same for the grid and pairwise reviews; the pick pass
alone had 13/14 states. The grid review then removed three correct ones: the vertical
half-fold while hands creased it ("mid-fold"), one side folded up before the mirror fold
("mid-fold"), and both top tips folded, judged "same state, only zoomed in" as the frame
with one tip folded. Small symmetric or one-corner changes are where the grid review goes
wrong; the pairwise review kept all three.

## How methods are scored

`data/annotations/keyframe_labels.json`: 7 tutorials, hand-labelled from 1 fps contact
sheets (2 fps where fast), plus the heart video as split `extra`. Per video, a window for
every fold state (finished, before the next manipulation starts) plus optional windows
(rotations, colouring, in-hand shaping). `scripts/keyframe_bench.py` scores a method's
timestamps:

- **found**: fold states with at least one pick in their window (recall = found / states)
- **bad picks**: mid-fold frames, title cards, end screens, and second picks of a state
- **F1**: combines recall and precision = found / (found + bad)

| Split | Videos | Fold states | Notes |
|---|---|---|---|
| dev (used for every choice) | LLbY489TJu0, pOq1ZI0dqg4, Nk4af3ZU_tM, 4sZ0Qz8Uzfs | 48 | white paper on white table, wood table + zoom changes, fiddly in-hand ladybug |
| test (held out, scored once at the end) | J2Im3lshuek (yacht), OxCkeXG9LXY (ladybug), I0U6NFTSN8k | 24 | the first two are our exact design matches; jump cuts and a title card in OxCk |
| extra | Z9PmOw4eMEA (heart, origamiway.com) | 14 | exact design match; hands hold the paper in almost every finished state |

The dev videos are "different design" candidates from the audit: keyframing does not care
which model is folded, and the exact matches were too few to tune on. Two other matches
are not downloadable from here (dUCyGdFoI7o, and also 3N7EUi3-PG8, wrv0UEBOWAA,
2puh0FXvzKk, vFGg6vH1toE: "This video is not available").

## Results

| Method | Dev found | Dev bad | Dev F1 | Test found | Test bad | Test F1 |
|---|---|---|---|---|---|---|
| Stillness, repo default | 23/48 | 20 | 0.51 | 14/24 | 17 | 0.51 |
| Stillness, thresholds tuned on dev | 28/48 | 16 | 0.61 | 15/24 | 17 | 0.54 |
| Stillness + edge signal, blip smoothing, scene-aware intro (tuned on dev) | 30/48 | 19 | 0.62 | 15/24 | 24 | 0.48 |
| Codec I-frames (keyframe flag from ffprobe, no decoding) | 37/48 | 77 | 0.46 | 18/24 | 48 | 0.40 |
| Claude picks from 2 s candidates, one pass | 38/48 | 14 | 0.76 | 20/24 | 13 | 0.70 |
| Claude picks from 1 s candidates, one pass | 43/48 | 35 | 0.68 | – | – | – |
| Claude picks from 2 s candidates + grid review | 34/48 | 5 | 0.78 | – | – | – |
| Claude picks from 1 s candidates + grid review (`review="grid"`, previous default) | 38/48 | 14 | 0.76 | 17/24 | 3 | 0.77 |
| **Claude picks from 1 s candidates + pairwise review (`review="pairwise"`, default)** | **38/48** | **8** | **0.81** | **19/24** | **1** | **0.86** |

Rows marked – were dropped on dev and not run on test.

- **Pairwise review** (2026-11-22): developed on dev, test run once. Same candidates and
  pick pass as the grid review; only the review differs (see above). On dev it finds the
  same states with 6 fewer bad picks; on test it finds 2 more states with 2 fewer bad picks.
- **Grid review, 1 s vs 2 s candidates**: same dev F1, but 1 s finds more folds. A missed
  fold is worse than an extra frame downstream, because FoldingAgent would have to recover
  two folds from one keyframe pair. The grid-review row is the repo code run through
  `scripts/keyframe_bench.py`; the experiment version scored 38/48 with 12 bad on dev and
  17/24 with 5 bad on test (a few grids differ by one frame).
- **Codec I-frames** (2026-11-22, the "compression" baseline; scored with a one-off script,
  not in the repo): YouTube's encodes put an I-frame on a fixed timer, about every 5.3 s
  (6.0 s for the heart video); only the heart's 6.5 s I-frame (intro-to-folding cut) is
  placed by content. So I-frames behave like uniform sampling every ~5 s: states last long
  enough that most get one, but about two thirds of the frames land mid-fold or repeat a
  state.

## What we learned

- **Rests are the wrong unit.** Even with no merging at all, the stillness method's rest
  frames contain only 74% of the labelled states: many folds end with the hands still on
  the paper, moving straight into the next fold. A plain grid with each slot snapped to its
  calmest frame contains 99% (2 s) or 100% (1 s). So the problem is choosing, not finding.
- **Threshold tuning does not transfer.** Every classical change that helped on dev (edge
  change for white paper, median-smoothed motion, keeping a hand-free opening frame when it
  shows the same scene) was neutral or worse on test.
- **A vision model is good at "is this a new, finished state?"** It skips title cards,
  mid-fold frames and colouring without any special rules. Most of its remaining bad picks
  are repeats.
- **Compare one pair at a time, with what happens next.** A single grid of all picks hides
  small changes and cannot tell a finished fold from one in progress; judging each pick
  against the last keyframe at full size, with frames 1–2 s later, fixes both.
- White paper on a white table (pOq1) went from 2/10 states with stillness to 8/10 with
  the grid review and 9/10 with the pairwise review.

## Known misses of pairwise review

- Brief states held for about 1 s with hands moving (4sZ0 at 9–11 s and 24 s, OxCk at
  23 s and 30–34 s).
- LLbY489TJu0 misses 5 of 16 states (52–150 s); not inspected yet.
- In-hand work where the hands cover the paper (I0U6 40–54 s and 74–82 s): judged
  "midfold" and dropped. This is the intended trade-off (a missing keyframe over a wrong
  one), but it loses real states in videos folded mostly in the air.
- A first state shown under a caption (heart flat square at 7–9 s) is never picked.
- The finished model with dots drawn on may be dropped as "colouring", leaving the
  undecorated final fold instead (OxCk). For the fold program this is the right frame.

## Rerun or try a new idea

```bash
python scripts/keyframe_bench.py                         # all methods, all labelled videos
python scripts/keyframe_bench.py --method stillness      # no API calls
python scripts/keyframe_bench.py --method pairwise --split extra   # heart only
python scripts/keyframe_bench.py --method pick --method grid --method pairwise --split test
                                                         # the three review options
python scripts/keyframe_bench.py --split dev --sheets    # contact sheets under data/keyframes/bench/
```

To compare a new method, add it to `METHODS` in `scripts/keyframe_bench.py` (a function
from video path to a `KeyframeAnalysis`), tune it on `--split dev` only, then run the test
split once. To grow the benchmark, label more videos in `keyframe_labels.json` (same
format; times in seconds).

Not tried yet: HaMeR hands for the snapping signal, Gemini as the picker (team backbone),
and a cheaper picker model (Claude Haiku 5.5) to see how much accuracy it gives up.
