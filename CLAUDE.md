# Origami AD

## Goal

Generate audio descriptions (AD) of YouTube origami tutorials that blind and low-vision
(BLV) users can follow to fold the model, delivered through YouDescribe.

Two channels feed the description:

1. **FoldingAgent: what happens to the paper.** Recovers an executable fold program from
   keyframes; each step records the action taken and the resulting paper state.
   - Paper: arXiv 2609.00377 (SIGGRAPH Asia 2026)
   - Dataset: https://huggingface.co/datasets/mayaweiz/PurelandFold (CC-BY-4.0)
   - Code: https://github.com/maya-moriya/FoldingAgent (submodule `external/FoldingAgent`)
2. **3D hand reconstruction: how a person does it.** Adds fine-grained hand motion
   (which corner was grasped, where it went, when it pressed), because keyframes taken
   from YouTube will not be as clean as the hand-picked keyframes in PurelandFold.
   - Papers: https://arxiv.org/pdf/2312.05251 (HaMeR), https://arxiv.org/pdf/2508.09629
   - Code: https://github.com/geopavlakos/hamer, https://github.com/gkarv/Hand-Texture-Module

Main idea: describe tutorials with accurate, reproducible steps in natural language.
Instead of describing raw frames, recover the fold program and generate the description
from it. The description is derived from the program, not the pixels.

## Starting point: how FoldingAgent works

PurelandFold: 27 Pureland fold programs, the authors' own top-down videos, hand-picked
keyframes (337 total, one row per keyframe), ground truth for every step. Columns:
`sequence`, `step`, `image`, `cp.svg`, `cp.fold` (FOLD JSON), `flat_folder`.

For each pair of keyframes:

- The backbone model (Gemini in the paper) looks at the two photos and proposes a fold as
  simulator operations, e.g. `add_vertex(edge=[2,3], position=0.414)` then
  `fold(edge=[1,5], direction=-1)`.
- The simulator holds a geometric model of the paper, applies the operations, and renders
  a flat drawing (outline, creases, which side faces up). The resulting state has
  `vertices`, `faces`, `edges` (each typed M/V/B) and `layers` (face stacking order).
- A second call, the critic, sees a 2x2 grid: previous and current photos on top, the
  simulator's previous and proposed drawings below. It answers Match or Mismatch.
- Match: the step is saved and becomes the start state for the next keyframe.
  Mismatch: roll back and try another fold, up to five times.

Our addition: `src/origami_ad/describe.py` (CLI `scripts/describe_run.py RUN_DIR` or
`--latest`) replays a run's `checkpoints.json` and writes one plain-language sentence per
step for the action (from the operations, with vertices named by where they sit on the
paper) and one for the resulting paper (outline shape and size from the vertices,
thickness from the layer order, visible colour from which side of each face is up).
Output: `description.md` / `description.json` in the run dir. This is the first,
rule-based draft of the AD.

## Plan

### 1. Reproduce — smoke test done, team familiarisation next
- Run FoldingAgent on PurelandFold's own keyframes: `scripts/reproduce.py` (defaults
  yacht 5, ladybug 13, snake 21 keyframes); logs cost/runtime, evaluates all metrics.
- 2026-10-11 yacht run (Gemini 3.1 Pro Preview, cap 40): 5/5 frames solved, 19 iterations,
  26 tool calls, 250 s, 316K in / 7K out tokens, ~$0.72. Kept as the example run in
  `results/purelandfold/runs/` (whitelisted in `.gitignore`).
- Next: each teammate runs one other sequence of the 27 to get familiar with FoldingAgent.
- Done when: results for 3 sequences in `results/purelandfold/` roughly match the paper.

### 2. Find YouTube videos — code ready, needs YOUTUBE_API_KEY
- `python scripts/coverage_audit.py --sequence yacht` (repeatable `--sequence`,
  `--max-results`, default 10). API responses cached under `data/youtube/api_cache`.
- Manual review of `design_match` via `audit/review.py` (Streamlit + CLI).
- Target: start small, ~5–10 videos that fold the exact design.
- `python -m origami_ad.bases` writes `data/annotations/bases.json`: opening steps shared
  across sequences (signature = creases snapped to 0.01, canonical under D4 and M/V swap).
  23 shared states; the single diagonal is shared by 20 sequences.

### 3. Extract keyframes and run FoldingAgent on them — the key test
Goal: keyframes as close as possible to PurelandFold's: one frame per fold, taken after
the fold is finished, paper flat and visible, hands out of the way where possible. Then
run FoldingAgent on them and score against the benchmark. Does it still produce reasonable
programs when keyframes are not hand-picked and the camera is not the authors' setup?

**Keyframe quality decides everything downstream.** Compared on 7 hand-labelled
tutorials plus the heart video (`data/annotations/keyframe_labels.json`,
`scripts/keyframe_bench.py`); write-up, method map and numbers in
`results/keyframing/README.md`.

- Default since 2026-11-22: `keyframe_select.py` with the pairwise review
  (`--keyframer pairwise`, `SelectParams.review="pairwise"`). After the pick pass (below),
  each pick is judged against the last kept keyframe at 512 px, with frames 1 s before and 1–2 s
  after, as `new` / `same` / `midfold`, and kept only when new (for `same`, the clearer
  frame stays). Held-out test: 19/24 states, 1 bad pick; heart: 13/14, 2 bad. About
  $0.15–0.25 per video on top of the pick pass.
- Previous default (2026-11-01): same module with the grid review (`--keyframer grid`). One calm
  candidate per second (least motion + hand cover, from `keyframing`'s signals), shown to
  Claude Opus 5.5 in numbered grids with the last chosen keyframe as tile 0; it keeps one
  frame per new finished state, then one grid review over all picks drops repeats and
  mid-fold frames. Held-out test: 17/24 states, 3 bad picks (stillness: 14/24, 17 bad).
  ~$0.04 per video minute, replies cached in `data/cache/keyframe_select/`.
  Both Claude keyframers fall back to stillness without `ANTHROPIC_API_KEY`.
- Stillness method (`keyframing.py`, `--keyframer stillness`, settings in
  `KeyframeParams`): sample at 5 fps, detect
  hands with MediaPipe and mask them out (dilated landmark hull + forearm), measure how
  much of the hand-free image changes between frames, take one frame per pause where the
  paper stopped changing (rest >= 0.4 s; least hand cover, then least motion), merge
  near-duplicate pauses (adaptive threshold, 35% of the video's typical fold change).
  Intro: start at first hand appearance; outro: one shot after hands leave.
  Its rest frames contain only 74% of labelled states (folds often end without a rest), and
  threshold tuning on dev did not carry over to test.
- Run: `python scripts/run_youtube.py VIDEO_ID SEQUENCE --keyframes-only`.
  Outputs under `data/keyframes/youtube_auto/VIDEO_ID/`: `keyframes_auto.jpg` (chosen
  frames with timestamps), `keyframes_auto.json` (per-frame signals; with `pairwise` or
  `grid`, the candidates, picks and model replies under `params.selection`),
  `keyframes.json` (the manifest FoldingAgent reads).
- Tried and dropped: tuned thresholds, edge-based change, smoothed motion, scene-aware
  intro, 2 s candidates, no review pass, codec I-frames (uniform ~5 s, mostly mid-fold).
  Not yet tried: HaMeR hands for snapping, Gemini or
  Claude Haiku 5.5 as the picker.
- How to judge a new idea: add it to `scripts/keyframe_bench.py`, tune on `--split dev`,
  run test once, add a row to `results/keyframing/README.md`.
- Final check: `python scripts/run_youtube.py VIDEO_ID SEQUENCE` (condition B, default).
  `eval.py` aligns the agent's checkpoints to PurelandFold steps by DTW on CPD
  (`alignment: auto_dtw`). Compare the YouTube row with the benchmark row in
  `results/summary.md`; read `description.md` to see where it went wrong. Each run costs
  $1–5, so use it to confirm a choice, not to sweep settings.
- Condition A (manual timestamps, ablation only): `scripts/alignment.py new|prepare|sheet|check`
  then `run_youtube.py --condition A`. Skipped steps stay unaligned; `eval` reports coverage.

### 4. HaMeR — not started
- First as the hand signal for keyframing: replace MediaPipe behind the same `HandSignal`
  interface in `keyframing.py`, score with the procedure in step 3.
- Then feed hand motion to FoldingAgent as evidence for the fold choice, re-run, compare
  scores with vs without hands.
- MediaPipe model file: `data/models/hand_landmarker.task` (auto-downloaded).

## Compare with dataset ground truth

Six metrics vs ground truth plus completion rate, cost and time. Benchmark row vs YouTube
row per model, then with vs without hands. `results/summary.{csv,md}` holds the table.

| Metric | What it measures | Paper | Our yacht run |
|---|---|---|---|
| Completed | all keyframes solved | 100% | yes |
| CV | predicted state is valid, flat-foldable (Flat-Folder) | 96% | 100% |
| TSS | topology: vertices, edges, faces, crease types (0–1) | 0.90 | 1.00 |
| GS | geometry of the folded state (0–1) | 0.58 | 0.69 |
| CS | physical constraints satisfied (0–1) | 0.77 | 0.97 |
| FFS | final folded shape and layer order (0–1) | 0.39 | 0.47 |
| CPD | crease-pattern distance (lower is better) | 189 | 66 |
| Cost | per sequence | ~$8.90, 31 min, ~140 model calls | $0.72, 4 min, 19 iterations |

TSS/GS/CS/FFS are reimplemented from the OrigamiSpace appendix (`metrics/origamispace.py`);
swap in the authors' evaluator if released. CPD follows Oh et al. (`metrics/cpd.py`). CV
uses Flat-Folder headlessly (`scripts/flatfolder_compile.mjs`, submodule `external/flat-folder`).

## Repo layout

```
origami-ad/
├── CLAUDE.md
├── .env.example            # YOUTUBE_API_KEY, GEMINI_API_KEY/GOOGLE_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, HF_TOKEN
├── pyproject.toml, environment.yml
├── external/FoldingAgent/  # git submodule, unmodified
├── external/flat-folder/   # git submodule, unmodified (Node, used headlessly)
├── data/
│   ├── purelandfold/       # cached HF dataset (gitignored)
│   ├── youtube/            # downloaded videos + frames + API cache (gitignored, never committed)
│   ├── keyframes/          # manifests + frame dirs built for FoldingAgent (gitignored)
│   ├── cache/flatfolder/   # content-addressed compile results (gitignored)
│   ├── models/             # downloaded weights (gitignored)
│   └── annotations/        # audit CSV, alignment JSONs, bases.json, paper_colors.json, pricing.json (committed)
├── src/origami_ad/
│   ├── paths.py            # repo paths, .env loading
│   ├── dataset.py          # load PurelandFold, list sequences, get states
│   ├── audit/search.py     # YouTube Data API search + video details (cached)
│   ├── audit/review.py     # Streamlit/CLI helper to fill manual review columns
│   ├── bases.py            # shared opening-step detection across sequences
│   ├── video.py            # yt-dlp download, ffmpeg frame extraction
│   ├── keyframing.py       # automatic keyframes: hand-masked paper stillness (MediaPipe Hands)
│   ├── keyframe_select.py  # default keyframer: calm candidates, Claude pick pass, pairwise
│   │                       # review (or grid review, the previous default)
│   ├── keyframe_bench.py   # score keyframes against keyframe_labels.json
│   ├── align.py            # optional manual YT timestamp -> step mapping (Condition A)
│   ├── run_agent.py        # manifests + wrapper to run FoldingAgent, JSON run logs
│   ├── describe.py         # fold program -> plain-language action + paper sentences
│   ├── eval.py             # per-run metrics, DTW alignment, comparison tables
│   ├── testing.py          # scripted LLM backend for key-free end-to-end tests
│   └── metrics/            # flatfolder (CV), cpd, origamispace (TSS/GS/CS/FFS), fold_utils
├── scripts/                # reproduce.py, coverage_audit.py, alignment.py, run_youtube.py,
│                           # evaluate.py, describe_run.py, keyframe_bench.py, flatfolder_compile.mjs
├── tests/
└── results/                # <condition>/{runs,logs,metrics}; logs+metrics committed,
                            # runs gitignored except whitelisted example runs
```

## Conventions

- Python 3.11+, conda env from `environment.yml` (name `origami-ad`), type hints, `ruff`.
- All secrets via `.env` (copy `.env.example`); never hard-code keys.
- Never commit video files or frames. Store only YouTube video IDs and timestamps.
- Every agent run writes a JSON log: model, sequence, video_id, num keyframes, tool calls,
  tokens, cost, wall time, success flags.
- Model backbone is configurable (paper uses Gemini 3.1 Pro Preview; Claude and GPT also
  work). `data/annotations/pricing.json` holds per-token rates for cost estimates.
- Before writing code that touches FoldingAgent, inspect its repo. Do not assume its API.

## FoldingAgent facts (verified from the repo; keep in sync if the submodule moves)

- Entry point: `foldingagent.main.run_agent(manifest_path, *, backend=None, resume_run_dir=None)`.
  Manifest `keyframes.json`: `name`, `frames` (dir relative to the manifest), `front_color`,
  `back_color`, `keyframes` (list of global frame indices). Frames are `{index:06d}.jpg|png`.
- Model: `config.DEFAULT_MODEL`; we pass our own backend built from a model id
  (`run_agent.make_backend`). Critic/comparator follow the backend's `.model`. Output root
  is `config.OUT_DIR`, pointed at `results/<condition>/runs/`.
- `observe_movement` samples intermediate frames from the sibling `frames`/`all_frames`
  dir; PurelandFold has keyframes only, so it degrades to a two-frame strip. YouTube runs
  use the fixed-fps frame dump so it works properly.
- No metrics ship with the repo. Checkpoints store `raw_representation` with *folded*
  vertex positions; `Origami.from_representation(raw).export_cp()` gives the crease-pattern
  FOLD form used by the dataset's `cp.fold` (unit-square coordinates + `faceOrders`).
- Run output: `metadata.json` (tokens, stop_reason, frames_solved, elapsed),
  `checkpoints.json`, `figs/` (critic grids, movement strips), `llm_calls/` (every prompt
  and response), `logs/agent_log.jsonl` (`event: tool_call` rows give tool-call counts).

## Open questions

- Keyframing thresholds were tuned on two videos; re-check on new channels (patterned
  paper, oblique cameras, white paper on white tables); adjust `KeyframeParams` or set
  `paper_region`.
- Which backbone to standardise on given cost (Gemini vs Claude vs GPT)?
- Homography rectification for oblique videos in Stage 1, or only log it as a failure mode?
- Videos that skip steps: currently unaligned; revisit if multi-action transitions are needed.
