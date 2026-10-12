# Origami AD, Stage 1

YouTube coverage audit for PurelandFold models and a transfer study of
[FoldingAgent](https://github.com/maya-moriya/FoldingAgent) from its clean benchmark to
in-the-wild YouTube tutorials. See `CLAUDE.md` for scope and task list.

## Setup

```bash
git clone --recurse-submodules <this repo>
cd origami-ad
conda env create -f environment.yml     # Python 3.11, ffmpeg, node, FoldingAgent + simulator, all deps
conda activate origami-ad
cp .env.example .env                    # fill in GOOGLE_API_KEY (Gemini) and YOUTUBE_API_KEY
python -m origami_ad.dataset            # downloads PurelandFold on first use, lists 27 sequences
pytest                                  # fast checks (pytest --run-slow for end-to-end)
```

Without conda: `pip install -e external/FoldingAgent "git+https://github.com/maya-moriya/FoldingAgentSimulator.git@f6d4fc6aaf4dfa2c82a8b5f7274c198498f4f122" -e .` into a Python 3.11 env, plus `ffmpeg` and `node` on PATH. MediaPipe's hand model is downloaded automatically to `data/models/` on first use.

## Commands

| Task | Command |
|---|---|
| T0 dataset check | `python -m origami_ad.dataset` |
| T1 reproduce (3 sequences) | `python scripts/reproduce.py [--model gemini-3.1-pro-preview]` |
| T2 coverage audit | `python scripts/coverage_audit.py` (needs `YOUTUBE_API_KEY`) |
| T3 manual review | `streamlit run src/origami_ad/audit/review.py` or `python -m origami_ad.audit.review --cli` |
| T4 shared opening steps | `python -m origami_ad.bases` -> `data/annotations/bases.json` |
| T6 run on a YouTube video (automatic keyframes) | `python scripts/run_youtube.py --condition B VIDEO_ID SEQUENCE` (add `--keyframes-only` to just inspect `keyframes_auto.jpg`) |
| T5/T6 manual-timestamp ablation (Condition A) | `python scripts/alignment.py new VIDEO_ID SEQ` ... then `python scripts/run_youtube.py VIDEO_ID` |
| Re-evaluate / summary table | `python scripts/evaluate.py` -> `results/summary.{csv,md}` |
| Tests | `pytest` (fast) / `pytest --run-slow` (dataset + node + scripted end-to-end run) |

## Metrics

* **CV**: Flat-Folder compiles the predicted crease pattern (`scripts/flatfolder_compile.mjs`, headless Node over the `external/flat-folder` submodule). Validated against the dataset's own `flat_folder` column: identical success flags and constraint counts; folded coordinates agree up to a rigid motion (root-face choice).
* **CPD**: Oh et al. 2015 edge + vertex bipartite matching, paper's "UP" weights, configurable unmatched-element penalty (`origami_ad.metrics.cpd`).
* **TSS / GS / CS / FFS**: reimplemented from the OrigamiSpace appendix with the published weights and penalties; sub-scores that the paper leaves under-specified are documented in `origami_ad.metrics.origamispace` and should be read as comparable across our runs rather than bit-exact with the official evaluator.

## Layout

See `CLAUDE.md`. Videos, frames, the dataset cache and FoldingAgent run directories are gitignored; only video ids, timestamps, annotations and metric summaries are committed.
