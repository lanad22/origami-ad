"""End-to-end: dataset -> manifest -> FoldingAgent loop (scripted backend) -> log -> metrics.

Needs the cached dataset and node; run with ``pytest --run-slow``.
"""

import json

import pytest

from origami_ad.eval import evaluate_run, write_metrics
from origami_ad.run_agent import build_purelandfold_manifest, run
from origami_ad.testing import ScriptedBackend, diagonal_fold_script

pytestmark = pytest.mark.slow


def test_scripted_run_on_yacht(tmp_path):
    spec = build_purelandfold_manifest("yacht", root=tmp_path / "keyframes")
    assert spec.manifest.exists() and len(spec.keyframes) == 5
    backend = ScriptedBackend(diagonal_fold_script(5))
    log = run(spec, results_root=tmp_path / "results", backend=backend)
    assert log["stop_reason"] == "all_frames_solved", log
    assert log["frames_solved"] == [1, 2, 3, 4, 5]
    assert log["tool_calls"]["save_checkpoint"] == 5 and log["tool_calls"]["fold"] == 1
    assert json.loads(open(log["log_path"]).read())["sequence"] == "yacht"

    res = evaluate_run(log["run_dir"], run_log=log)
    assert res["num_solved_steps"] == 5
    steps = {r["step"]: r for r in res["per_step"]}
    assert steps[1]["cv"] == 1 and steps[1]["tss"] == pytest.approx(1.0) and steps[1]["cpd"] == 0
    assert res["cv"] == 1.0
    assert "cpd_final" in res and "tss_mean" in res
    path = write_metrics(res, results_root=tmp_path / "results")
    assert path.exists()


def test_scripted_condition_b_on_downloaded_video(tmp_path):
    """Automatic keyframes -> agent loop -> DTW alignment. Needs a downloaded test video."""
    from origami_ad.run_agent import build_auto_manifest
    from origami_ad.video import video_path

    vid = "wrv0UEBOWAA"
    if video_path(vid) is None:
        pytest.skip(f"video {vid} not downloaded (data/youtube/{vid}/video.mp4)")
    spec = build_auto_manifest(vid, "ladybug", root=tmp_path / "keyframes")
    assert 8 <= len(spec.keyframes) <= 20
    backend = ScriptedBackend(diagonal_fold_script(len(spec.keyframes)))
    log = run(spec, results_root=tmp_path / "results", backend=backend)
    assert log["stop_reason"] == "all_frames_solved"
    res = evaluate_run(log["run_dir"], run_log=log)
    assert res["alignment"] == "auto_dtw"
    steps = [r["step"] for r in res["per_step"]]
    assert steps == sorted(steps) and steps[0] == 1 and steps[-1] == 13
    assert res["per_step"][0]["cpd"] == 0  # the opening square matches step 1 exactly
