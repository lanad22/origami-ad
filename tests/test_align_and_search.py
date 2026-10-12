import csv

import pytest

from origami_ad.align import Alignment, Keyframe
from origami_ad.audit.search import (
    COLUMNS,
    _row_from_item,
    parse_iso8601_duration,
    save_audit,
    summarize,
)
from origami_ad.dataset import display_name
from origami_ad.video import index_to_timestamp, timestamp_to_index


def test_duration_parsing():
    assert parse_iso8601_duration("PT4M13S") == 253
    assert parse_iso8601_duration("PT1H2M") == 3720
    assert parse_iso8601_duration("PT45S") == 45
    assert parse_iso8601_duration("") == 0


def test_display_name():
    assert display_name("gift_card_holder") == "gift card holder"
    assert display_name("rabbit_head_v1") == "rabbit head"


def test_alignment_validation():
    al = Alignment(
        video_id="abc",
        sequence="ladybug",
        keyframes=[Keyframe(1, 1.0), Keyframe(2, 5.0)],
        skipped_steps=[3],
    )
    assert al.validate() == []
    bad = Alignment(
        video_id="abc",
        sequence="ladybug",
        keyframes=[Keyframe(1, 5.0), Keyframe(2, 1.0)],
        skipped_steps=[2],
    )
    assert len(bad.validate()) == 2
    with pytest.raises(ValueError):
        bad.save()


def test_alignment_roundtrip_dict():
    al = Alignment(video_id="abc", sequence="ladybug", keyframes=[Keyframe(1, 1.0, "x")])
    d = Alignment.from_dict(
        {
            "video_id": "abc",
            "sequence": "ladybug",
            "keyframes": [{"step": 1, "timestamp_s": 1.0, "notes": "x"}],
        }
    )
    assert d == al


def test_timestamp_index_roundtrip():
    assert timestamp_to_index(4.2, 2.0) == 8
    assert index_to_timestamp(8, 2.0) == 4.0


def test_audit_row_and_csv(tmp_path):
    item = {
        "id": "vid1",
        "snippet": {"title": "Origami ladybug", "channelTitle": "C", "publishedAt": "2020"},
        "contentDetails": {"duration": "PT3M", "caption": "true", "definition": "hd"},
        "statistics": {"viewCount": "1234"},
    }
    row = _row_from_item("ladybug", "origami ladybug", item)
    assert row["duration_s"] == 180 and row["has_captions"] is True and row["views"] == 1234
    path = tmp_path / "audit.csv"
    save_audit([row], path)
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == COLUMNS
    row["design_match"] = "exact"
    s = summarize([row])
    assert s[0]["exact"] == 1 and s[0]["candidates"] == 1
