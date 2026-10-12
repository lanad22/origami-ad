"""YouTube Data API v3 coverage search for every PurelandFold sequence (T2).

Quota: ``search.list`` costs 100 units per call and the default daily quota is 10,000,
so 27 sequences x 3 templates = 81 searches = 8,100 units. Every response is cached on
disk under ``data/youtube/api_cache`` and a cached query is never re-sent.

Output: ``data/annotations/coverage_audit.csv`` with the automatic columns filled and
the manual review columns left blank (see ``origami_ad.audit.review``).
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import requests

from origami_ad.dataset import display_name, list_sequences
from origami_ad.paths import ANNOTATIONS_DIR, YOUTUBE_DIR, require_env

API_BASE = "https://www.googleapis.com/youtube/v3"
CACHE_DIR = YOUTUBE_DIR / "api_cache"
AUDIT_CSV = ANNOTATIONS_DIR / "coverage_audit.csv"
#: Only the rows reviewed as an exact design match: the experiment's input list.
MATCHES_CSV = ANNOTATIONS_DIR / "matches.csv"
MATCH_COLUMNS = [
    "sequence",
    "video_id",
    "url",
    "title",
    "channel",
    "duration_s",
    "camera_view",
    "continuous_shot",
    "narration",
    "paper",
    "notes",
]

QUERY_TEMPLATES = (
    "origami {name}",
    "easy origami {name}",
    "how to fold {name} origami",
)

AUTO_COLUMNS = [
    "sequence",
    "query",
    "video_id",
    "title",
    "channel",
    "duration_s",
    "has_captions",
    "views",
    "published_at",
    "definition",
]
MANUAL_COLUMNS = [
    "design_match",  # exact / same_model_diff_sequence / different_design / unclear
    "camera_view",  # top_down / oblique / mixed
    "continuous_shot",  # yes / mostly / no
    "narration",  # voice / text_only / none
    "paper",  # plain / two_tone / patterned
    "notes",
]
COLUMNS = AUTO_COLUMNS + MANUAL_COLUMNS

_ISO8601_DURATION = re.compile(
    r"P(?:(?P<d>\d+)D)?T?(?:(?P<h>\d+)H)?(?:(?P<m>\d+)M)?(?:(?P<s>\d+)S)?"
)


def parse_iso8601_duration(text: str) -> int:
    """``PT4M13S`` -> 253 seconds."""
    m = _ISO8601_DURATION.fullmatch(text or "")
    if not m:
        return 0
    d, h, mi, s = (int(m.group(k) or 0) for k in ("d", "h", "m", "s"))
    return ((d * 24 + h) * 60 + mi) * 60 + s


def _cache_path(endpoint: str, params: dict[str, Any]) -> Path:
    key = json.dumps({"endpoint": endpoint, **params}, sort_keys=True)
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    return CACHE_DIR / endpoint / f"{digest}.json"


def _get(endpoint: str, params: dict[str, Any], *, api_key: str) -> dict[str, Any]:
    """GET with on-disk caching. The API key is never part of the cache key."""
    path = _cache_path(endpoint, params)
    if path.exists():
        return json.loads(path.read_text())
    resp = requests.get(f"{API_BASE}/{endpoint}", params={**params, "key": api_key}, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"YouTube API {endpoint} failed ({resp.status_code}): {resp.text[:500]}")
    data = resp.json()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1))
    return data


def search_videos(query: str, *, api_key: str, max_results: int = 10) -> list[str]:
    """Video ids for a query (100 quota units, cached)."""
    data = _get(
        "search",
        {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": max_results,
            "relevanceLanguage": "en",
            "safeSearch": "none",
        },
        api_key=api_key,
    )
    return [it["id"]["videoId"] for it in data.get("items", []) if "videoId" in it.get("id", {})]


def video_details(video_ids: list[str], *, api_key: str) -> dict[str, dict[str, Any]]:
    """``videos.list`` (1 quota unit per call, up to 50 ids) -> {video_id: item}."""
    out: dict[str, dict[str, Any]] = {}
    ids = list(dict.fromkeys(video_ids))
    for i in range(0, len(ids), 50):
        chunk = ids[i : i + 50]
        data = _get(
            "videos",
            {"part": "contentDetails,snippet,statistics", "id": ",".join(chunk)},
            api_key=api_key,
        )
        for item in data.get("items", []):
            out[item["id"]] = item
    return out


def _row_from_item(sequence: str, query: str, item: dict[str, Any]) -> dict[str, Any]:
    sn = item.get("snippet", {})
    cd = item.get("contentDetails", {})
    st = item.get("statistics", {})
    return {
        "sequence": sequence,
        "query": query,
        "video_id": item["id"],
        "title": sn.get("title", ""),
        "channel": sn.get("channelTitle", ""),
        "duration_s": parse_iso8601_duration(cd.get("duration", "")),
        "has_captions": str(cd.get("caption", "")).lower() == "true",
        "views": int(st.get("viewCount", 0) or 0),
        "published_at": sn.get("publishedAt", ""),
        "definition": cd.get("definition", ""),
        **{c: "" for c in MANUAL_COLUMNS},
    }


def load_audit(path: Path = AUDIT_CSV) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_audit(rows: list[dict[str, Any]], path: Path = AUDIT_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLUMNS})
    if path == AUDIT_CSV:
        write_matches(rows)


def exact_matches(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        if r.get("design_match") != "exact":
            continue
        out.append(
            {
                **{c: r.get(c, "") for c in MATCH_COLUMNS},
                "url": f"https://www.youtube.com/watch?v={r['video_id']}",
            }
        )
    return sorted(out, key=lambda r: (r["sequence"], r["video_id"]))


def write_matches(rows: list[dict[str, Any]], path: Path = MATCHES_CSV) -> list[dict[str, Any]]:
    """Rewrite matches.csv from the audit rows (only ``design_match == exact``)."""
    matches = exact_matches(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MATCH_COLUMNS)
        w.writeheader()
        w.writerows(matches)
    return matches


def load_matches(path: Path = MATCHES_CSV) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_audit(
    sequences: list[str] | None = None,
    *,
    max_results: int = 10,
    templates: tuple[str, ...] = QUERY_TEMPLATES,
) -> list[dict[str, Any]]:
    """Search every sequence, fetch details, merge into the audit CSV.

    Existing rows (and their manual annotations) are preserved; a video found for a
    sequence under several queries is kept once, under the first query that found it.
    """
    api_key = require_env("YOUTUBE_API_KEY")
    sequences = sequences or list_sequences()
    existing = load_audit()
    by_key = {(r["sequence"], r["video_id"]): r for r in existing}

    for seq in sequences:
        name = display_name(seq)
        found: dict[str, str] = {}
        for tpl in templates:
            q = tpl.format(name=name)
            for vid in search_videos(q, api_key=api_key, max_results=max_results):
                found.setdefault(vid, q)
        details = video_details(list(found), api_key=api_key)
        for vid, q in found.items():
            item = details.get(vid)
            if item is None or (seq, vid) in by_key:
                continue
            by_key[(seq, vid)] = _row_from_item(seq, q, item)

    rows = sorted(by_key.values(), key=lambda r: (r["sequence"], -int(r["views"] or 0)))
    save_audit(rows)
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Candidates per sequence, plus how many were reviewed as exact matches."""
    summary: dict[str, dict[str, Any]] = {}
    for r in rows:
        s = summary.setdefault(
            r["sequence"], {"sequence": r["sequence"], "candidates": 0, "reviewed": 0, "exact": 0}
        )
        s["candidates"] += 1
        if r.get("design_match"):
            s["reviewed"] += 1
        if r.get("design_match") == "exact":
            s["exact"] += 1
    return sorted(summary.values(), key=lambda s: s["sequence"])


def print_summary(rows: list[dict[str, Any]]) -> None:
    print(f"{'sequence':20s} {'cands':>5s} {'revd':>5s} {'exact':>5s}")
    for s in summarize(rows):
        print(f"{s['sequence']:20s} {s['candidates']:5d} {s['reviewed']:5d} {s['exact']:5d}")
