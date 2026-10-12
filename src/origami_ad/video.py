"""Download YouTube videos with yt-dlp and extract frames with ffmpeg (T5).

Videos and frames live under ``data/youtube/{video_id}/`` and are gitignored. Only
video ids and timestamps are ever committed (see ``origami_ad.align``).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from origami_ad.paths import YOUTUBE_DIR

#: Lowest resolution at which paper edges and creases stay legible in our tests.
DEFAULT_MAX_HEIGHT = 480
#: Frame sampling rate for the agent's ``observe_movement`` filmstrips.
DEFAULT_FPS = 2.0


def video_dir(video_id: str) -> Path:
    return YOUTUBE_DIR / video_id


def video_path(video_id: str) -> Path | None:
    d = video_dir(video_id)
    if not d.is_dir():
        return None
    for p in sorted(d.glob("video.*")):
        if p.suffix.lower() in {".mp4", ".webm", ".mkv"}:
            return p
    return None


def download(video_id: str, *, max_height: int = DEFAULT_MAX_HEIGHT, force: bool = False) -> Path:
    """Download ``https://youtu.be/{video_id}`` as ``data/youtube/{video_id}/video.mp4``."""
    if shutil.which("yt-dlp") is None:
        raise RuntimeError("yt-dlp not found on PATH")
    existing = video_path(video_id)
    if existing and not force:
        return existing
    d = video_dir(video_id)
    d.mkdir(parents=True, exist_ok=True)
    fmt = (
        f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]"
        f"/best[height<={max_height}]/best"
    )
    cmd = [
        "yt-dlp",
        "-f",
        fmt,
        "--merge-output-format",
        "mp4",
        "--write-info-json",
        "--no-playlist",
        "-o",
        str(d / "video.%(ext)s"),
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    subprocess.run(cmd, check=True)
    out = video_path(video_id)
    if out is None:
        raise RuntimeError(f"yt-dlp finished but no video file in {d}")
    return out


def probe_duration(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    data = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)
    return float(data["format"]["duration"])


def extract_frames(
    video_id: str,
    *,
    fps: float = DEFAULT_FPS,
    force: bool = False,
) -> Path:
    """Extract frames to ``data/youtube/{video_id}/frames/{index:06d}.jpg`` at *fps*.

    Frame ``i`` is at ``i / fps`` seconds. The directory is laid out exactly as
    FoldingAgent expects (frames named by global index alone), so a manifest can
    point at it directly and ``observe_movement`` can sample between keyframes.
    """
    path = video_path(video_id)
    if path is None:
        raise FileNotFoundError(f"video {video_id} not downloaded; call download() first")
    frames = video_dir(video_id) / "frames"
    meta = video_dir(video_id) / "frames.json"
    if frames.is_dir() and meta.exists() and not force:
        if json.loads(meta.read_text()).get("fps") == fps:
            return frames
    if frames.is_dir():
        shutil.rmtree(frames)
    frames.mkdir(parents=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-vf",
        f"fps={fps}",
        "-q:v",
        "3",
        "-start_number",
        "0",
        str(frames / "%06d.jpg"),
    ]
    subprocess.run(cmd, check=True)
    n = len(list(frames.glob("*.jpg")))
    meta.write_text(json.dumps({"video_id": video_id, "fps": fps, "num_frames": n}, indent=2))
    return frames


def timestamp_to_index(timestamp_s: float, fps: float = DEFAULT_FPS) -> int:
    return int(round(timestamp_s * fps))


def index_to_timestamp(index: int, fps: float = DEFAULT_FPS) -> float:
    return index / fps
