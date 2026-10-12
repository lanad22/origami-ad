#!/usr/bin/env python3
"""T5: create / validate alignment files and prepare videos.

python scripts/alignment.py new VIDEO_ID SEQUENCE      # write a template to fill in
python scripts/alignment.py prepare VIDEO_ID [--fps 2]  # download + extract frames
python scripts/alignment.py check [VIDEO_ID]            # validate alignment files
python scripts/alignment.py sheet VIDEO_ID --every 5    # contact sheet of frames (every N s)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _common  # noqa: F401
from origami_ad.align import Alignment, list_alignments, new_alignment_template
from origami_ad.dataset import num_steps
from origami_ad.video import DEFAULT_FPS, download, extract_frames, video_dir


def cmd_new(a) -> None:
    al = new_alignment_template(a.video_id, a.sequence, num_steps(a.sequence))
    if al.path.exists() and not a.force:
        raise SystemExit(f"{al.path} exists (use --force to overwrite)")
    # The template deliberately fails validation (timestamps -1), so write it directly.
    al.path.parent.mkdir(parents=True, exist_ok=True)
    import dataclasses
    import json

    al.path.write_text(json.dumps(dataclasses.asdict(al), indent=2) + "\n")
    print(f"wrote {al.path}; fill in timestamp_s for each step (seconds), -1 = not shown")


def cmd_prepare(a) -> None:
    path = download(a.video_id)
    frames = extract_frames(a.video_id, fps=a.fps)
    print(f"video: {path}\nframes: {frames} ({len(list(frames.glob('*.jpg')))} @ {a.fps} fps)")


def cmd_check(a) -> None:
    als = [Alignment.load(a.video_id)] if a.video_id else list_alignments()
    for al in als:
        problems = al.validate()
        done = sum(1 for k in al.keyframes if k.timestamp_s >= 0)
        status = "OK " if not problems else "BAD"
        print(
            f"{status} {al.video_id} {al.sequence:16s} {done}/{len(al.keyframes)} steps timed, "
            f"skipped={al.skipped_steps} extra={len(al.extra_steps)} {'; '.join(problems)}"
        )


def cmd_sheet(a) -> None:
    from PIL import Image, ImageDraw

    frames = video_dir(a.video_id) / "frames"
    files = sorted(frames.glob("*.jpg"))
    if not files:
        raise SystemExit("no frames; run prepare first")
    step = max(1, int(round(a.every * a.fps)))
    picks = files[::step]
    w, h = 192, 108
    cols = 8
    rows = (len(picks) + cols - 1) // cols
    sheet = Image.new("RGB", (w * cols, (h + 14) * rows), "white")
    d = ImageDraw.Draw(sheet)
    for i, f in enumerate(picks):
        r, c = divmod(i, cols)
        sheet.paste(Image.open(f).convert("RGB").resize((w, h)), (c * w, r * (h + 14) + 14))
        d.text((c * w + 2, r * (h + 14)), f"{int(f.stem) / a.fps:.1f}s", fill="black")
    out = Path(video_dir(a.video_id) / "contact_sheet.jpg")
    sheet.save(out, quality=85)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("new")
    s.add_argument("video_id")
    s.add_argument("sequence")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_new)
    s = sub.add_parser("prepare")
    s.add_argument("video_id")
    s.add_argument("--fps", type=float, default=DEFAULT_FPS)
    s.set_defaults(fn=cmd_prepare)
    s = sub.add_parser("check")
    s.add_argument("video_id", nargs="?")
    s.set_defaults(fn=cmd_check)
    s = sub.add_parser("sheet")
    s.add_argument("video_id")
    s.add_argument("--every", type=float, default=5.0)
    s.add_argument("--fps", type=float, default=DEFAULT_FPS)
    s.set_defaults(fn=cmd_sheet)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
