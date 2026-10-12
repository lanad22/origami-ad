"""Manual review helper for the coverage audit (T3).

Streamlit page (one reviewer): shows every PurelandFold keyframe of the model next to the
YouTube candidate and asks one question, same design or not, plus a notes field. "yes" is
stored as ``design_match = exact`` (and lands in ``matches.csv``), "no" as ``different_design``.
The other audit columns stay available in the CSV for anyone who wants to fill them later.

    streamlit run src/origami_ad/audit/review.py

A terminal fallback exists for machines without Streamlit:

    python -m origami_ad.audit.review --cli [--sequence ladybug]
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from origami_ad.audit.search import AUDIT_CSV, load_audit, print_summary, save_audit
from origami_ad.dataset import get_sequence, list_sequences

EMBED_CONTROLS = 1  # 1 = normal YouTube player bar, 0 = hide it


def thumbnail_url(video_id: str) -> str:
    return f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"


def watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def _rows_for(rows: list[dict[str, Any]], sequence: str) -> list[dict[str, Any]]:
    return [r for r in rows if r["sequence"] == sequence]


# ── Streamlit ─────────────────────────────────────────────────────────────────


def streamlit_app() -> None:
    import streamlit as st
    import streamlit.components.v1 as components

    st.set_page_config(page_title="Origami AD coverage review", layout="wide")
    rows = load_audit()
    if not rows:
        st.error(f"No audit CSV at {AUDIT_CSV}. Run scripts/coverage_audit.py first.")
        return

    sequences = [s for s in list_sequences() if _rows_for(rows, s)]
    seq = st.sidebar.selectbox("Sequence", sequences)
    only_unreviewed = st.sidebar.checkbox("Only unreviewed", value=True)
    print_view = st.sidebar.checkbox("Print view (thumbnail instead of player)", value=False)
    cands = _rows_for(rows, seq)
    if only_unreviewed:
        cands = [r for r in cands if not r.get("design_match")] or cands
    labels = [
        f"{r['video_id']}  |  {r['title'][:60]}  ({int(r['duration_s'] or 0) // 60}m)"
        for r in cands
    ]
    pick = st.sidebar.radio("Candidate", range(len(cands)), format_func=lambda i: labels[i])
    row = cands[pick]

    steps = get_sequence(seq)
    left, right = st.columns([1, 1])
    with left:
        st.subheader(f"PurelandFold: {seq} ({len(steps)} steps)")
        key = f"step_{seq}"
        if key not in st.session_state:
            st.session_state[key] = steps[-1].step
        chosen = next(s_ for s_ in steps if s_.step == st.session_state[key])
        st.image(chosen.image, caption=f"step {chosen.step}", use_container_width=True)
        per_row = 5
        for i in range(0, len(steps), per_row):
            cols = st.columns(per_row)
            for col, step in zip(cols, steps[i : i + per_row], strict=False):
                col.image(step.image, use_container_width=True)
                if col.button(
                    f"step {step.step}", key=f"btn_{seq}_{step.step}", use_container_width=True
                ):
                    st.session_state[key] = step.step
                    st.rerun()
    with right:
        st.subheader(row["title"])
        st.caption(
            f"{row['channel']}  |  {row['views']} views  |  {row['duration_s']} s  |  "
            f"captions={row['has_captions']}"
        )
        vid = row["video_id"]
        # YouTube embed. controls=0 hides the player bar; rel=0 limits "more videos" to the
        # same channel; iv_load_policy=3 hides annotations. Uploader cards and the paused
        # overlay cannot be disabled by an embed. Set EMBED_CONTROLS = 1 to get the bar back.
        if print_view:
            st.image(thumbnail_url(vid), use_container_width=True)
            st.caption(watch_url(vid))
        else:
            components.html(
                f'<iframe width="100%" height="360" src="https://www.youtube.com/embed/{vid}'
                f"?controls={EMBED_CONTROLS}&rel=0&modestbranding=1&iv_load_policy=3"
                f'&playsinline=1" frameborder="0" '
                f'allow="encrypted-media; picture-in-picture" allowfullscreen></iframe>',
                height=370,
            )

    st.markdown("---")
    with st.form("review"):
        current = row.get("design_match", "")
        idx = {"exact": 0, "different_design": 1}.get(current, 2)
        choice = st.radio(
            "Does this video fold the same design as PurelandFold?",
            ["yes", "no", "skip"],
            index=idx,
            horizontal=True,
        )
        notes = st.text_input("notes", value=row.get("notes", ""))
        if st.form_submit_button("Save"):
            values = {"notes": notes}
            if choice == "yes":
                values["design_match"] = "exact"
            elif choice == "no":
                values["design_match"] = "different_design"
            for r in rows:
                if r["sequence"] == row["sequence"] and r["video_id"] == row["video_id"]:
                    r.update(values)
            save_audit(rows)
            st.success("saved")
            st.rerun()

    with st.expander("Summary"):
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            print_summary(rows)
        st.code(buf.getvalue())


# ── CLI fallback ──────────────────────────────────────────────────────────────


def cli(sequence: str | None) -> None:
    rows = load_audit()
    if not rows:
        print(f"No audit CSV at {AUDIT_CSV}. Run scripts/coverage_audit.py first.")
        return
    todo = [
        r
        for r in rows
        if not r.get("design_match") and (sequence is None or r["sequence"] == sequence)
    ]
    print(f"{len(todo)} unreviewed candidates")
    for r in todo:
        print(
            f"\n[{r['sequence']}] {r['title']}\n  {watch_url(r['video_id'])}  "
            f"({r['duration_s']} s, {r['views']} views)"
        )
        ans = input("  same design? [y/n/enter to skip, q to quit]: ").strip().lower()
        if ans == "q":
            save_audit(rows)
            return
        if ans == "y":
            r["design_match"] = "exact"
        elif ans == "n":
            r["design_match"] = "different_design"
        r["notes"] = input("  notes: ").strip()
        save_audit(rows)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cli", action="store_true")
    p.add_argument("--sequence", default=None)
    args = p.parse_args(argv)
    if args.cli:
        cli(args.sequence)
    else:
        print(
            "Run with Streamlit:  streamlit run src/origami_ad/audit/review.py\n"
            "or use --cli for the terminal fallback."
        )


if __name__ == "__main__":
    if any("streamlit" in a for a in sys.argv[:1]) or "streamlit" in sys.modules:
        streamlit_app()
    else:
        main()
