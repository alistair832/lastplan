from __future__ import annotations

import pandas as pd
import streamlit as st

from progress_store import (
    FRUITS,
    ensure_progress_state,
    learning_level,
    progress_json,
    progress_snapshot,
    restore_progress_json,
)

FRUIT_EMOJI = {
    "Apple": "🍎",
    "Banana": "🍌",
    "Grape": "🍇",
    "Mango": "🥭",
    "Strawberry": "🍓",
}


def show_adult_dashboard() -> None:
    ensure_progress_state()
    snapshot = progress_snapshot()
    stats = snapshot["progress_stats"]
    unlocked = len(snapshot["unlocked_fruits"])
    stars = snapshot["reward_stars"]

    st.header("📊 Parent / Teacher Dashboard")
    st.caption(
        "A simple learning summary. No child name, account, or personal profile is required."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🍓 Fruits unlocked", f"{unlocked} / {len(FRUITS)}")
    c2.metric("⭐ Stars", stars)
    c3.metric("📷 Known scans", stats["known_scans"])
    c4.metric("🎮 Quiz correct", f"{stats['quiz_correct']} / {stats['quiz_attempts']}")

    rows = []
    fruit_progress = snapshot["fruit_progress"]
    for fruit in FRUITS:
        data = fruit_progress[fruit]
        rows.append(
            {
                "Fruit": f"{FRUIT_EMOJI[fruit]} {fruit}",
                "Unlocked": "✅" if fruit in snapshot["unlocked_fruits"] else "—",
                "Scans": data["scans"],
                "Quiz": f"{data['quiz_correct']} / {data['quiz_attempts']}",
                "Activities tried": len(data["activities"]),
                "Challenge level": f"{learning_level(fruit)} / 6",
            }
        )

    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("### 💾 Fruit Passport")
    st.write(
        "Streamlit sessions can reset. An adult can save this small progress file and load it "
        "again later. It stores learning progress only — no photos and no child name."
    )

    st.download_button(
        "💾 Save Fruit Passport",
        data=progress_json().encode("utf-8"),
        file_name="fruitscan_kids_progress.json",
        mime="application/json",
        use_container_width=True,
    )

    uploaded = st.file_uploader(
        "Load a saved Fruit Passport",
        type=["json"],
        key="fruit_passport_upload",
    )
    if uploaded is not None:
        if st.button(
            "📥 Restore this progress",
            key="restore_fruit_passport",
            type="primary",
            use_container_width=True,
        ):
            ok, message = restore_progress_json(uploaded.getvalue())
            if ok:
                st.success(message)
                st.rerun()
            else:
                st.error(message)

    with st.expander("ℹ️ What is saved?", expanded=False):
        st.write(
            "Unlocked fruits, stars, scan counts, quiz results, activity participation, "
            "and adaptive challenge progress."
        )
        st.write("Camera photos are not included in the Fruit Passport.")
