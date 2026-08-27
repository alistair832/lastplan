from __future__ import annotations

import html

import streamlit as st

from progress_store import record_activity_open

ACTIVITY_INFO = {
    "think": {
        "emoji": "🧠",
        "title": "Think & Discover",
        "subtitle": "Look closer, predict, compare, and solve.",
        "button": "OPEN THINK & DISCOVER",
    },
    "kitchen": {
        "emoji": "👩‍🍳",
        "title": "Fruit Kitchen",
        "subtitle": "Explore food, drinks, and desserts with an adult.",
        "button": "OPEN FRUIT KITCHEN",
    },
    "games": {
        "emoji": "🎮",
        "title": "Quiz & Games",
        "subtitle": "Play picture and thinking games to earn stars.",
        "button": "OPEN QUIZ & GAMES",
    },
}


def _ensure_activity_state() -> None:
    if "activity_page" not in st.session_state:
        st.session_state.activity_page = "home"


def reset_activity_home() -> None:
    st.session_state.activity_page = "home"


def activity_home(fruit_name: str, fruit_emoji: str) -> str | None:
    _ensure_activity_state()
    selected = st.session_state.activity_page
    if selected in ACTIVITY_INFO:
        return selected

    st.markdown(
        f"## 🌟 What do you want to do with {fruit_emoji} {html.escape(fruit_name)}?"
    )
    st.caption("Choose one big activity. You can come back here anytime.")

    columns = st.columns(3)
    for column, activity_key in zip(columns, ("think", "kitchen", "games")):
        info = ACTIVITY_INFO[activity_key]
        with column:
            with st.container(border=True):
                st.markdown(
                    f"<div style='text-align:center;font-size:68px;line-height:1.1'>"
                    f"{info['emoji']}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div style='text-align:center;font-size:1.45rem;font-weight:800'>"
                    f"{html.escape(info['title'])}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div style='text-align:center;min-height:58px;margin:8px 0 14px 0'>"
                    f"{html.escape(info['subtitle'])}</div>",
                    unsafe_allow_html=True,
                )
                if st.button(
                    info["button"],
                    key=f"open_activity_{activity_key}",
                    type="primary" if activity_key == "think" else "secondary",
                    use_container_width=True,
                ):
                    record_activity_open(fruit_name, activity_key)
                    st.session_state.activity_page = activity_key
                    st.rerun()
    return None


def activity_header(activity_key: str, fruit_name: str, fruit_emoji: str) -> None:
    info = ACTIVITY_INFO[activity_key]
    back_col, title_col = st.columns([1, 4])
    with back_col:
        if st.button(
            "⬅️ Back to Activities",
            key=f"activity_back_{activity_key}",
            use_container_width=True,
        ):
            reset_activity_home()
            st.rerun()
    with title_col:
        st.markdown(
            f"## {info['emoji']} {info['title']} — {fruit_emoji} {fruit_name}"
        )


def show_plant_needs_cards(info: dict) -> None:
    st.markdown("### 🌿 What does this plant need?")
    cards = [
        ("☀️", "Sun", info["sun"]),
        ("💧", "Water", info["water"]),
        ("🪴", "Soil", info["soil"]),
        ("🌤️", "Weather", info["weather"]),
    ]
    columns = st.columns(4)
    for column, (emoji, title, text) in zip(columns, cards):
        with column:
            with st.container(border=True):
                st.markdown(
                    f"<div style='text-align:center;font-size:48px'>{emoji}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div style='text-align:center;font-weight:800;font-size:1.15rem'>"
                    f"{html.escape(title)}</div>",
                    unsafe_allow_html=True,
                )
                st.caption(text)

    st.success(
        "🌱 Plant food: ☀️ Sunlight + 💧 Water + 🌬️ Air → 🍬 Plant food + 🌿 Growth"
    )
