from __future__ import annotations

import html
import json

import streamlit as st
import streamlit.components.v1 as components

from fruit_education import FRUIT_EDUCATION, fruit_names

FRUIT_EMOJI = {
    "Apple": "🍎",
    "Banana": "🍌",
    "Grape": "🍇",
    "Mango": "🥭",
    "Strawberry": "🍓",
}


def _pexels_photo(photo_id: int) -> str:
    return (
        f"https://images.pexels.com/photos/{photo_id}/pexels-photo-{photo_id}.jpeg"
        "?auto=compress&cs=tinysrgb&w=900&h=600&fit=crop"
    )


FRUIT_PHOTOS = {
    "Apple": _pexels_photo(7156064),
    "Banana": _pexels_photo(1166648),
    "Grape": _pexels_photo(708777),
    "Mango": _pexels_photo(7156058),
    "Strawberry": _pexels_photo(934056),
}

RECIPE_PHOTOS = {
    "Apple": {
        "food": _pexels_photo(6769478),
        "drink": _pexels_photo(11789746),
        "dessert": _pexels_photo(7994282),
    },
    "Banana": {
        "food": _pexels_photo(6823302),
        "drink": _pexels_photo(4051764),
        "dessert": _pexels_photo(15860793),
    },
    "Grape": {
        "food": _pexels_photo(6769478),
        "drink": _pexels_photo(14064559),
        "dessert": _pexels_photo(7994282),
    },
    "Mango": {
        "food": _pexels_photo(34122326),
        "drink": _pexels_photo(10047619),
        "dessert": _pexels_photo(37299594),
    },
    "Strawberry": {
        "food": _pexels_photo(2424832),
        "drink": _pexels_photo(5338141),
        "dessert": _pexels_photo(34487793),
    },
}

PICTURE_CARDS = {
    "Apple": [("🍎", "Fruit"), ("⚪", "White inside + core"), ("🌰", "Small brown seeds"), ("🌳", "Apple tree"), ("🥣", "Apple snack")],
    "Banana": [("🍌", "Fruit"), ("🟡", "Soft pale inside"), ("•••", "Tiny seed traces"), ("🌿", "Banana plant"), ("🥤", "Banana smoothie")],
    "Grape": [("🍇", "Fruit"), ("🟣", "Juicy inside"), ("🌰", "Some have seeds"), ("🌿", "Grape vine"), ("🥣", "Grape yogurt")],
    "Mango": [("🥭", "Fruit"), ("🟠", "Orange/yellow flesh"), ("🪨", "One big pit"), ("🌳", "Mango tree"), ("🥤", "Mango smoothie")],
    "Strawberry": [("🍓", "Fruit"), ("🔴", "Soft red inside"), ("✨", "Seeds outside"), ("🌱", "Small plant"), ("🍨", "Berry dessert")],
}


def _ensure_state() -> None:
    defaults = {
        "unlocked_fruits": set(),
        "reward_stars": 0,
        "rewarded_events": set(),
        "scan_corrections": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value.copy() if isinstance(value, (set, dict)) else value


def child_styles() -> None:
    st.markdown(
        """
        <style>
        .stButton > button {
            min-height: 58px;
            border-radius: 16px;
            font-size: 1.08rem;
            font-weight: 700;
        }
        div[data-testid="stFileUploader"] { border-radius: 18px; }
        div[data-testid="stMetric"] {
            border: 1px solid rgba(128,128,128,.22);
            border-radius: 16px;
            padding: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def award_star(event_key: str, amount: int = 1) -> bool:
    _ensure_state()
    if event_key in st.session_state.rewarded_events:
        return False
    st.session_state.rewarded_events.add(event_key)
    st.session_state.reward_stars += max(0, int(amount))
    return True


def unlock_fruit(fruit_name: str, event_key: str | None = None) -> bool:
    _ensure_state()
    if fruit_name not in FRUIT_EMOJI:
        return False
    was_new = fruit_name not in st.session_state.unlocked_fruits
    st.session_state.unlocked_fruits.add(fruit_name)
    if event_key:
        award_star(f"unlock:{event_key}", 1)
    return was_new


def show_progress_banner() -> None:
    _ensure_state()
    unlocked = len(st.session_state.unlocked_fruits)
    total = len(FRUIT_EMOJI)
    stars = st.session_state.reward_stars
    star_line = "⭐" * unlocked + "☆" * (total - unlocked)
    left, middle, right = st.columns(3)
    left.metric("🌟 My stars", stars)
    middle.metric("📖 Fruits learned", f"{unlocked} / {total}")
    right.metric("🏆 Collection", star_line)
    st.progress(unlocked / total if total else 0)
    st.caption(f"You learned **{unlocked} of {total} fruits!** {star_line}")


def show_collection_book() -> None:
    _ensure_state()
    st.markdown("### 📖 My Fruit Collection Book")
    cols = st.columns(5)
    for col, fruit in zip(cols, fruit_names()):
        unlocked = fruit in st.session_state.unlocked_fruits
        with col:
            st.markdown(
                f"<div style='text-align:center;font-size:52px'>{FRUIT_EMOJI[fruit] if unlocked else '❓'}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='text-align:center;font-weight:800'>{html.escape(fruit)}</div>",
                unsafe_allow_html=True,
            )
            st.caption("✅ Unlocked!" if unlocked else "Scan me to unlock")


def repeat_pronunciation_button(fruit_name: str) -> None:
    word = json.dumps(fruit_name)
    components.html(
        f"""
        <div style="font-family:Arial,sans-serif">
          <button id="repeatWord" style="width:100%;min-height:64px;border:0;border-radius:18px;font-size:22px;font-weight:800;cursor:pointer;background:#fff3bf;color:#3b2f00;">
            🔊 Say It Again
          </button>
        </div>
        <script>
        const btn = document.getElementById('repeatWord');
        btn.addEventListener('click', () => {{
          if (!('speechSynthesis' in window)) {{
            btn.textContent = '🔇 Audio not supported';
            return;
          }}
          window.speechSynthesis.cancel();
          const voice = new SpeechSynthesisUtterance({word});
          voice.rate = 0.55;
          voice.pitch = 1.05;
          window.speechSynthesis.speak(voice);
        }});
        </script>
        """,
        height=78,
    )


def show_picture_story(fruit_name: str) -> None:
    cards = PICTURE_CARDS[fruit_name]
    card_html = "".join(
        f"""<div style="flex:1;min-width:112px;text-align:center;padding:12px;border:1px solid #ddd;border-radius:18px;background:#fffdf7">
          <div style="font-size:44px;line-height:1.15">{icon}</div>
          <div style="margin-top:8px;font-size:15px;font-weight:700;color:#333">{html.escape(label)}</div>
        </div>"""
        for icon, label in cards
    )
    components.html(
        f"""
        <div style="font-family:Arial,sans-serif">
          <div style="font-size:20px;font-weight:800;margin-bottom:10px">👀 Look at the pictures</div>
          <div style="display:flex;gap:10px;flex-wrap:wrap">{card_html}</div>
        </div>
        """,
        height=175,
    )


def show_quick_recipe_mode(fruit_name: str) -> None:
    info = FRUIT_EDUCATION[fruit_name]
    st.markdown(f"### 👩‍🍳 What can we make with {fruit_name.lower()}?")
    st.warning("👨‍👩‍👧 **Ask an adult to help you!** Adults handle cutting, blending, heat, allergies, and choking safety.")

    recipes = [
        ("🥪", "Food", "food", info["food"]),
        ("🥤", "Drink", "drink", info["drink"]),
        ("🍨", "Dessert", "dessert", info["dessert"]),
    ]
    cols = st.columns(3)
    for col, (icon, kind, photo_key, recipe) in zip(cols, recipes):
        with col:
            st.image(
                RECIPE_PHOTOS[fruit_name][photo_key],
                use_container_width=True,
            )
            st.markdown(f"### {icon} {kind}")
            st.markdown(f"**{recipe['name']}**")
            with st.expander("👩‍🍳 See the easy steps", expanded=False):
                for index, step in enumerate(recipe["steps"], start=1):
                    st.write(f"**{index}.** {step}")

    st.caption("📷 Example learning photos — the recipe you make may look a little different.")


def corrected_label(scan_token: str) -> str | None:
    _ensure_state()
    return st.session_state.scan_corrections.get(scan_token)


def save_correction(scan_token: str, fruit_name: str, history_id: str | None = None) -> None:
    _ensure_state()
    st.session_state.scan_corrections[scan_token] = fruit_name
    unlock_fruit(fruit_name, f"correction:{scan_token}:{fruit_name}")
    if history_id:
        for item in st.session_state.get("camera_history", []):
            if item.get("id") == history_id:
                old = item.get("result") or {}
                item["result"] = {
                    "verified": True,
                    "label": fruit_name,
                    "confidence": float(old.get("confidence", 0.0)),
                    "dataset_similarity": float(old.get("dataset_similarity", 0.0)),
                    "verification_score": float(old.get("verification_score", 0.0)),
                    "corrected": True,
                }
                break
