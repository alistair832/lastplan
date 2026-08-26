from __future__ import annotations

import html
import json
import random

import streamlit as st
import streamlit.components.v1 as components

from fruit_education import FRUIT_EDUCATION, fruit_names
from fruit_learning_extras import get_learning_extra

FRUIT_EMOJI = {
    "Apple": "🍎",
    "Banana": "🍌",
    "Grape": "🍇",
    "Mango": "🥭",
    "Strawberry": "🍓",
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
        "find_fruit_target": None,
        "find_fruit_completed": set(),
        "last_auto_voice_token": None,
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


def auto_announce_fruit(fruit_name: str, token: str) -> None:
    _ensure_state()
    if st.session_state.last_auto_voice_token == token:
        return
    st.session_state.last_auto_voice_token = token
    phrase = json.dumps(f"This is a {fruit_name}! {fruit_name}.")
    word = json.dumps(fruit_name)
    components.html(
        f"""
        <script>
        if ('speechSynthesis' in window) {{
          window.speechSynthesis.cancel();
          const intro = new SpeechSynthesisUtterance({phrase});
          intro.rate = 0.85;
          intro.pitch = 1.08;
          intro.onend = () => {{
            const slow = new SpeechSynthesisUtterance({word});
            slow.rate = 0.55;
            slow.pitch = 1.05;
            window.speechSynthesis.speak(slow);
          }};
          window.speechSynthesis.speak(intro);
        }}
        </script>
        """,
        height=0,
    )


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


def show_growth_animation(fruit_name: str) -> None:
    extra = get_learning_extra(fruit_name)
    icons = extra["growth_icons"]
    stages = [
        ("Seed / start", icons[0] if len(icons) > 0 else "🌱"),
        ("Plant", icons[1] if len(icons) > 1 else "🌿"),
        ("Flower", icons[2] if len(icons) > 2 else "🌸"),
        ("Baby fruit", icons[3] if len(icons) > 3 else "🟢"),
        ("Ripe fruit", FRUIT_EMOJI[fruit_name]),
    ]
    blocks = ""
    for index, (label, icon) in enumerate(stages):
        arrow = "<div class='arrow'>➜</div>" if index < len(stages) - 1 else ""
        blocks += f"""<div class="grow-card" style="animation-delay:{index * 0.18}s">
          <div class="grow-icon">{icon}</div><div class="grow-label">{html.escape(label)}</div>
        </div>{arrow}"""
    components.html(
        f"""
        <style>
        .grow-wrap{{font-family:Arial,sans-serif;display:flex;align-items:center;gap:8px;flex-wrap:wrap}}
        .grow-card{{min-width:105px;flex:1;text-align:center;padding:12px 8px;border-radius:18px;background:#f7fbf2;border:1px solid #d7e6cd;animation:pop 1.8s ease-in-out infinite}}
        .grow-icon{{font-size:42px}} .grow-label{{font-size:14px;font-weight:700;margin-top:5px}}
        .arrow{{font-size:26px;color:#777}}
        @keyframes pop{{0%,100%{{transform:scale(1)}}50%{{transform:scale(1.05)}}}}
        @media (prefers-reduced-motion: reduce){{.grow-card{{animation:none}}}}
        </style>
        <div style="font-family:Arial,sans-serif;font-size:20px;font-weight:800;margin-bottom:10px">🌱 How does it grow?</div>
        <div class="grow-wrap">{blocks}</div>
        """,
        height=150,
    )


def show_quick_recipe_mode(fruit_name: str) -> None:
    info = FRUIT_EDUCATION[fruit_name]
    st.markdown(f"### 👩‍🍳 What can we make with {fruit_name.lower()}?")
    st.warning("👨‍👩‍👧 **Ask an adult to help you!** Adults handle cutting, blending, heat, allergies, and choking safety.")
    recipes = [("🥪", "Food", info["food"]), ("🥤", "Drink", info["drink"]), ("🍨", "Dessert", info["dessert"])]
    cols = st.columns(3)
    for col, (icon, kind, recipe) in zip(cols, recipes):
        with col:
            st.markdown(f"## {icon}")
            st.markdown(f"**{kind}: {recipe['name']}**")
            st.caption(" → ".join(recipe["steps"][:3]))


def show_picture_quiz(fruit_name: str, namespace: str) -> None:
    _ensure_state()
    fruits = fruit_names()
    distractors = [name for name in fruits if name != fruit_name]
    rnd = random.Random(f"{namespace}:{fruit_name}")
    choices = [fruit_name] + rnd.sample(distractors, 2)
    rnd.shuffle(choices)
    st.markdown(f"### 🎮 Which one is the {fruit_name.lower()}?")
    st.write("Tap the correct fruit picture.")
    cols = st.columns(3)
    for col, choice in zip(cols, choices):
        with col:
            if st.button(f"{FRUIT_EMOJI[choice]}  {choice}", key=f"pic_quiz_{namespace}_{choice}", use_container_width=True):
                if choice == fruit_name:
                    award_star(f"quiz:{namespace}:{fruit_name}", 1)
                    st.success("⭐ Correct! Great job!")
                    st.balloons()
                else:
                    st.info(f"Good try! Look for the {FRUIT_EMOJI[fruit_name]} {fruit_name}.")


def show_random_find_game() -> None:
    _ensure_state()
    st.markdown("### 🎯 Find a Fruit Game")
    if st.session_state.find_fruit_target is None:
        if st.button("🎲 Give me a fruit to find", use_container_width=True):
            st.session_state.find_fruit_target = random.choice(fruit_names())
            st.rerun()
        return
    target = st.session_state.find_fruit_target
    st.info(f"Can you find a **{FRUIT_EMOJI[target]} {target}**? Take a photo and scan it!")
    left, right = st.columns(2)
    with left:
        if st.button("🔄 Pick another fruit", use_container_width=True):
            choices = [f for f in fruit_names() if f != target]
            st.session_state.find_fruit_target = random.choice(choices)
            st.rerun()
    with right:
        if st.button("✖️ Stop game", use_container_width=True):
            st.session_state.find_fruit_target = None
            st.rerun()


def check_find_game(fruit_name: str, scan_token: str) -> bool:
    _ensure_state()
    target = st.session_state.find_fruit_target
    if not target or target != fruit_name:
        return False
    event = f"find:{scan_token}:{fruit_name}"
    if event not in st.session_state.find_fruit_completed:
        st.session_state.find_fruit_completed.add(event)
        award_star(event, 2)
    st.session_state.find_fruit_target = None
    return True


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
