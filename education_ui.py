from __future__ import annotations

import json

import streamlit as st
import streamlit.components.v1 as components

from fruit_education import FRUIT_EDUCATION, fruit_names


def pronunciation_button(fruit_name: str) -> None:
    info = FRUIT_EDUCATION[fruit_name]
    word = json.dumps(info["speech"])
    button_text = json.dumps(f"🔊 Hear: {fruit_name}")
    components.html(
        f"""
        <div style="font-family:Arial,sans-serif;display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
          <button id="sayFruit" style="font-size:18px;padding:10px 18px;border:0;border-radius:12px;background:#fff3bf;color:#3b2f00;cursor:pointer;font-weight:700;">
          </button>
          <span style="font-size:18px;color:#444;">Say it: <strong>{info['pronunciation']}</strong></span>
        </div>
        <script>
          const button = document.getElementById('sayFruit');
          button.textContent = {button_text};
          button.addEventListener('click', () => {{
            if (!('speechSynthesis' in window)) {{
              button.textContent = 'Audio is not supported in this browser';
              return;
            }}
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance({word});
            utterance.rate = 0.72;
            utterance.pitch = 1.05;
            window.speechSynthesis.speak(utterance);
          }});
        </script>
        """,
        height=72,
    )


def recipe_card(icon: str, title: str, recipe: dict) -> None:
    st.markdown(f"#### {icon} {title}: {recipe['name']}")
    for index, step in enumerate(recipe["steps"], start=1):
        st.write(f"**{index}.** {step}")


def show_fruit_lesson(fruit_name: str, heading: bool = True) -> None:
    if fruit_name not in FRUIT_EDUCATION:
        st.info("A learning lesson is not available for this fruit yet.")
        return

    info = FRUIT_EDUCATION[fruit_name]

    if heading:
        st.header(f"{info['emoji']} Let's Learn About {fruit_name}!")
    else:
        st.subheader(f"{info['emoji']} Now let's learn about {fruit_name}!")

    st.markdown("### 🗣️ Say the fruit name")
    pronunciation_button(fruit_name)

    st.markdown("### 🌟 What is it?")
    st.info(info["general"])

    seed_col, plant_col = st.columns(2)
    with seed_col:
        st.markdown(f"### 🌰 {info['seed_title']}")
        st.write(info["seed_description"])
    with plant_col:
        st.markdown("### 🌱 What does it grow on?")
        st.metric("Plant type", info["plant"])
        st.write("Look for the seed or growing part, then see how the plant changes as it gets bigger.")

    st.markdown("### 🌱 How does it grow?")
    steps = info["growth_steps"]
    icons = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣"]
    step_columns = st.columns(len(steps))
    for index, (column, step) in enumerate(zip(step_columns, steps)):
        with column:
            st.markdown(f"#### {icons[index]}")
            st.caption(step)

    st.markdown("### ☀️ What does the plant need?")
    weather, soil, water, sun = st.columns(4)
    with weather:
        st.markdown("#### 🌤️ Weather")
        st.write(info["weather"])
    with soil:
        st.markdown("#### 🪴 Soil / Dirt")
        st.write(info["soil"])
    with water:
        st.markdown("#### 💧 Water")
        st.write(info["water"])
    with sun:
        st.markdown("#### ☀️ Sunlight")
        st.write(info["sun"])

    st.markdown("### 🌿 Photosynthesis — how leaves make food")
    st.success(info["photosynthesis"])
    st.markdown(
        "**Easy idea:** ☀️ Sunlight + 💧 Water + 🌬️ Air → 🍬 Plant sugar/food + 🌿 Growth"
    )

    st.markdown("### 👩‍🍳 Little Fruit Kitchen")
    st.caption("These activities are for learning with an adult. Adults should handle knives, blenders, heat, and choking hazards.")
    food_tab, drink_tab, dessert_tab = st.tabs(["🥪 Food", "🥤 Drink", "🍨 Dessert"])
    with food_tab:
        recipe_card("🥪", "Make a snack", info["food"])
    with drink_tab:
        recipe_card("🥤", "Make a drink", info["drink"])
    with dessert_tab:
        recipe_card("🍨", "Make a dessert", info["dessert"])

    st.warning(f"🧑‍🧑‍🧒 **Adult safety note:** {info['safety']}")

    st.markdown("### ⭐ Remember these 4 things")
    st.markdown(
        f"- **Name:** {fruit_name} ({info['pronunciation']})\n"
        f"- **Grows on:** {info['plant']}\n"
        f"- **Needs:** sunlight, water, suitable soil, and the right weather\n"
        f"- **Leaves:** use photosynthesis to make food for the plant"
    )


def show_learning_browser() -> None:
    st.header("🎓 Fruit Learning Garden")
    st.write(
        "Choose a fruit to learn its name, seeds, growing needs, photosynthesis, and a few simple food activities."
    )

    fruit_name = st.selectbox(
        "Which fruit would you like to learn today?",
        fruit_names(),
        format_func=lambda name: f"{FRUIT_EDUCATION[name]['emoji']} {name}",
    )
    show_fruit_lesson(fruit_name)
