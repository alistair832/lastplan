from __future__ import annotations

import html
import json

import streamlit as st
import streamlit.components.v1 as components

from fruit_education import FRUIT_EDUCATION, fruit_names
from fruit_learning_extras import get_learning_extra


def pronunciation_button(fruit_name: str) -> None:
    info = FRUIT_EDUCATION[fruit_name]
    word = json.dumps(info["speech"])
    button_text = json.dumps(f"🔊 Hear the word: {fruit_name}")
    components.html(
        f"""
        <div style="font-family:Arial,sans-serif;display:flex;align-items:center;gap:12px;flex-wrap:wrap;padding:4px 0;">
          <button id="sayFruit" style="font-size:18px;padding:12px 18px;border:0;border-radius:14px;background:#fff3bf;color:#3b2f00;cursor:pointer;font-weight:700;min-height:46px;">
          </button>
          <span style="font-size:18px;color:#444;">Say it slowly: <strong>{html.escape(info['pronunciation'])}</strong></span>
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
            utterance.rate = 0.68;
            utterance.pitch = 1.05;
            window.speechSynthesis.speak(utterance);
          }});
        </script>
        """,
        height=78,
    )


def _seed_svg(fruit_name: str) -> str:
    if fruit_name == "Apple":
        shapes = """
        <circle cx="160" cy="105" r="78" fill="#e85d5d"/>
        <circle cx="160" cy="105" r="58" fill="#fff5dc"/>
        <ellipse cx="160" cy="83" rx="8" ry="18" fill="#7a4a21" transform="rotate(0 160 83)"/>
        <ellipse cx="182" cy="101" rx="8" ry="18" fill="#7a4a21" transform="rotate(72 182 101)"/>
        <ellipse cx="173" cy="128" rx="8" ry="18" fill="#7a4a21" transform="rotate(144 173 128)"/>
        <ellipse cx="147" cy="128" rx="8" ry="18" fill="#7a4a21" transform="rotate(216 147 128)"/>
        <ellipse cx="138" cy="101" rx="8" ry="18" fill="#7a4a21" transform="rotate(288 138 101)"/>
        <text x="280" y="92" font-size="22" font-weight="700" fill="#333">Small brown seeds</text>
        <text x="280" y="122" font-size="18" fill="#555">inside the apple core</text>
        """
    elif fruit_name == "Banana":
        dots = "".join(
            f'<circle cx="{138 + (i % 4) * 16}" cy="{88 + (i // 4) * 18}" r="4" fill="#5b4a32"/>'
            for i in range(12)
        )
        shapes = f"""
        <circle cx="165" cy="108" r="76" fill="#f7e39a" stroke="#e2c95e" stroke-width="10"/>
        <circle cx="165" cy="108" r="54" fill="#fff4c4"/>
        {dots}
        <text x="280" y="92" font-size="22" font-weight="700" fill="#333">Tiny dark dots</text>
        <text x="280" y="122" font-size="18" fill="#555">are undeveloped seed traces</text>
        """
    elif fruit_name == "Grape":
        shapes = """
        <circle cx="160" cy="108" r="76" fill="#7f5aa2"/>
        <circle cx="160" cy="108" r="58" fill="#bea1d4"/>
        <ellipse cx="145" cy="108" rx="12" ry="28" fill="#c69a62" transform="rotate(-18 145 108)"/>
        <ellipse cx="176" cy="108" rx="12" ry="28" fill="#c69a62" transform="rotate(18 176 108)"/>
        <text x="280" y="92" font-size="22" font-weight="700" fill="#333">Some grapes have seeds</text>
        <text x="280" y="122" font-size="18" fill="#555">small, hard, tan or brown</text>
        """
    elif fruit_name == "Mango":
        shapes = """
        <ellipse cx="160" cy="108" rx="94" ry="70" fill="#f5a63a"/>
        <ellipse cx="160" cy="108" rx="62" ry="40" fill="#ffd26a"/>
        <ellipse cx="160" cy="108" rx="42" ry="24" fill="#b98b57" transform="rotate(-8 160 108)"/>
        <text x="280" y="92" font-size="22" font-weight="700" fill="#333">One large flat pit</text>
        <text x="280" y="122" font-size="18" fill="#555">protects the mango seed</text>
        """
    else:
        dots = "".join(
            f'<ellipse cx="{112 + (i % 5) * 23}" cy="{75 + (i // 5) * 31}" rx="5" ry="8" fill="#f6d36d" transform="rotate({(i % 5 - 2) * 8} {112 + (i % 5) * 23} {75 + (i // 5) * 31})"/>'
            for i in range(15)
        )
        shapes = f"""
        <path d="M160 35 C100 35 75 80 92 125 C108 168 160 193 160 193 C160 193 212 168 228 125 C245 80 220 35 160 35 Z" fill="#e94f57"/>
        {dots}
        <text x="280" y="92" font-size="22" font-weight="700" fill="#333">Tiny dots outside</text>
        <text x="280" y="122" font-size="18" fill="#555">are achenes containing seeds</text>
        """

    return f"""
    <svg viewBox="0 0 540 220" role="img" aria-label="Simple diagram showing the seeds of a {html.escape(fruit_name)}" style="width:100%;max-width:680px;height:auto;">
      <rect x="4" y="4" width="532" height="212" rx="20" fill="#fffdf5" stroke="#ddd6c8" stroke-width="2"/>
      {shapes}
      <text x="280" y="166" font-size="15" fill="#777">Simple learning diagram — not to scale</text>
    </svg>
    """


def seed_visual(fruit_name: str) -> None:
    components.html(_seed_svg(fruit_name), height=250)


def recipe_card(icon: str, title: str, recipe: dict) -> None:
    st.markdown(f"#### {icon} {title}: {recipe['name']}")
    for index, step in enumerate(recipe["steps"], start=1):
        st.write(f"**{index}.** {step}")


def show_quiz(fruit_name: str, namespace: str) -> None:
    extra = get_learning_extra(fruit_name)
    quiz = extra["quiz"]
    st.markdown(f"### ⭐ {fruit_name} Mini Quiz")
    st.write("Choose one answer for each question. You can try again as many times as you like!")

    with st.form(f"quiz_{namespace}_{fruit_name}"):
        answers = []
        for index, item in enumerate(quiz, start=1):
            answer = st.radio(
                f"{index}. {item['question']}",
                item["options"],
                index=None,
                key=f"{namespace}_{fruit_name}_q_{index}",
            )
            answers.append(answer)
        submitted = st.form_submit_button("🌟 Check my answers", use_container_width=True)

    if submitted:
        if any(answer is None for answer in answers):
            st.warning("Choose an answer for every question first.")
            return

        score = 0
        for index, (answer, item) in enumerate(zip(answers, quiz), start=1):
            if answer == item["answer"]:
                score += 1
                st.success(f"{index}. Correct! {item['explanation']}")
            else:
                st.info(f"{index}. Good try! The answer is **{item['answer']}**. {item['explanation']}")

        st.markdown(f"## {'⭐' * score}{'☆' * (len(quiz) - score)}")
        st.write(f"You got **{score} out of {len(quiz)}**!")
        if score == len(quiz):
            st.balloons()
            st.success(f"Amazing! You are a {fruit_name} Star! 🌟")
        elif score >= len(quiz) - 1:
            st.success("Great learning! Try once more for every star.")
        else:
            st.info("Nice try! Look through the lesson again and have another go.")


def show_fruit_lesson(fruit_name: str, heading: bool = True) -> None:
    if fruit_name not in FRUIT_EDUCATION:
        st.info("A learning lesson is not available for this fruit yet.")
        return

    info = FRUIT_EDUCATION[fruit_name]
    extra = get_learning_extra(fruit_name)
    namespace = "learn" if heading else "scan"

    if heading:
        st.header(f"{info['emoji']} Let's Learn About {fruit_name}!")
    else:
        st.subheader(f"{info['emoji']} Now let's learn about {fruit_name}!")

    meet_tab, seed_tab, grow_tab, kitchen_tab, quiz_tab = st.tabs(
        ["👋 Meet It", "🌰 Seeds", "🌱 Grow It", "👩‍🍳 Make It", "⭐ Quiz"]
    )

    with meet_tab:
        st.markdown("### 🗣️ Say the fruit name")
        pronunciation_button(fruit_name)
        st.markdown("### 🌟 What is it?")
        st.info(info["general"])

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 👀 Look")
            st.write(f"**Colours:** {extra['colors']}")
            st.write(f"**Inside:** {extra['inside']}")
        with c2:
            st.markdown("#### ✋ Feel & taste")
            st.write(f"**Feel:** {extra['feel']}")
            st.write(f"**Taste:** {extra['taste']}")

        st.success(f"💡 **Fun fact:** {extra['fun_fact']}")
        st.caption("Taste activities should only happen with an adult who has checked allergies and age-appropriate food safety.")

    with seed_tab:
        st.markdown(f"### 🌰 {info['seed_title']}")
        seed_visual(fruit_name)
        st.write(info["seed_description"])
        st.info(f"🔎 **Little scientist mission:** {extra['mission']}")

    with grow_tab:
        st.markdown("### 🌱 How does it grow?")
        steps = info["growth_steps"]
        icons = extra["growth_icons"]
        for index, (icon, step) in enumerate(zip(icons, steps), start=1):
            left, right = st.columns([0.12, 0.88])
            with left:
                st.markdown(f"## {icon}")
            with right:
                st.markdown(f"**Step {index}**")
                st.write(step)

        st.markdown("### ☀️ What does the plant need?")
        weather, soil = st.columns(2)
        with weather:
            st.info(f"🌤️ **Weather**\n\n{info['weather']}")
        with soil:
            st.info(f"🪴 **Soil / Dirt**\n\n{info['soil']}")
        water, sun = st.columns(2)
        with water:
            st.info(f"💧 **Water**\n\n{info['water']}")
        with sun:
            st.info(f"☀️ **Sunlight**\n\n{info['sun']}")

        st.markdown("### 🌿 How do leaves make food?")
        st.write(info["photosynthesis"])
        st.success("☀️ Sunlight + 💧 Water + 🌬️ Carbon dioxide → 🍬 Plant sugar/food + 🌱 Growth")
        st.caption("This is a simple early-years model of photosynthesis. Plants also release oxygen during photosynthesis.")

    with kitchen_tab:
        st.markdown("### 👩‍🍳 Little Fruit Kitchen")
        st.warning("🧑‍🧑‍🧒 An adult must supervise. Adults handle knives, blenders, heat, allergies, and choking hazards.")
        food_tab, drink_tab, dessert_tab = st.tabs(["🥪 Food", "🥤 Drink", "🍨 Dessert"])
        with food_tab:
            recipe_card("🥪", "Make a snack", info["food"])
        with drink_tab:
            recipe_card("🥤", "Make a drink", info["drink"])
        with dessert_tab:
            recipe_card("🍨", "Make a dessert", info["dessert"])
        st.error(f"**Safety for {fruit_name}:** {info['safety']}")

    with quiz_tab:
        show_quiz(fruit_name, namespace)


def show_learning_browser() -> None:
    st.header("🎓 Fruit Learning Garden")
    st.write("Pick a fruit. Listen to its name, investigate its seeds, learn how it grows, make a simple recipe, then earn quiz stars!")

    fruit_name = st.radio(
        "Which fruit would you like to learn today?",
        fruit_names(),
        horizontal=True,
        format_func=lambda name: f"{FRUIT_EDUCATION[name]['emoji']} {name}",
        key="learning_fruit_choice",
    )
    show_fruit_lesson(fruit_name)
    st.divider()
    with st.expander("🎮 Play the all-fruits challenge", expanded=False):
        show_fruit_game()


def show_fruit_game() -> None:
    st.header("🎮 Fruit Explorer Game")
    st.write("Use what you learned to match fruits with their clues. Ask an adult or teacher for help if you need it.")

    questions = [
        {
            "question": "🍇 Which fruit grows in bunches on a climbing vine?",
            "options": ["Apple", "Grape", "Mango"],
            "answer": "Grape",
        },
        {
            "question": "🥭 Which fruit usually has one very large flat pit?",
            "options": ["Mango", "Strawberry", "Banana"],
            "answer": "Mango",
        },
        {
            "question": "🍓 Which fruit has tiny seed-containing dots on the outside?",
            "options": ["Apple", "Strawberry", "Grape"],
            "answer": "Strawberry",
        },
        {
            "question": "🍌 Which fruit grows on a giant herb instead of a woody tree?",
            "options": ["Banana", "Apple", "Mango"],
            "answer": "Banana",
        },
        {
            "question": "🍎 Which fruit has small brown seeds inside a core?",
            "options": ["Apple", "Banana", "Grape"],
            "answer": "Apple",
        },
    ]

    with st.form("fruit_explorer_game"):
        answers = []
        for index, item in enumerate(questions, start=1):
            answers.append(
                st.radio(
                    f"{index}. {item['question']}",
                    item["options"],
                    index=None,
                    key=f"game_q_{index}",
                )
            )
        submitted = st.form_submit_button("🚀 Finish the game", use_container_width=True)

    if submitted:
        if any(answer is None for answer in answers):
            st.warning("Answer every clue before finishing the game.")
            return
        score = sum(answer == item["answer"] for answer, item in zip(answers, questions))
        st.progress(score / len(questions))
        st.markdown(f"## {'⭐' * score}{'☆' * (len(questions) - score)}")
        st.write(f"You matched **{score} out of {len(questions)}** clues correctly.")
        if score == len(questions):
            st.balloons()
            st.success("Fruit Explorer Champion! You know all five fruits! 🏆")
        elif score >= 3:
            st.success("Great exploring! Visit Learn Fruits and try again for five stars.")
        else:
            st.info("Good start! Learn about one fruit at a time, then come back and play again.")
