from __future__ import annotations

import random

import streamlit as st

from progress_store import learning_level, record_adaptive_success

FRUIT_EMOJI = {
    "Apple": "🍎",
    "Banana": "🍌",
    "Grape": "🍇",
    "Mango": "🥭",
    "Strawberry": "🍓",
}

QUESTIONS = {
    "Apple": [
        ("Which shape is a good clue for an apple?", ["⭕ Round", "📏 Long", "🍇 Bunch"], "⭕ Round"),
        ("Where are apple seeds?", ["Inside a core", "Only outside", "No seeds"], "Inside a core"),
        ("Where does an apple grow?", ["On a tree", "Under water", "On a vine"], "On a tree"),
        ("Which fruit is usually longer than an apple?", ["Banana", "Grape", "Strawberry"], "Banana"),
        ("What may happen with no water?", ["The plant may wilt", "It grows faster", "It makes chocolate"], "The plant may wilt"),
        ("Which other fruit in FruitScan also grows on a tree?", ["Mango", "Grape", "Banana"], "Mango"),
    ],
    "Banana": [
        ("Which shape is a good clue for a banana?", ["📏 Long and curved", "⭕ Round", "🍇 Bunch of tiny balls"], "📏 Long and curved"),
        ("What can you see inside many eating bananas?", ["Tiny seed traces", "One hard pit", "Seeds only outside"], "Tiny seed traces"),
        ("What does a banana grow on?", ["A large herb plant", "A woody apple tree", "A grape vine"], "A large herb plant"),
        ("Which fruit is usually rounder than a banana?", ["Apple", "Banana", "None"], "Apple"),
        ("What may happen with no water?", ["The plant may wilt", "It grows normally forever", "It becomes a toy"], "The plant may wilt"),
        ("Which FruitScan fruit also does NOT grow on a woody tree?", ["Grape", "Apple", "Mango"], "Grape"),
    ],
    "Grape": [
        ("Which clue fits grapes?", ["🍇 Grow in bunches", "📏 One long curved fruit", "🪨 One big pit"], "🍇 Grow in bunches"),
        ("What is true about grape seeds?", ["Some grapes have seeds", "All seeds are outside", "Grapes have one huge pit"], "Some grapes have seeds"),
        ("Where do grapes grow?", ["On a vine", "On a mango tree", "Under soil"], "On a vine"),
        ("Which fruit is much larger than one grape?", ["Mango", "Grape", "None"], "Mango"),
        ("What may happen with no water?", ["The vine may wilt", "The vine makes juice itself", "Nothing changes"], "The vine may wilt"),
        ("Which FruitScan fruit also grows on a non-tree plant?", ["Strawberry", "Apple", "Mango"], "Strawberry"),
    ],
    "Mango": [
        ("Which shape is a useful mango clue?", ["🥚 Oval", "📏 Very long and curved", "🍇 Tiny bunch"], "🥚 Oval"),
        ("What is inside a mango?", ["One large flat pit", "Seeds only outside", "A core of many tiny seeds"], "One large flat pit"),
        ("Where does a mango grow?", ["On a tree", "On a vine", "Under water"], "On a tree"),
        ("Which fruit is usually much smaller than a mango?", ["Grape", "Mango", "None"], "Grape"),
        ("What may happen with no water?", ["The tree may wilt or struggle", "It makes more fruit", "It becomes sweeter immediately"], "The tree may wilt or struggle"),
        ("Which other FruitScan fruit also grows on a tree?", ["Apple", "Banana", "Strawberry"], "Apple"),
    ],
    "Strawberry": [
        ("Which clue fits a ripe strawberry?", ["🔴 Red", "📏 Long and curved", "🪨 One large pit"], "🔴 Red"),
        ("Where are the seed-containing dots?", ["On the outside", "In one big pit", "Only in a core"], "On the outside"),
        ("What does a strawberry grow on?", ["A small low plant", "A tall woody tree", "A climbing grape vine"], "A small low plant"),
        ("Which fruit is usually longer?", ["Banana", "Strawberry", "They are always the same"], "Banana"),
        ("What may happen with no water?", ["The plant may wilt", "It turns into a tree", "It grows normally forever"], "The plant may wilt"),
        ("Which FruitScan fruit also grows on a non-tree plant?", ["Grape", "Apple", "Mango"], "Grape"),
    ],
}

LEVEL_NAMES = {
    1: "Recognise",
    2: "Observe",
    3: "Understand",
    4: "Compare",
    5: "Reason",
    6: "Apply",
}


def show_adaptive_challenge(fruit_name: str) -> None:
    level = learning_level(fruit_name)
    prompt, options, answer = QUESTIONS[fruit_name][level - 1]
    emoji = FRUIT_EMOJI[fruit_name]

    st.markdown("### 🌈 My Challenge Level")
    st.progress(level / 6)
    st.markdown(f"**Level {level} of 6 — {LEVEL_NAMES[level]}**")
    st.caption("Questions become a little harder as you learn.")

    st.markdown(f"### {emoji} {prompt}")
    choices = list(options)
    rnd = random.Random(f"adaptive:{fruit_name}:{level}")
    rnd.shuffle(choices)

    state_key = f"adaptive_answer_{fruit_name}_{level}"
    columns = st.columns(len(choices))
    for column, choice in zip(columns, choices):
        with column:
            if st.button(
                choice,
                key=f"adaptive_choice_{fruit_name}_{level}_{choice}",
                use_container_width=True,
            ):
                st.session_state[state_key] = choice

    selected = st.session_state.get(state_key)
    if not selected:
        return

    if selected == answer:
        new = record_adaptive_success(fruit_name, level)
        st.success("⭐ Great thinking! You found the best answer.")
        if new and level < 6:
            st.info("🌟 Your next challenge level is now ready!")
            if st.button(
                "➡️ Try my next level",
                key=f"adaptive_next_{fruit_name}_{level}",
                type="primary",
                use_container_width=True,
            ):
                st.rerun()
    else:
        st.info("👀 Look again and try another idea. You can change your answer.")
