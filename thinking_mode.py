from __future__ import annotations

import random

import streamlit as st

from fruit_education import FRUIT_EDUCATION, fruit_names

FRUIT_EMOJI = {
    "Apple": "🍎",
    "Banana": "🍌",
    "Grape": "🍇",
    "Mango": "🥭",
    "Strawberry": "🍓",
}

TRAITS = {
    "Apple": {
        "best_clue": "⭕ Round shape",
        "reasons": ["🔴 It can be red", "⭕ It is round", "🌿 It has a stem"],
        "inside_options": ["🪨 One big pit", "🌱 Several small seeds in a core", "✨ Seed dots on the outside"],
        "inside_answer": "🌱 Several small seeds in a core",
        "mystery": ["I can be red, green, or yellow.", "I am usually round.", "My small brown seeds sit in a core."],
        "teach": ["Its seeds sit inside a core.", "It grows on a tree.", "It can be red, green, or yellow."],
    },
    "Banana": {
        "best_clue": "📏 Long and curved",
        "reasons": ["🟡 It is often yellow", "📏 It is long and curved", "🧥 It has a peel"],
        "inside_options": ["🪨 One big pit", "••• Tiny dark seed traces", "✨ Seed dots on the outside"],
        "inside_answer": "••• Tiny dark seed traces",
        "mystery": ["I am long and curved.", "You peel my skin.", "I grow on a large herb, not a woody tree."],
        "teach": ["It grows on a large herb.", "Ripe bananas are usually yellow.", "It has tiny undeveloped seed traces."],
    },
    "Grape": {
        "best_clue": "🫧 Small and round",
        "reasons": ["🟣 It may be purple", "🫧 It is small and round", "🍇 It grows in a bunch"],
        "inside_options": ["🌰 Some have small seeds inside", "🪨 One big flat pit", "✨ Seeds only on the outside"],
        "inside_answer": "🌰 Some have small seeds inside",
        "mystery": ["I grow in bunches.", "I can be green, red, or purple.", "I grow on a climbing vine."],
        "teach": ["It grows on a vine.", "Grapes grow in bunches.", "Some grapes have small seeds inside."],
    },
    "Mango": {
        "best_clue": "🥚 Oval shape",
        "reasons": ["🟠 It may be yellow or orange", "🥚 It is often oval", "🥭 It has a thick skin"],
        "inside_options": ["🪨 One big flat pit", "🌱 Many tiny seeds in a core", "✨ Seed dots on the outside"],
        "inside_answer": "🪨 One big flat pit",
        "mystery": ["I am a tropical fruit.", "My flesh is often yellow or orange.", "I have one large flat pit."],
        "teach": ["It has one large flat pit.", "It grows on an evergreen tree.", "Its flesh is usually yellow or orange."],
    },
    "Strawberry": {
        "best_clue": "❤️ Small heart-like shape",
        "reasons": ["🔴 It is red when ripe", "❤️ It has a heart-like shape", "✨ It has tiny dots outside"],
        "inside_options": ["✨ Seed-containing dots on the outside", "🪨 One big pit", "🌱 A hard seed core"],
        "inside_answer": "✨ Seed-containing dots on the outside",
        "mystery": ["I am red when ripe.", "I grow on a small low plant.", "My seed-containing dots are on the outside."],
        "teach": ["Its seed-containing dots are outside.", "It grows on a small low plant.", "It turns red as it ripens."],
    },
}

TREE_FRUITS = {"Apple", "Mango"}


def _thinking_scans() -> dict:
    if "thinking_scans" not in st.session_state:
        st.session_state.thinking_scans = {}
    return st.session_state.thinking_scans


def _scan_state(scan_token: str) -> dict:
    scans = _thinking_scans()
    if scan_token not in scans:
        scans[scan_token] = {
            "guess": None,
            "reason": None,
            "revealed": False,
        }
    return scans[scan_token]


def show_guess_before_reveal(
    image,
    source_caption: str,
    fruit_name: str,
    scan_token: str,
) -> bool:
    """Gate a verified AI result behind child prediction and reasoning."""
    state = _scan_state(scan_token)
    if state["revealed"]:
        return True

    st.divider()
    st.markdown("## 🧠 Think Before FruitScan Tells You")
    st.caption("See → Think → Predict → Explain → Discover")
    st.image(image, caption=source_caption, use_container_width=True)

    if state["guess"] is None:
        st.markdown("### 🤔 What fruit do YOU think this is?")
        options = [fruit_name]
        others = [name for name in fruit_names() if name != fruit_name]
        rnd = random.Random(f"guess:{scan_token}:{fruit_name}")
        options.extend(rnd.sample(others, 2))
        rnd.shuffle(options)

        columns = st.columns(3)
        for column, option in zip(columns, options):
            with column:
                if st.button(
                    f"{FRUIT_EMOJI[option]} {option}",
                    key=f"think_guess_{scan_token}_{option}",
                    use_container_width=True,
                ):
                    state["guess"] = option
                    st.rerun()
        st.info("There is no penalty for guessing. Look carefully and choose what you notice.")
        return False

    st.markdown(
        f"### 💭 You predicted: {FRUIT_EMOJI[state['guess']]} **{state['guess']}**"
    )

    if state["reason"] is None:
        st.markdown("### ❓ Why do you think so?")
        reasons = TRAITS[state["guess"]]["reasons"]
        for index, reason in enumerate(reasons):
            if st.button(
                reason,
                key=f"think_reason_{scan_token}_{index}",
                use_container_width=True,
            ):
                state["reason"] = reason
                st.rerun()

        if st.button(
            "↩️ Change my fruit guess",
            key=f"think_change_guess_{scan_token}",
            use_container_width=True,
        ):
            state["guess"] = None
            state["reason"] = None
            st.rerun()
        return False

    st.success(f"🗣️ Your reason: **{state['reason']}**")
    st.write("Great thinking. Now you are ready to discover what FruitScan found.")

    if st.button(
        "🔍 DISCOVER THE FRUIT",
        key=f"think_reveal_{scan_token}",
        type="primary",
        use_container_width=True,
    ):
        state["revealed"] = True
        st.rerun()

    return False


def show_prediction_feedback(fruit_name: str, scan_token: str) -> None:
    state = _scan_state(scan_token)
    guess = state.get("guess")
    if not guess:
        return

    if guess == fruit_name:
        st.success(
            f"⭐ Your prediction matched! You guessed {FRUIT_EMOJI[fruit_name]} {fruit_name}."
        )
    else:
        st.info(
            f"👀 Let's look carefully. You predicted {FRUIT_EMOJI[guess]} {guess}, "
            f"and FruitScan found {FRUIT_EMOJI[fruit_name]} {fruit_name}. "
            "Compare the clues and see what you notice now."
        )


def _single_choice(
    prompt: str,
    options: list[str],
    answer: str,
    key_prefix: str,
    explanation: str,
) -> None:
    st.markdown(f"### {prompt}")
    state_key = f"{key_prefix}_answer"

    columns = st.columns(len(options))
    for column, option in zip(columns, options):
        with column:
            if st.button(
                option,
                key=f"{key_prefix}_{option}",
                use_container_width=True,
            ):
                st.session_state[state_key] = option

    selected = st.session_state.get(state_key)
    if not selected:
        return

    if selected == answer:
        st.success(f"⭐ Nice thinking! {explanation}")
    else:
        st.info(f"👀 Let's look again. Hint: {explanation}")


def show_observation_activity(fruit_name: str, namespace: str) -> None:
    trait = TRAITS[fruit_name]
    distractors = {
        "Apple": ["📏 Long and curved", "🍇 Grows in a bunch"],
        "Banana": ["⭕ Round shape", "✨ Seeds outside"],
        "Grape": ["📏 Long and curved", "🥚 Large oval"],
        "Mango": ["🫧 Tiny and round", "📏 Long and curved"],
        "Strawberry": ["📏 Long and curved", "🫧 Grows in a bunch"],
    }[fruit_name]
    options = [trait["best_clue"], *distractors]
    rnd = random.Random(f"observe:{namespace}:{fruit_name}")
    rnd.shuffle(options)
    _single_choice(
        "👀 What do you notice first?",
        options,
        trait["best_clue"],
        f"observe_{namespace}_{fruit_name}",
        f"A useful clue is: {trait['best_clue']}.",
    )


def show_look_inside_activity(fruit_name: str, namespace: str) -> None:
    trait = TRAITS[fruit_name]
    options = list(trait["inside_options"])
    rnd = random.Random(f"inside:{namespace}:{fruit_name}")
    rnd.shuffle(options)

    _single_choice(
        f"🔮 What do you predict is inside the {fruit_name.lower()}?",
        options,
        trait["inside_answer"],
        f"inside_{namespace}_{fruit_name}",
        FRUIT_EDUCATION[fruit_name]["seed_description"],
    )


def show_growth_order_activity(fruit_name: str, namespace: str) -> None:
    correct = [
        "🌱 Start",
        "🌿 Plant / tree",
        "🌸 Flower",
        f"{FRUIT_EMOJI[fruit_name]} Fruit",
    ]
    key = f"growth_order_{namespace}_{fruit_name}"
    if key not in st.session_state:
        st.session_state[key] = []

    selected = st.session_state[key]
    st.markdown("### 🧩 Put growth in the correct order")
    st.caption("Tap each stage from first to last.")

    shuffled = list(correct)
    rnd = random.Random(f"growth:{namespace}:{fruit_name}")
    rnd.shuffle(shuffled)

    columns = st.columns(4)
    for column, stage in zip(columns, shuffled):
        with column:
            if st.button(
                stage,
                key=f"{key}_{stage}",
                use_container_width=True,
                disabled=stage in selected,
            ):
                selected.append(stage)
                st.rerun()

    if selected:
        st.write("**Your order:** " + " → ".join(selected))

    if len(selected) == len(correct):
        if selected == correct:
            st.success("⭐ Great sequencing! The plant grows step by step before the fruit appears.")
        else:
            st.info("👀 Let's rethink the order. Which stage needs to happen first?")
        if st.button("🔄 Try the growth puzzle again", key=f"{key}_reset"):
            st.session_state[key] = []
            st.rerun()


def show_compare_activity(fruit_name: str, namespace: str) -> None:
    compare = "Banana" if fruit_name in TREE_FRUITS else "Apple"
    answer = fruit_name if fruit_name in TREE_FRUITS else compare
    options = [fruit_name, compare]

    st.markdown(
        f"### ⚖️ Compare: {FRUIT_EMOJI[fruit_name]} {fruit_name} vs "
        f"{FRUIT_EMOJI[compare]} {compare}"
    )
    _single_choice(
        "Which one grows on a tree?",
        [f"{FRUIT_EMOJI[name]} {name}" for name in options],
        f"{FRUIT_EMOJI[answer]} {answer}",
        f"compare_{namespace}_{fruit_name}",
        f"{answer} grows on a tree.",
    )


def show_cause_effect_activity(fruit_name: str, namespace: str) -> None:
    _single_choice(
        "💭 What happens if the plant gets no water?",
        ["🌱 It keeps growing normally", "🥀 It may wilt", "🍫 It makes chocolate"],
        "🥀 It may wilt",
        f"water_{namespace}_{fruit_name}",
        "Plants need water for healthy growth, so without enough water they may wilt.",
    )


def show_teach_me_activity(fruit_name: str, namespace: str) -> None:
    key = f"teach_{namespace}_{fruit_name}"
    st.markdown("### 🗣️ Teach Me!")
    st.write("FruitScan forgot something. Choose one thing you can teach it.")
    choice = st.radio(
        "What did you learn?",
        TRAITS[fruit_name]["teach"],
        index=None,
        key=key,
    )
    if choice:
        st.success(f"⭐ Great teaching! You remembered: **{choice}**")


def show_thinking_lesson(fruit_name: str, namespace: str) -> None:
    st.markdown("## 🧠 Think & Discover")
    st.caption("Observe, predict, compare, sequence, explain, and apply what you learn.")

    tabs = st.tabs(
        [
            "👀 Observe",
            "🔮 Look Inside",
            "🌱 Growth Puzzle",
            "⚖️ Compare",
            "💭 What If?",
            "🗣️ Teach Me",
        ]
    )
    with tabs[0]:
        show_observation_activity(fruit_name, namespace)
    with tabs[1]:
        show_look_inside_activity(fruit_name, namespace)
    with tabs[2]:
        show_growth_order_activity(fruit_name, namespace)
    with tabs[3]:
        show_compare_activity(fruit_name, namespace)
    with tabs[4]:
        show_cause_effect_activity(fruit_name, namespace)
    with tabs[5]:
        show_teach_me_activity(fruit_name, namespace)


def show_mystery_fruit_game(fruit_name: str, namespace: str) -> None:
    trait = TRAITS[fruit_name]
    st.markdown("### 🕵️ Mystery Fruit")
    for clue in trait["mystery"]:
        st.write(f"🔎 {clue}")

    others = [name for name in fruit_names() if name != fruit_name]
    rnd = random.Random(f"mystery:{namespace}:{fruit_name}")
    choices = [fruit_name, *rnd.sample(others, 2)]
    rnd.shuffle(choices)

    _single_choice(
        "Who am I?",
        [f"{FRUIT_EMOJI[name]} {name}" for name in choices],
        f"{FRUIT_EMOJI[fruit_name]} {fruit_name}",
        f"mystery_{namespace}_{fruit_name}",
        f"The clues describe {fruit_name}.",
    )


def show_sort_game(namespace: str) -> None:
    st.markdown("### 🧺 Sort the Fruits")
    st.write("Which fruits grow on trees?")
    selected = st.multiselect(
        "Choose all that belong in the tree-fruit basket:",
        fruit_names(),
        format_func=lambda name: f"{FRUIT_EMOJI[name]} {name}",
        key=f"sort_tree_{namespace}",
    )
    if st.button("✅ Check my basket", key=f"sort_check_{namespace}"):
        if set(selected) == TREE_FRUITS:
            st.success("⭐ Great sorting! Apple and Mango grow on trees.")
        else:
            st.info("👀 Let's look again. Hint: there are two tree fruits in this set.")


def show_odd_one_out_game(namespace: str) -> None:
    st.markdown("### 🚫 Which One Doesn't Belong?")
    st.write("Apple and Mango grow on trees. Which one below does **not** grow on a tree?")
    _single_choice(
        "Choose the odd one out.",
        ["🍎 Apple", "🥭 Mango", "🍌 Banana"],
        "🍌 Banana",
        f"odd_{namespace}",
        "Banana grows on a large herb, while Apple and Mango grow on trees.",
    )


def show_thinking_games(fruit_name: str, namespace: str) -> None:
    tabs = st.tabs(["🕵️ Mystery Fruit", "🧺 Sort It", "🚫 Odd One Out"])
    with tabs[0]:
        show_mystery_fruit_game(fruit_name, namespace)
    with tabs[1]:
        show_sort_game(namespace)
    with tabs[2]:
        show_odd_one_out_game(namespace)
