from __future__ import annotations

import random

import streamlit as st

from child_experience import FRUIT_EMOJI, FRUIT_PHOTOS, award_star
from fruit_education import fruit_names
from progress_store import record_quiz_attempt


def show_picture_quiz(fruit_name: str, namespace: str) -> None:
    fruits = fruit_names()
    distractors = [name for name in fruits if name != fruit_name]
    rnd = random.Random(f"{namespace}:{fruit_name}")
    choices = [fruit_name] + rnd.sample(distractors, 2)
    rnd.shuffle(choices)

    st.markdown(f"### 🖼️ Which one is the {fruit_name.lower()}?")
    st.write("Look at the three photos and tap the picture you think is correct.")

    cols = st.columns(3)
    for position, (col, choice) in enumerate(zip(cols, choices), start=1):
        with col:
            with st.container(border=True):
                st.image(FRUIT_PHOTOS[choice], use_container_width=True)
                st.caption(f"Picture {position}")
                if st.button(
                    "👆 Choose this picture",
                    key=f"tracked_pic_quiz_{namespace}_{choice}",
                    use_container_width=True,
                ):
                    correct = choice == fruit_name
                    record_quiz_attempt(
                        fruit_name,
                        correct,
                        event_key=f"{namespace}:{fruit_name}:{choice}:{st.session_state.get('reward_stars', 0)}",
                    )
                    if correct:
                        award_star(f"quiz:{namespace}:{fruit_name}", 1)
                        st.success(
                            f"⭐ Great looking! That is the {FRUIT_EMOJI[fruit_name]} {fruit_name}."
                        )
                    else:
                        st.info(
                            f"👀 Good try. Look again for the {FRUIT_EMOJI[fruit_name]} {fruit_name}."
                        )
