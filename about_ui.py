from __future__ import annotations

import streamlit as st


def show_about() -> None:
    st.header("ℹ️ About FruitScan Kids")
    st.write(
        "FruitScan Kids is designed around short, visual activities so young learners "
        "can scan, think, discover, cook, and play without seeing everything at once."
    )
    st.markdown(
        """
### Child-friendly learning journey

**1. 📷 Scan & Think** — Take or choose a clear fruit picture.  
**2. 🤔 Predict** — Guess before FruitScan reveals an accepted result.  
**3. 🌟 Activities** — Choose one large activity card at a time.  
**4. 🧠 Think & Discover** — Observe, predict, compare, sequence, and reason.  
**5. 👩‍🍳 Fruit Kitchen** — Explore food, drink, and dessert ideas with an adult.  
**6. 🎮 Quiz & Games** — Use picture and thinking games plus adaptive challenge levels.  
**7. 🎓 Learn Fruits** — Browse the collection and fruit lessons.  
**8. ⚙️ Adult** — View the parent / teacher learning summary and project information.
        """
    )
    st.warning(
        "👨‍👩‍👧 Food activities require adult supervision. Adults should handle knives, "
        "blenders, heat, allergies, and age-appropriate choking safety."
    )
    st.info(
        "Camera History and the Parent / Teacher Dashboard are session-based. "
        "Camera photos are not saved to GitHub."
    )
    st.caption(
        "The trained model is loaded automatically by the deployed app. Model training and "
        "maintenance tools stay in the GitHub project for developers and are not shown to children."
    )
