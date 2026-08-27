from __future__ import annotations

import json

import streamlit as st
import streamlit.components.v1 as components


CAMERA_TIPS = (
    ("🍎", "One fruit", "Show one fruit clearly."),
    ("🎯", "Middle", "Put the fruit near the centre."),
    ("☀️", "Bright", "Use enough light."),
    ("🔍", "Close", "Move close enough to see the fruit."),
)


def show_camera_guidance() -> None:
    st.markdown("### 📷 Ready for a good fruit photo?")
    columns = st.columns(4)
    for column, (emoji, title, text) in zip(columns, CAMERA_TIPS):
        with column:
            with st.container(border=True):
                st.markdown(
                    f"<div style='text-align:center;font-size:44px'>{emoji}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**{title}**")
                st.caption(text)

    components.html(
        """
        <div style="
          height:150px;border:4px dashed #9aa0a6;border-radius:24px;
          display:flex;align-items:center;justify-content:center;
          font-family:Arial,sans-serif;text-align:center;font-weight:800;
          font-size:22px;">
          🍓<br>PUT ONE FRUIT HERE
        </div>
        """,
        height=170,
    )

    instruction = json.dumps(
        "Show one fruit. Put it in the middle. Use enough light. Move close enough to see it clearly."
    )
    components.html(
        f"""
        <button id="readTips" style="
          width:100%;min-height:56px;border-radius:16px;border:1px solid #bbb;
          font-size:18px;font-weight:800;cursor:pointer;background:white;">
          🔊 Read the camera tips
        </button>
        <script>
        const btn = document.getElementById('readTips');
        btn.addEventListener('click', () => {{
          if (!('speechSynthesis' in window)) {{
            btn.textContent = '🔇 Audio not supported';
            return;
          }}
          window.speechSynthesis.cancel();
          const voice = new SpeechSynthesisUtterance({instruction});
          voice.rate = 0.72;
          voice.pitch = 1.02;
          window.speechSynthesis.speak(voice);
        }});
        </script>
        """,
        height=70,
    )
