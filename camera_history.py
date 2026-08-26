from __future__ import annotations

import hashlib
import io

import streamlit as st
from PIL import Image

MAX_CAMERA_HISTORY = 12


def _ensure_state() -> None:
    if "camera_history" not in st.session_state:
        st.session_state.camera_history = []
    if "camera_history_seen" not in st.session_state:
        st.session_state.camera_history_seen = set()
    if "camera_history_selected_id" not in st.session_state:
        st.session_state.camera_history_selected_id = None
    if "camera_history_counter" not in st.session_state:
        st.session_state.camera_history_counter = 0


def remember_camera_photo(photo_bytes: bytes) -> bool:
    """Remember a camera image once and select it when it is newly captured."""
    _ensure_state()
    digest = hashlib.sha256(photo_bytes).hexdigest()

    if digest in st.session_state.camera_history_seen:
        return False

    st.session_state.camera_history_seen.add(digest)
    st.session_state.camera_history_counter += 1
    item = {
        "id": digest,
        "name": f"Camera photo #{st.session_state.camera_history_counter}",
        "bytes": photo_bytes,
    }
    st.session_state.camera_history.append(item)
    st.session_state.camera_history = st.session_state.camera_history[-MAX_CAMERA_HISTORY:]
    st.session_state.camera_history_selected_id = digest
    return True


def _selected_item() -> dict | None:
    _ensure_state()
    selected_id = st.session_state.camera_history_selected_id
    for item in st.session_state.camera_history:
        if item["id"] == selected_id:
            return item
    return None


def selected_camera_image() -> tuple[Image.Image | None, str]:
    item = _selected_item()
    if item is None:
        return None, "Camera fruit"

    image = Image.open(io.BytesIO(item["bytes"])).convert("RGB")
    return image, f"Reviewing {item['name']}"


def show_camera_history() -> None:
    """Render camera thumbnails and controls for the current Streamlit session."""
    _ensure_state()

    st.markdown("### 📸 Camera History")
    st.caption(
        "Review photos taken during this Streamlit session. The newest 12 photos are kept temporarily and are not saved to GitHub."
    )

    history = st.session_state.camera_history
    if not history:
        st.info("Your camera history is empty. Take a fruit picture and it will appear here.")
        return

    action_left, action_right = st.columns([1, 1])
    with action_left:
        if st.button("⏮️ Review latest photo", use_container_width=True, key="camera_history_latest"):
            st.session_state.camera_history_selected_id = history[-1]["id"]
            st.rerun()
    with action_right:
        if st.button("🗑️ Clear camera history", use_container_width=True, key="camera_history_clear"):
            # Keep the seen hashes so the still-active camera_input value is not
            # immediately re-added on the rerun after clearing.
            st.session_state.camera_history = []
            st.session_state.camera_history_selected_id = None
            st.rerun()

    newest_first = list(reversed(history))
    for start in range(0, len(newest_first), 4):
        row = newest_first[start : start + 4]
        columns = st.columns(4)
        for column, item in zip(columns, row):
            with column:
                thumb = Image.open(io.BytesIO(item["bytes"])).convert("RGB")
                selected = item["id"] == st.session_state.camera_history_selected_id
                caption = f"{'✅ ' if selected else ''}{item['name']}"
                st.image(thumb, caption=caption, use_container_width=True)
                if st.button(
                    "👀 Review",
                    key=f"camera_history_review_{item['id'][:16]}",
                    use_container_width=True,
                    disabled=selected,
                ):
                    st.session_state.camera_history_selected_id = item["id"]
                    st.rerun()
