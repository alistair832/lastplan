from __future__ import annotations

import hashlib
import io

import streamlit as st
from PIL import Image

MAX_CAMERA_HISTORY = 12

FRUIT_EMOJI = {
    "Apple": "🍎",
    "Banana": "🍌",
    "Grape": "🍇",
    "Mango": "🥭",
    "Strawberry": "🍓",
}


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
        "number": st.session_state.camera_history_counter,
        "bytes": photo_bytes,
        "result": None,
    }
    st.session_state.camera_history.append(item)
    st.session_state.camera_history = st.session_state.camera_history[-MAX_CAMERA_HISTORY:]
    st.session_state.camera_history_selected_id = digest
    return True


def selected_camera_id() -> str | None:
    _ensure_state()
    return st.session_state.camera_history_selected_id


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


def update_camera_result(photo_id: str | None, result) -> None:
    """Attach the latest FruitScan result to one history item."""
    if not photo_id:
        return

    _ensure_state()
    for item in st.session_state.camera_history:
        if item["id"] != photo_id:
            continue

        existing = item.get("result") or {}
        if existing.get("corrected"):
            return

        item["result"] = {
            "verified": bool(result.verified),
            "label": result.label if result.verified else "Unknown / Not Verified",
            "confidence": float(result.confidence),
            "dataset_similarity": float(result.dataset_similarity),
            "verification_score": float(result.verification_score),
        }
        return


def _delete_history_item(photo_id: str) -> None:
    _ensure_state()
    history = st.session_state.camera_history
    st.session_state.camera_history = [item for item in history if item["id"] != photo_id]

    if st.session_state.camera_history_selected_id == photo_id:
        if st.session_state.camera_history:
            st.session_state.camera_history_selected_id = st.session_state.camera_history[-1]["id"]
        else:
            st.session_state.camera_history_selected_id = None


def _result_caption(item: dict) -> tuple[str, str]:
    result = item.get("result")
    if not result:
        return "🔍 Ready to scan", "FruitScan result will appear here."

    if result["verified"]:
        label = result["label"]
        emoji = FRUIT_EMOJI.get(label, "🍓")
        title = f"{emoji} {label}"
        if result.get("corrected"):
            return title, "✅ Corrected by the learner"
        detail = f"Confidence: {result['confidence'] * 100:.1f}%"
        return title, detail

    return "❓ Unknown", "Did not pass the recognition gateway — try another picture."


def show_camera_history() -> None:
    """Render camera thumbnails, FruitScan results, and review controls."""
    _ensure_state()

    st.markdown("### 📸 My Fruit Camera History")
    st.caption(
        "Review photos taken during this Streamlit session. The newest 12 photos are kept temporarily and are not saved to GitHub."
    )

    history = st.session_state.camera_history
    if not history:
        st.info("Your camera history is empty. Take a fruit picture and it will appear here.")
        return

    top_left, top_middle, top_right = st.columns([1, 1, 1])
    with top_left:
        st.metric("Photos this session", len(history))
    with top_middle:
        verified_count = sum(
            1 for item in history if item.get("result") and item["result"].get("verified")
        )
        st.metric("Fruits recognised", verified_count)
    with top_right:
        learned = {
            item["result"]["label"]
            for item in history
            if item.get("result") and item["result"].get("verified")
        }
        st.metric("Different fruits", len(learned))

    action_left, action_right = st.columns([1, 1])
    with action_left:
        if st.button(
            "⏮️ Review latest photo",
            use_container_width=True,
            key="camera_history_latest",
        ):
            st.session_state.camera_history_selected_id = history[-1]["id"]
            st.rerun()
    with action_right:
        if st.button(
            "🗑️ Clear all history",
            use_container_width=True,
            key="camera_history_clear",
        ):
            st.session_state.camera_history = []
            st.session_state.camera_history_selected_id = None
            st.rerun()

    newest_first = list(reversed(history))
    for start in range(0, len(newest_first), 3):
        row = newest_first[start : start + 3]
        columns = st.columns(3)
        for column, item in zip(columns, row):
            with column:
                thumb = Image.open(io.BytesIO(item["bytes"])).convert("RGB")
                selected = item["id"] == st.session_state.camera_history_selected_id
                caption = f"{'✅ ' if selected else ''}{item['name']}"
                st.image(thumb, caption=caption, use_container_width=True)

                title, detail = _result_caption(item)
                st.markdown(f"**{title}**")
                st.caption(detail)

                review_col, delete_col = st.columns(2)
                with review_col:
                    if st.button(
                        "👀 Review",
                        key=f"camera_history_review_{item['id'][:16]}",
                        use_container_width=True,
                        disabled=selected,
                    ):
                        st.session_state.camera_history_selected_id = item["id"]
                        st.rerun()
                with delete_col:
                    if st.button(
                        "🗑️ Delete",
                        key=f"camera_history_delete_{item['id'][:16]}",
                        use_container_width=True,
                    ):
                        _delete_history_item(item["id"])
                        st.rerun()
