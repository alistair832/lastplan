from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from camera_history import (
    remember_camera_photo,
    selected_camera_id,
    selected_camera_image,
    show_camera_history,
    update_camera_result,
)
from child_experience import (
    FRUIT_EMOJI,
    child_styles,
    corrected_label,
    repeat_pronunciation_button,
    save_correction,
    show_collection_book,
    show_picture_quiz,
    show_picture_story,
    show_progress_banner,
    show_quick_recipe_mode,
    unlock_fruit,
)
from dataset_utils import save_summary, scan_dataset
from education_ui import show_learning_browser
from fruit_education import FRUIT_EDUCATION, fruit_names
from prepare_dataset import extract_dataset
from thinking_mode import (
    show_guess_before_reveal,
    show_prediction_feedback,
    show_thinking_games,
    show_thinking_lesson,
)
from verifier import FruitVerifier

CHECKPOINT = Path("artifacts/fruit_classifier.pt")
SUMMARY = Path("artifacts/dataset_summary.json")
DATA_DIR = Path("data")
UPLOAD_DIR = Path("artifacts/uploads")

st.set_page_config(page_title="FruitScan Kids", page_icon="🍓", layout="wide")
child_styles()
st.title("🍓 FruitScan Kids")
st.caption("Scan, think, discover, cook, and play — one simple activity at a time!")


def show_dataset_summary(summary: dict) -> None:
    rows = []
    for split, counts in summary.get("splits", {}).items():
        for fruit, count in counts.items():
            rows.append({"Split": split, "Fruit": fruit, "Images": count})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.write(f"**Total images:** {summary.get('total_images', 0):,}")
    corrupt = summary.get("corrupt_images", [])
    if corrupt:
        st.error(f"Corrupt/unreadable images found: {len(corrupt)}")
    else:
        st.success("Dataset image scan completed with no corrupt images found.")


@st.cache_resource
def load_verifier(checkpoint_mtime: float):
    del checkpoint_mtime
    return FruitVerifier(CHECKPOINT)


def get_verifier():
    if not CHECKPOINT.exists():
        return None, None
    try:
        return load_verifier(CHECKPOINT.stat().st_mtime), None
    except Exception as exc:
        return None, str(exc)


def current_learning_fruit() -> str:
    fruits = fruit_names()
    current = st.session_state.get("current_learning_fruit")
    return current if current in fruits else fruits[0]


def activity_fruit_picker() -> str:
    fruits = fruit_names()
    current = current_learning_fruit()
    selected = st.selectbox(
        "🍓 Choose the fruit for this activity",
        fruits,
        index=fruits.index(current),
        format_func=lambda name: f"{FRUIT_EMOJI[name]} {name}",
        key="activity_fruit_picker",
    )
    st.session_state.current_learning_fruit = selected
    return selected


def sync_activity_fruit(fruit_name: str) -> None:
    if fruit_name not in fruit_names():
        return
    st.session_state.current_learning_fruit = fruit_name
    st.session_state.activity_fruit_picker = fruit_name


def _ensure_scan_widget_versions() -> None:
    if "camera_widget_version" not in st.session_state:
        st.session_state.camera_widget_version = 0
    if "upload_widget_version" not in st.session_state:
        st.session_state.upload_widget_version = 0


def _start_fresh_camera() -> None:
    _ensure_scan_widget_versions()
    st.session_state.scan_input_mode = "camera"
    st.session_state.camera_widget_version += 1
    st.session_state.camera_history_selected_id = None


def _start_fresh_upload() -> None:
    _ensure_scan_widget_versions()
    st.session_state.scan_input_mode = "upload"
    st.session_state.upload_widget_version += 1


def show_correction_controls(
    scan_token: str,
    current_label: str,
    history_id: str | None,
) -> None:
    """Correction is only available after an image passes the Unknown Gateway."""
    state_key = f"correction_open_{scan_token}"

    if st.button(
        "❌ That's not my fruit",
        key=f"wrong_recognition_{scan_token}",
        use_container_width=True,
    ):
        st.session_state[state_key] = True

    if not st.session_state.get(state_key, False):
        return

    st.markdown("#### Help FruitScan use the right known fruit")
    options = fruit_names()
    default_index = options.index(current_label) if current_label in options else 0
    correct = st.selectbox(
        "Which trained fruit is really in the picture?",
        options,
        index=default_index,
        format_func=lambda name: f"{FRUIT_EMOJI[name]} {name}",
        key=f"correct_fruit_{scan_token}",
    )

    left, right = st.columns(2)
    with left:
        if st.button(
            "✅ Use this fruit",
            key=f"save_correction_{scan_token}",
            type="primary",
            use_container_width=True,
        ):
            save_correction(scan_token, correct, history_id)
            sync_activity_fruit(correct)
            st.session_state[state_key] = False
            st.rerun()

    with right:
        if st.button(
            "📷 Retake picture",
            key=f"retake_{scan_token}",
            use_container_width=True,
        ):
            _start_fresh_camera()
            st.session_state[state_key] = False
            st.rerun()


def show_unknown_gateway(
    image: Image.Image,
    source_caption: str,
    result,
    history_id: str | None,
    scan_token: str,
) -> None:
    """Child-facing rejection screen for out-of-dataset or uncertain images."""
    if history_id is not None:
        update_camera_result(history_id, result)

    st.divider()
    image_col, message_col = st.columns([1, 1.1])

    with image_col:
        st.image(image, caption=source_caption, use_container_width=True)

    with message_col:
        st.error("❓ Unknown / Not Recognised")
        st.markdown("### This picture did not safely match a trained fruit.")
        st.write(
            "FruitScan will not guess when the picture does not pass the recognition gateway."
        )
        st.caption(
            "FruitScan currently recognises only: 🍎 Apple, 🍌 Banana, 🍇 Grape, "
            "🥭 Mango, and 🍓 Strawberry."
        )
        st.info(
            "Try one clear fruit in the centre of the picture with enough light and not too much background."
        )

        retake_col, upload_col = st.columns(2)
        with retake_col:
            if st.button(
                "📷 Retake Photo",
                key=f"unknown_retake_{scan_token}",
                type="primary",
                use_container_width=True,
            ):
                _start_fresh_camera()
                st.rerun()

        with upload_col:
            if st.button(
                "🖼️ Choose Another Picture",
                key=f"unknown_upload_{scan_token}",
                use_container_width=True,
            ):
                _start_fresh_upload()
                st.rerun()

    st.caption(
        "The Unknown Gateway is intentionally conservative: a valid fruit may sometimes be "
        "rejected and asked to retry instead of receiving a wrong fruit name."
    )


def show_child_result(
    image: Image.Image,
    verifier: FruitVerifier,
    source_caption: str,
    scan_token: str,
    history_id: str | None = None,
) -> None:
    with st.spinner("🔎 FruitScan is checking the picture..."):
        result = verifier.predict(image)

    # Gateway first: an uncertain or out-of-profile image never reaches guessing,
    # correction, rewards, lessons, or a forced five-class fruit label.
    if not result.verified:
        show_unknown_gateway(
            image,
            source_caption,
            result,
            history_id,
            scan_token,
        )
        return

    model_label = result.label
    effective_label = corrected_label(scan_token) or model_label

    ready_to_reveal = show_guess_before_reveal(
        image,
        source_caption,
        effective_label,
        scan_token,
    )
    if not ready_to_reveal:
        return

    if history_id is not None:
        update_camera_result(history_id, result)

    st.divider()
    sync_activity_fruit(effective_label)
    emoji = FRUIT_EMOJI[effective_label]
    newly_unlocked = unlock_fruit(effective_label, scan_token)

    result_left, result_right = st.columns([1, 1.1])
    with result_left:
        st.image(image, caption=source_caption, use_container_width=True)
    with result_right:
        st.success(f"🎉 FruitScan found: {emoji} {effective_label}")
        st.markdown(f"# {emoji} {effective_label}")
        show_prediction_feedback(effective_label, scan_token)

        if newly_unlocked:
            st.info("⭐ New fruit unlocked in your Collection Book!")

        st.caption("Tap the button if you want to hear and practise the fruit name.")
        repeat_pronunciation_button(effective_label)
        show_correction_controls(scan_token, effective_label, history_id)

    st.markdown("### 🌟 Choose what to do next")
    st.info(
        f"Your {emoji} **{effective_label}** is ready in **🌟 Activities**. "
        "Open that section above, then choose **🧠 Think & Discover**, "
        "**👩‍🍳 Fruit Kitchen**, or **🎮 Quiz & Games**."
    )


def show_think_activity(fruit_name: str) -> None:
    st.header(f"🧠 Think & Discover — {FRUIT_EMOJI[fruit_name]} {fruit_name}")
    st.caption(
        "Observe, predict, compare, sequence, and explain. Work through only the activities you want."
    )

    show_picture_story(fruit_name)
    show_thinking_lesson(fruit_name, namespace=f"activity-{fruit_name.lower()}")

    info = FRUIT_EDUCATION[fruit_name]
    st.markdown("### 🌿 What does the plant need?")
    needs_left, needs_right = st.columns(2)
    with needs_left:
        st.info(f"☀️ **Sun:** {info['sun']}")
        st.info(f"💧 **Water:** {info['water']}")
    with needs_right:
        st.info(f"🪴 **Soil:** {info['soil']}")
        st.info(f"🌤️ **Weather:** {info['weather']}")

    st.success(
        "🌿 **Photosynthesis:** ☀️ Sunlight + 💧 Water + 🌬️ Air → 🍬 Plant food + 🌱 Growth"
    )


def show_kitchen_activity(fruit_name: str) -> None:
    st.header(f"👩‍🍳 Fruit Kitchen — {FRUIT_EMOJI[fruit_name]} {fruit_name}")
    st.caption(
        "Choose a food, drink, or dessert and explore the simple steps with an adult."
    )
    show_quick_recipe_mode(fruit_name)


def show_games_activity(fruit_name: str) -> None:
    st.header(f"🎮 Quiz & Games — {FRUIT_EMOJI[fruit_name]} {fruit_name}")
    st.caption("Choose either a picture quiz or a thinking game.")

    picture_tab, thinking_tab = st.tabs(
        ["🖼️ Picture Quiz", "🧠 Thinking Games"]
    )
    with picture_tab:
        show_picture_quiz(fruit_name, namespace=f"activity-{fruit_name.lower()}")
    with thinking_tab:
        show_thinking_games(fruit_name, namespace=f"activity-{fruit_name.lower()}")


scan_tab, activities_tab, learn_tab, adult_tab = st.tabs(
    ["📷 Scan & Think", "🌟 Activities", "🎓 Learn Fruits", "⚙️ Adult"]
)

with scan_tab:
    verifier, verifier_error = get_verifier()

    if verifier is None:
        st.warning("The fruit recognition model is not ready yet.")
        if verifier_error:
            st.error(f"Checkpoint loading error: {verifier_error}")
        st.write("Open **Adult → Model Setup** to prepare the trained FruitScan model.")
    else:
        with st.sidebar:
            st.header("🍓 FruitScan Kids")
            st.success("Fruit recognition ready")
            st.success("Unknown Gateway ready")
            st.info("Learning mode: Think & Discover")
            st.caption(
                "Unknown or uncertain pictures are rejected instead of forced into a fruit class."
            )

        show_progress_banner()
        show_collection_book()
        st.divider()

        st.header("📷 Scan a Fruit")
        st.write("Choose how you want to show FruitScan a fruit.")

        if "scan_input_mode" not in st.session_state:
            st.session_state.scan_input_mode = None
        _ensure_scan_widget_versions()

        take_col, choose_col = st.columns(2)
        with take_col:
            if st.button(
                "📷 TAKE A PHOTO",
                key="big_take_photo",
                type="primary",
                use_container_width=True,
            ):
                _start_fresh_camera()
                st.rerun()
        with choose_col:
            if st.button(
                "🖼️ CHOOSE A PICTURE",
                key="big_choose_picture",
                use_container_width=True,
            ):
                _start_fresh_upload()
                st.rerun()

        mode = st.session_state.scan_input_mode
        image = None
        source_caption = "Fruit picture"
        history_id = None
        scan_token = None

        if mode == "camera":
            st.markdown("### 🎥 Camera")
            st.caption("Hold one fruit clearly in the picture, then take the photo.")

            camera_photo = st.camera_input(
                "Take your fruit photo",
                key=f"front_camera_{st.session_state.camera_widget_version}",
                help="Your browser or phone controls which physical camera is used.",
            )

            if camera_photo is not None:
                try:
                    remember_camera_photo(camera_photo.getvalue())
                except Exception:
                    st.error("I could not add this picture to Camera History.")

            try:
                image, source_caption = selected_camera_image()
                history_id = selected_camera_id()
                scan_token = history_id
            except Exception:
                image = None
                history_id = None
                scan_token = None
                st.error("I could not reopen this camera picture.")

            if image is None:
                st.info("Take a photo above. Your pictures will appear in Camera History.")

        elif mode == "upload":
            st.markdown("### 🖼️ Choose a Picture")
            uploaded = st.file_uploader(
                "Choose a JPG, PNG, JPEG, or WebP fruit picture",
                type=["jpg", "jpeg", "png", "webp"],
                key=f"fruit_image_{st.session_state.upload_widget_version}",
            )
            if uploaded is not None:
                try:
                    raw = uploaded.getvalue()
                    image = Image.open(uploaded).convert("RGB")
                    source_caption = "Chosen fruit picture"
                    scan_token = f"upload-{hashlib.sha256(raw).hexdigest()}"
                except Exception:
                    st.error("I could not read this picture.")

        if mode is not None:
            if st.button("↩️ Choose a different picture method", key="change_scan_mode"):
                st.session_state.scan_input_mode = None
                st.session_state.camera_widget_version += 1
                st.session_state.upload_widget_version += 1
                st.session_state.camera_history_selected_id = None
                st.rerun()

        if image is not None and scan_token:
            show_child_result(
                image,
                verifier,
                source_caption,
                scan_token=scan_token,
                history_id=history_id,
            )

        if mode == "camera":
            st.divider()
            show_camera_history()

with activities_tab:
    st.header("🌟 Activities")
    st.write(
        "Choose one fruit, then open just the activity you want. "
        "The latest recognised fruit is selected automatically."
    )
    activity_fruit = activity_fruit_picker()
    st.divider()

    think_tab, kitchen_tab, games_tab = st.tabs(
        ["🧠 Think & Discover", "👩‍🍳 Fruit Kitchen", "🎮 Quiz & Games"]
    )
    with think_tab:
        show_think_activity(activity_fruit)
    with kitchen_tab:
        show_kitchen_activity(activity_fruit)
    with games_tab:
        show_games_activity(activity_fruit)

with learn_tab:
    show_progress_banner()
    show_collection_book()
    st.divider()
    show_learning_browser()

with adult_tab:
    setup_tab, about_tab = st.tabs(["🛠️ Model Setup", "ℹ️ About"])

    with setup_tab:
        st.header("🛠️ Model Setup")
        st.write(
            "This section is for teachers, parents, or project setup. "
            "Children can use Scan & Think, Activities, and Learn Fruits."
        )

        if CHECKPOINT.exists():
            st.success("A trained model checkpoint is available.")
            st.download_button(
                "⬇️ Download trained model",
                data=CHECKPOINT.read_bytes(),
                file_name="fruit_classifier.pt",
                mime="application/octet-stream",
                use_container_width=True,
            )
        else:
            st.warning("No trained model has been installed yet.")

        st.subheader("Option A — Install an existing trained model")
        model_upload = st.file_uploader(
            "Upload fruit_classifier.pt",
            type=["pt"],
            key="checkpoint_upload",
            help="Use a checkpoint produced by this project only.",
        )

        if model_upload is not None:
            if model_upload.size > 100 * 1024 * 1024:
                st.error("The checkpoint is larger than 100 MB and was not accepted.")
            elif st.button("Install uploaded model", use_container_width=True):
                CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
                CHECKPOINT.write_bytes(model_upload.getbuffer())
                st.cache_resource.clear()
                try:
                    FruitVerifier(CHECKPOINT)
                except Exception as exc:
                    CHECKPOINT.unlink(missing_ok=True)
                    st.error(f"This checkpoint is not compatible with FruitScan Kids: {exc}")
                else:
                    st.success("Model installed successfully.")
                    st.rerun()

        st.divider()
        st.subheader("Option B — Build the model from the Kaggle ZIP")
        st.caption(
            "Full training is best done on a computer or cloud runner with enough CPU/RAM."
        )

        zip_upload = st.file_uploader(
            "Upload the Fruits Classification ZIP",
            type=["zip"],
            key="dataset_zip",
        )

        if zip_upload is not None:
            if st.button("1️⃣ Extract and scan dataset", use_container_width=True):
                UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                zip_path = UPLOAD_DIR / "fruits-classification.zip"
                zip_path.write_bytes(zip_upload.getbuffer())
                try:
                    with st.spinner("Extracting and checking the dataset images..."):
                        root = extract_dataset(zip_path, DATA_DIR)
                        summary = scan_dataset(root, verify_images=True)
                        save_summary(summary, SUMMARY)
                    st.success("Dataset prepared successfully.")
                    show_dataset_summary(summary)
                except Exception as exc:
                    st.exception(exc)

        if SUMMARY.exists():
            try:
                saved_summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
                with st.expander("Current dataset scan", expanded=False):
                    show_dataset_summary(saved_summary)
            except Exception:
                pass

        dataset_ready = all(
            (DATA_DIR / split).exists() for split in ("train", "valid", "test")
        )
        if dataset_ready:
            st.success("Training dataset is ready.")
            epochs = st.slider(
                "Training epochs",
                min_value=4,
                max_value=16,
                value=8,
                step=1,
            )

            if st.button(
                "2️⃣ Train classifier + verifier",
                type="primary",
                use_container_width=True,
            ):
                CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
                cmd = [
                    sys.executable,
                    "-u",
                    "train.py",
                    "--data",
                    str(DATA_DIR),
                    "--epochs",
                    str(epochs),
                    "--workers",
                    "0",
                ]

                log_box = st.empty()
                lines: list[str] = []

                with st.status("Training FruitScan Kids...", expanded=True) as status:
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )

                    assert process.stdout is not None

                    for line in process.stdout:
                        lines.append(line.rstrip())
                        log_box.code("\n".join(lines[-24:]))

                    return_code = process.wait()

                    if return_code == 0 and CHECKPOINT.exists():
                        status.update(label="Training completed", state="complete")
                        st.cache_resource.clear()
                        st.success("The trained model is ready. Open Scan & Think.")
                        st.rerun()
                    else:
                        status.update(label="Training failed", state="error")
                        st.error(
                            "Training did not finish successfully. Check the log above."
                        )

    with about_tab:
        st.header("ℹ️ About FruitScan Kids")
        st.write(
            "FruitScan Kids keeps scanning, thinking, recipes, and games in separate spaces "
            "so children can focus on one activity at a time."
        )
        st.markdown(
            """
### Simple learning journey

**1. 📷 Scan & Think** — Take or provide a picture. The Unknown Gateway checks whether it safely matches one of the trained fruits before any fruit name is shown.  
**2. 🤔 Predict & Discover** — For accepted pictures, predict the fruit, explain a clue, then reveal FruitScan's answer.  
**3. 🌟 Activities** — The recognised fruit is carried into the activity area automatically.  
**4. 🧠 Think & Discover** — Observe, predict the inside, compare fruits, sequence growth, reason, and teach back.  
**5. 👩‍🍳 Fruit Kitchen** — Explore food, drink, and dessert ideas in one separate place.  
**6. 🎮 Quiz & Games** — Play Picture Quiz and Thinking Games without crowding the scan page.  
**7. 🎓 Learn Fruits** — Browse fruit lessons and the Collection Book.  
**8. ⚙️ Adult** — Keep model setup and project information away from the child learning flow.
            """
        )
        st.info(
            "The recognition model is trained for five classes only: Apple, Banana, Grape, "
            "Mango, and Strawberry. The Unknown Gateway reduces forced guesses for other "
            "fruits or unrelated images, but no closed-set image model can guarantee perfect "
            "unknown detection."
        )
        st.warning(
            "👨‍👩‍👧 Food activities require adult supervision. Adults should handle knives, "
            "blenders, heat, allergies, and age-appropriate choking safety."
        )
        st.info(
            "Camera History is temporary and session-only. It keeps the newest 12 camera "
            "pictures while the current Streamlit session is active and does not save "
            "children's photos to GitHub."
        )
