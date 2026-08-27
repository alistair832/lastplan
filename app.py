from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st
from PIL import Image

from activity_cards import activity_header, activity_home, show_plant_needs_cards
from adaptive_learning import show_adaptive_challenge
from adult_dashboard import show_adult_dashboard
from camera_guidance import show_camera_guidance
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
    show_picture_story,
    show_progress_banner,
    show_quick_recipe_mode,
    unlock_fruit,
)
from child_quiz import show_picture_quiz
from education_ui import show_learning_browser
from fruit_education import FRUIT_EDUCATION, fruit_names
from model_setup_ui import show_about, show_model_setup
from progress_store import ensure_progress_state, record_scan
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
ensure_progress_state()

st.markdown(
    """
    <style>
    .block-container { max-width: 1180px; padding-top: 1.5rem; }
    .stButton > button {
        min-height: 64px;
        border-radius: 18px;
        font-size: 1.08rem;
        font-weight: 800;
    }
    div[data-baseweb="select"] > div { min-height: 56px; border-radius: 16px; }
    div[data-testid="stFileUploader"] { border-radius: 18px; }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.22);
        border-radius: 18px;
        padding: 12px;
    }
    @media (max-width: 760px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        h1 { font-size: 2rem !important; }
        h2 { font-size: 1.55rem !important; }
        h3 { font-size: 1.25rem !important; }
        .stButton > button { min-height: 68px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🍓 FruitScan Kids")
st.caption("Scan • Think • Discover • Create • Play")


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
        "🍓 Choose a fruit",
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
    record_scan(scan_token, None, verified=False)
    if history_id is not None:
        update_camera_result(history_id, result)

    st.divider()
    image_col, message_col = st.columns([1, 1.1])

    with image_col:
        st.image(image, caption=source_caption, use_container_width=True)

    with message_col:
        st.error("❓ Unknown / Not Recognised")
        st.markdown("### Let's try another picture")
        st.write(
            "FruitScan is not sure this picture matches one of the five fruits it knows."
        )
        st.caption(
            "🍎 Apple • 🍌 Banana • 🍇 Grape • 🥭 Mango • 🍓 Strawberry"
        )
        st.info("Try one clear fruit, good light, and a simple background.")

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


def show_child_result(
    image: Image.Image,
    verifier: FruitVerifier,
    source_caption: str,
    scan_token: str,
    history_id: str | None = None,
) -> None:
    with st.spinner("🔎 FruitScan is checking the picture..."):
        result = verifier.predict(image)

    if not result.verified:
        show_unknown_gateway(
            image,
            source_caption,
            result,
            history_id,
            scan_token,
        )
        return

    record_scan(scan_token, result.label, verified=True)
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
            st.info("⭐ New fruit unlocked!")

        st.caption("Want to practise the name?")
        repeat_pronunciation_button(effective_label)
        show_correction_controls(scan_token, effective_label, history_id)

    st.markdown("### 🌟 Ready for more?")
    st.info(
        f"Open **🌟 Activities**. Your {emoji} **{effective_label}** is already selected."
    )


def show_think_activity(fruit_name: str) -> None:
    activity_header("think", fruit_name, FRUIT_EMOJI[fruit_name])
    st.caption("Look closer, predict, compare, put things in order, and explain.")
    show_picture_story(fruit_name)
    show_thinking_lesson(fruit_name, namespace=f"activity-{fruit_name.lower()}")
    show_plant_needs_cards(FRUIT_EDUCATION[fruit_name])


def show_kitchen_activity(fruit_name: str) -> None:
    activity_header("kitchen", fruit_name, FRUIT_EMOJI[fruit_name])
    st.caption("Choose a food, drink, or dessert. Ask an adult to help.")
    show_quick_recipe_mode(fruit_name)


def _game_home(fruit_name: str) -> str | None:
    if "game_page" not in st.session_state:
        st.session_state.game_page = "home"
    current = st.session_state.game_page
    if current in {"picture", "thinking"}:
        return current

    st.markdown("### 🎮 Choose a game")
    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            st.markdown(
                "<div style='text-align:center;font-size:64px'>🖼️</div>",
                unsafe_allow_html=True,
            )
            st.markdown("### Picture Quiz")
            st.write("Look at real fruit photos and choose the correct one.")
            if st.button(
                "PLAY PICTURE QUIZ",
                key="open_picture_game",
                type="primary",
                use_container_width=True,
            ):
                st.session_state.game_page = "picture"
                st.rerun()

    with right:
        with st.container(border=True):
            st.markdown(
                "<div style='text-align:center;font-size:64px'>🧩</div>",
                unsafe_allow_html=True,
            )
            st.markdown("### Thinking Games")
            st.write("Solve mystery, sorting, and odd-one-out challenges.")
            if st.button(
                "PLAY THINKING GAMES",
                key="open_thinking_game",
                use_container_width=True,
            ):
                st.session_state.game_page = "thinking"
                st.rerun()

    show_adaptive_challenge(fruit_name)
    return None


def show_games_activity(fruit_name: str) -> None:
    activity_header("games", fruit_name, FRUIT_EMOJI[fruit_name])
    game = _game_home(fruit_name)
    if game is None:
        return

    if st.button(
        "⬅️ Back to Game Menu",
        key=f"game_back_{game}",
        use_container_width=True,
    ):
        st.session_state.game_page = "home"
        st.rerun()

    if game == "picture":
        show_picture_quiz(fruit_name, namespace=f"activity-{fruit_name.lower()}")
        st.divider()
        show_adaptive_challenge(fruit_name)
    else:
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
        st.write("Open **Adult → Model Setup** to prepare the trained model.")
    else:
        show_progress_banner()
        with st.expander("📖 My Fruit Collection", expanded=False):
            show_collection_book()

        st.divider()
        st.header("📷 Scan a Fruit")
        st.write("Choose one big button.")

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
            show_camera_guidance()
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
                st.info("Take a photo above. Your picture will appear below.")

        elif mode == "upload":
            st.markdown("### 🖼️ Choose a Picture")
            st.info("Choose one clear fruit picture with good light.")
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
            if st.button(
                "↩️ Choose a different picture method",
                key="change_scan_mode",
                use_container_width=True,
            ):
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
            with st.expander("📸 My Camera History", expanded=False):
                show_camera_history()

with activities_tab:
    st.header("🌟 Activities")
    st.caption("Choose a fruit, then choose one big activity.")
    activity_fruit = activity_fruit_picker()
    st.divider()

    selected_activity = activity_home(
        activity_fruit,
        FRUIT_EMOJI[activity_fruit],
    )
    if selected_activity == "think":
        show_think_activity(activity_fruit)
    elif selected_activity == "kitchen":
        show_kitchen_activity(activity_fruit)
    elif selected_activity == "games":
        show_games_activity(activity_fruit)

with learn_tab:
    show_progress_banner()
    with st.expander("📖 My Fruit Collection", expanded=True):
        show_collection_book()
    st.divider()
    show_learning_browser()

with adult_tab:
    dashboard_tab, setup_tab, about_tab = st.tabs(
        ["📊 Dashboard", "🛠️ Model Setup", "ℹ️ About"]
    )

    with dashboard_tab:
        show_adult_dashboard()

    with setup_tab:
        show_model_setup(
            CHECKPOINT,
            SUMMARY,
            DATA_DIR,
            UPLOAD_DIR,
        )

    with about_tab:
        show_about()
