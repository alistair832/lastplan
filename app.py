from __future__ import annotations

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
from dataset_utils import save_summary, scan_dataset
from education_ui import show_fruit_lesson, show_learning_browser
from prepare_dataset import extract_dataset
from verifier import FruitVerifier

CHECKPOINT = Path("artifacts/fruit_classifier.pt")
SUMMARY = Path("artifacts/dataset_summary.json")
DATA_DIR = Path("data")
UPLOAD_DIR = Path("artifacts/uploads")

st.set_page_config(page_title="FruitScan Kids", page_icon="🍓", layout="wide")
st.title("🍓 FruitScan Kids")
st.caption(
    "Scan a fruit, hear its name, discover its seeds, learn how it grows, and make simple fruit food with an adult."
)


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


def show_prediction_result(
    image: Image.Image,
    verifier: FruitVerifier,
    source_caption: str,
    history_id: str | None = None,
) -> None:
    with st.spinner("Scanning and identifying the fruit..."):
        result = verifier.predict(image)

    if history_id is not None:
        update_camera_result(history_id, result)

    left, right = st.columns([1, 1.25])
    with left:
        st.image(image, caption=source_caption, use_container_width=True)

    with right:
        if result.verified:
            st.success(f"✅ I found a {result.label}!")
            st.header(result.label)
            st.write("Great job! Now we can learn about this fruit together.")
        else:
            st.warning("🤔 I am not sure what this is yet.")
            st.subheader("Try another picture")
            st.write(
                "Place one Apple, Banana, Grape, Mango, or Strawberry clearly in the picture and try again."
            )

        with st.expander("🔬 Teacher / model details", expanded=False):
            c1, c2, c3 = st.columns(3)
            c1.metric("Classifier confidence", f"{result.confidence * 100:.2f}%")
            c2.metric("Dataset similarity", f"{result.dataset_similarity * 100:.2f}%")
            c3.metric("Verification score", f"{result.verification_score:.1f}%")
            st.metric("Class separation margin", f"{result.class_margin * 100:.2f}%")
            if result.reasons:
                st.write("**Why the image was not verified:**")
                for reason in result.reasons:
                    st.write(f"- {reason}")

    if result.verified:
        st.divider()
        show_fruit_lesson(result.label, heading=False)

    with st.expander("📊 All fruit probabilities", expanded=False):
        prob_df = pd.DataFrame(
            [
                {"Fruit": name, "Probability": value}
                for name, value in result.probabilities.items()
            ]
        ).sort_values("Probability", ascending=False)
        prob_df["Probability %"] = prob_df["Probability"].map(
            lambda x: f"{x * 100:.2f}%"
        )
        st.dataframe(
            prob_df[["Fruit", "Probability %"]],
            hide_index=True,
            use_container_width=True,
        )
        st.bar_chart(prob_df.set_index("Fruit")["Probability"])


scan_tab, learn_tab, setup_tab, about_tab = st.tabs(
    ["📷 Scan & Learn", "🎓 Learn Fruits", "🛠️ Model Setup", "ℹ️ About"]
)

with scan_tab:
    verifier, verifier_error = get_verifier()
    if verifier is None:
        st.warning("The fruit recognition model is not ready yet.")
        if verifier_error:
            st.error(f"Checkpoint loading error: {verifier_error}")
        st.write(
            "Open **Model Setup**. You can install a trained `fruit_classifier.pt` file or prepare the dataset and train the model."
        )
    else:
        with st.sidebar:
            st.header("🍓 FruitScan Kids")
            st.success("Fruit recognition ready")
            st.success("Camera ready")
            st.info("Learning mode: Early Years")
            meta = verifier.metadata
            st.write(f"**Fruits:** {', '.join(verifier.class_names)}")
            st.write(f"**Model test accuracy:** {float(meta.get('test_accuracy', 0)) * 100:.2f}%")

        st.header("📷 Scan a Fruit")
        st.write(
            "Take a picture of one fruit or upload a photo. When FruitScan recognises it, a learning lesson will open automatically."
        )

        input_mode = st.radio(
            "Choose a picture source",
            ["📁 Upload Image", "🎥 Live Front Camera"],
            horizontal=True,
        )

        image = None
        source_caption = "Fruit image"
        history_id = None

        if input_mode == "📁 Upload Image":
            uploaded = st.file_uploader(
                "Choose a JPG, PNG, JPEG, or WebP fruit image",
                type=["jpg", "jpeg", "png", "webp"],
                key="fruit_image",
            )
            if uploaded is not None:
                try:
                    image = Image.open(uploaded).convert("RGB")
                    source_caption = "Uploaded fruit"
                except Exception:
                    st.error("I could not read this file as an image.")
            else:
                st.info("Choose a fruit picture to start learning.")

        else:
            st.markdown("### 🎥 Live Front Camera")
            st.caption(
                "Allow camera permission, hold one fruit clearly in front of the camera, then take a picture. Every camera photo is added to My Fruit Camera History."
            )
            camera_photo = st.camera_input(
                "Take a fruit picture",
                key="front_camera",
                help="Your browser or phone controls which physical camera is used.",
            )

            if camera_photo is not None:
                try:
                    remember_camera_photo(camera_photo.getvalue())
                except Exception:
                    st.error("I could not save this camera picture to the session history.")

            try:
                image, source_caption = selected_camera_image()
                history_id = selected_camera_id()
            except Exception:
                image = None
                history_id = None
                st.error("I could not reopen the selected camera picture.")

            if image is None:
                st.info("Take a fruit picture above, or choose a photo from Camera History to review it.")

        st.caption("FruitScan currently teaches: 🍎 Apple · 🍌 Banana · 🍇 Grape · 🥭 Mango · 🍓 Strawberry")

        if image is not None:
            show_prediction_result(
                image,
                verifier,
                source_caption,
                history_id=history_id,
            )

        if input_mode == "🎥 Live Front Camera":
            st.divider()
            show_camera_history()

with learn_tab:
    show_learning_browser()

with setup_tab:
    st.header("🛠️ Model Setup")
    st.write(
        "This section is for teachers, parents, or project setup. Children can use Scan & Learn and Learn Fruits."
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

    dataset_ready = all((DATA_DIR / split).exists() for split in ("train", "valid", "test"))
    if dataset_ready:
        st.success("Training dataset is ready.")
        epochs = st.slider("Training epochs", min_value=4, max_value=16, value=8, step=1)
        if st.button("2️⃣ Train classifier + verifier", type="primary", use_container_width=True):
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
                    st.success("The trained model is ready. Open Scan & Learn.")
                    st.rerun()
                else:
                    status.update(label="Training failed", state="error")
                    st.error("Training did not finish successfully. Check the log above.")

with about_tab:
    st.header("ℹ️ About FruitScan Kids")
    st.write(
        "FruitScan Kids is an educational fruit-recognition project designed to help early-years learners connect real objects with simple science and everyday activities."
    )
    st.markdown(
        """
### Learning journey

**1. See it** — Upload a picture or use the camera.  
**2. Name it** — FruitScan identifies the fruit.  
**3. Say it** — Press the pronunciation button and repeat the word.  
**4. Explore it** — Learn what the fruit and its seeds look like.  
**5. Grow it** — Learn about weather, soil, water, sunlight, flowers, and fruit growth.  
**6. Understand leaves** — Learn a simple idea of photosynthesis.  
**7. Make something** — Try a small food, drink, or dessert activity with an adult.  
**8. Review it** — My Fruit Camera History lets learners reopen photos and their lessons during the current session.

### Fruits currently included

🍎 Apple · 🍌 Banana · 🍇 Grape · 🥭 Mango · 🍓 Strawberry
        """
    )
    st.warning(
        "Food activities are educational examples for adult-supervised use. Adults should manage knives, blenders, heat, allergies, and age-appropriate choking safety."
    )
    st.info(
        "Camera History is temporary and session-only. It keeps the newest 12 camera photos, their FruitScan result, and simple learning progress while the current Streamlit session is active. Photos are not saved to GitHub."
    )
    st.info(
        "The AI classifier is a learning aid, not a perfect identification system. If FruitScan is unsure, it asks the learner to try another picture instead of forcing a fruit label."
    )
