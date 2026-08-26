from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from dataset_utils import save_summary, scan_dataset
from prepare_dataset import extract_dataset
from verifier import FruitVerifier

CHECKPOINT = Path("artifacts/fruit_classifier.pt")
SUMMARY = Path("artifacts/dataset_summary.json")
DATA_DIR = Path("data")
UPLOAD_DIR = Path("artifacts/uploads")

st.set_page_config(page_title="FruitScan AI", page_icon="🍓", layout="wide")
st.title("🍓 FruitScan AI")
st.caption(
    "Upload a fruit image, classify it, and verify the result against the learned dataset profile."
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


predict_tab, setup_tab, about_tab = st.tabs(
    ["🔎 Fruit Verification", "🛠️ Model Setup", "ℹ️ About"]
)

with setup_tab:
    st.header("Model Setup")
    st.write(
        "Use this page to prepare the Kaggle dataset and create the trained verifier without typing the Python commands manually."
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
                st.error(f"This checkpoint is not compatible with FruitScan AI: {exc}")
            else:
                st.success("Model installed successfully.")
                st.rerun()

    st.divider()
    st.subheader("Option B — Build the model from your Kaggle ZIP")
    st.caption(
        "This is best done on your own computer. Streamlit Community Cloud has limited CPU/RAM, so full model training there can be slow or may stop."
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
        st.info(
            "For the best speed, run this Streamlit app on a computer with a GPU. The first training run may also download MobileNetV3 pretrained weights."
        )
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
            with st.status("Training FruitScan AI...", expanded=True) as status:
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
                    st.success("The trained model is ready. Open the Fruit Verification tab.")
                    st.rerun()
                else:
                    status.update(label="Training failed", state="error")
                    st.error(
                        "Training did not finish successfully. Check the log above. On Streamlit Cloud, try training locally instead."
                    )

with predict_tab:
    verifier, verifier_error = get_verifier()
    if verifier is None:
        st.warning("The prediction model is not ready yet.")
        if verifier_error:
            st.error(f"Checkpoint loading error: {verifier_error}")
        st.write(
            "Open **Model Setup** above. You can either install a trained `fruit_classifier.pt` file or prepare the dataset and train the model from the Streamlit interface."
        )
    else:
        with st.sidebar:
            st.header("System status")
            st.success("Classifier ready")
            st.success("Dataset verifier ready")
            meta = verifier.metadata
            st.write(f"**Classes:** {', '.join(verifier.class_names)}")
            st.write(f"**Test accuracy:** {float(meta.get('test_accuracy', 0)) * 100:.2f}%")
            dataset_summary = meta.get("dataset_summary", {})
            st.write(f"**Dataset images:** {dataset_summary.get('total_images', 'N/A')}")
            st.write(f"**Confidence threshold:** {verifier.confidence_threshold * 100:.1f}%")

        st.subheader("Upload a fruit image")
        uploaded = st.file_uploader(
            "Choose a JPG, PNG, JPEG, or WebP image",
            type=["jpg", "jpeg", "png", "webp"],
            key="fruit_image",
        )

        if uploaded is None:
            st.info("Supported classes: Apple, Banana, Grape, Mango, and Strawberry.")
        else:
            try:
                image = Image.open(uploaded).convert("RGB")
            except Exception:
                st.error("Could not read this file as an image.")
            else:
                with st.spinner("Scanning and verifying image..."):
                    result = verifier.predict(image)

                left, right = st.columns([1, 1.25])
                with left:
                    st.image(image, caption="Uploaded image", use_container_width=True)
                with right:
                    if result.verified:
                        st.success(f"✅ VERIFIED: {result.label}")
                        st.subheader(result.label)
                    else:
                        st.warning("⚠️ UNKNOWN / NOT VERIFIED")
                        st.subheader("The system rejected this image")

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Classifier confidence", f"{result.confidence * 100:.2f}%")
                    c2.metric("Dataset similarity", f"{result.dataset_similarity * 100:.2f}%")
                    c3.metric("Verification score", f"{result.verification_score:.1f}%")
                    st.metric("Class separation margin", f"{result.class_margin * 100:.2f}%")

                    if result.reasons:
                        st.write("**Why it was rejected:**")
                        for reason in result.reasons:
                            st.write(f"- {reason}")

                st.divider()
                st.subheader("All fruit probabilities")
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

with about_tab:
    st.header("How FruitScan AI works")
    st.markdown(
        """
FruitScan AI uses more than the highest classification probability.

1. **MobileNetV3-Small classifier** predicts Apple, Banana, Grape, Mango, or Strawberry.
2. The uploaded picture is converted to a learned **feature embedding**.
3. The feature embedding is compared with **dataset class centroids** created from the training images.
4. A **confidence gate** rejects weak classifier predictions.
5. A **similarity gate** rejects images that do not resemble the learned fruit profile.
6. A **class-separation gate** rejects ambiguous images that look too similar to multiple classes.

Only when all three verification gates pass does the app display **VERIFIED**. Otherwise it returns **UNKNOWN / NOT VERIFIED**.
        """
    )
    st.info(
        "This reduces forced misclassification, but no five-class image model can guarantee perfect detection of every possible non-fruit image."
    )
