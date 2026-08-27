from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from dataset_utils import save_summary, scan_dataset
from prepare_dataset import extract_dataset
from verifier import FruitVerifier


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


def show_model_setup(
    checkpoint: Path,
    summary_path: Path,
    data_dir: Path,
    upload_dir: Path,
) -> None:
    st.header("🛠️ Model Setup")
    st.write(
        "This area is for a teacher, parent, or project developer. "
        "Children do not need to use these controls."
    )

    st.markdown("### ✅ Do I need to use Model Setup?")
    if checkpoint.exists():
        st.success(
            "Your trained FruitScan model is already installed and ready. "
            "You do not need to upload or train anything to use Scan & Think."
        )
        st.caption(
            "Only use the options below if you want to replace the trained model, "
            "restore another checkpoint, or deliberately retrain the project."
        )
        st.download_button(
            "⬇️ Download a backup of the trained model",
            data=checkpoint.read_bytes(),
            file_name="fruit_classifier.pt",
            mime="application/octet-stream",
            use_container_width=True,
        )
    else:
        st.warning(
            "No trained model is installed. Use Option A if you already have "
            "fruit_classifier.pt, or Option B if you need to build one from the dataset."
        )

    st.divider()
    st.subheader("Option A — Install an existing trained model")
    st.caption(
        "Use this only when you already have a FruitScan `fruit_classifier.pt` file. "
        "For the current deployed project, this is normally not required."
    )

    model_upload = st.file_uploader(
        "Upload fruit_classifier.pt",
        type=["pt"],
        key="checkpoint_upload",
        help="Use a checkpoint produced by this FruitScan project only.",
    )

    if model_upload is not None:
        if model_upload.size > 100 * 1024 * 1024:
            st.error("The checkpoint is larger than 100 MB and was not accepted.")
        elif st.button("Install uploaded model", use_container_width=True):
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(model_upload.getbuffer())
            st.cache_resource.clear()
            try:
                FruitVerifier(checkpoint)
            except Exception as exc:
                checkpoint.unlink(missing_ok=True)
                st.error(f"This checkpoint is not compatible with FruitScan Kids: {exc}")
            else:
                st.success("Model installed successfully. Scan & Think can now use it.")
                st.rerun()

    st.divider()
    st.subheader("Option B — Retrain from the Fruits Classification ZIP")
    st.warning(
        "⚠️ Only use this if you intentionally want to retrain the model. "
        "It is not needed for normal child use."
    )
    st.caption(
        "Training can use significant CPU, RAM, and time. A local computer or cloud training "
        "environment is usually more reliable than a small Streamlit deployment."
    )

    st.markdown(
        """
**Retraining steps**

1. Upload the same **Fruits Classification ZIP** used by this project.  
2. Click **Extract and scan dataset**.  
3. Check that the five classes are **Apple, Banana, Grape, Mango, Strawberry**.  
4. Choose the number of training epochs.  
5. Click **Train classifier + verifier**.  
6. When training finishes, FruitScan automatically uses the new checkpoint.
        """
    )

    zip_upload = st.file_uploader(
        "Upload the Fruits Classification ZIP",
        type=["zip"],
        key="dataset_zip",
    )

    if zip_upload is not None:
        if st.button("1️⃣ Extract and scan dataset", use_container_width=True):
            upload_dir.mkdir(parents=True, exist_ok=True)
            zip_path = upload_dir / "fruits-classification.zip"
            zip_path.write_bytes(zip_upload.getbuffer())
            try:
                with st.spinner("Extracting and checking the dataset images..."):
                    root = extract_dataset(zip_path, data_dir)
                    summary = scan_dataset(root, verify_images=True)
                    save_summary(summary, summary_path)
                st.success("Dataset prepared successfully.")
                show_dataset_summary(summary)
            except Exception as exc:
                st.exception(exc)

    if summary_path.exists():
        try:
            saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            with st.expander("Current dataset scan", expanded=False):
                show_dataset_summary(saved_summary)
        except Exception:
            pass

    dataset_ready = all(
        (data_dir / split).exists() for split in ("train", "valid", "test")
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
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                "-u",
                "train.py",
                "--data",
                str(data_dir),
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
                if return_code == 0 and checkpoint.exists():
                    status.update(label="Training completed", state="complete")
                    st.cache_resource.clear()
                    st.success("The trained model is ready. Open Scan & Think.")
                    st.rerun()
                else:
                    status.update(label="Training failed", state="error")
                    st.error(
                        "Training did not finish successfully. Check the log above."
                    )


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
**8. ⚙️ Adult** — View the learning dashboard, check model status, and access project information.
        """
    )
    st.warning(
        "👨‍👩‍👧 Food activities require adult supervision. Adults should handle knives, "
        "blenders, heat, allergies, and age-appropriate choking safety."
    )
    st.info(
        "Camera History and the learning dashboard are session-based. Camera photos are "
        "temporary and are not saved to GitHub."
    )
