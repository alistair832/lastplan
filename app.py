from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras

MODEL_PATH = Path("models/fruit_classifier.keras")
CLASSES_PATH = Path("models/class_names.json")
IMAGE_SIZE = (224, 224)

st.set_page_config(page_title="Fruit Classifier", page_icon="🍓", layout="centered")
st.title("Fruit Classification")
st.write("Upload a fruit image and the model will classify it as Apple, Banana, Grape, Mango, or Strawberry.")


@st.cache_resource
def load_assets():
    model = keras.models.load_model(MODEL_PATH)
    names = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    return model, names


if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
    st.warning("Model files are missing. Run `python train.py --data data` first.")
    st.stop()

model, class_names = load_assets()
uploaded = st.file_uploader("Choose a JPG, JPEG, or PNG image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    resized = image.resize(IMAGE_SIZE)
    batch = np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)
    probabilities = model.predict(batch, verbose=0)[0]
    best = int(np.argmax(probabilities))

    st.subheader(f"Prediction: {class_names[best]}")
    st.metric("Confidence", f"{probabilities[best] * 100:.2f}%")

    st.write("Class probabilities")
    ranking = sorted(
        ((class_names[i], float(probabilities[i])) for i in range(len(class_names))),
        key=lambda item: item[1],
        reverse=True,
    )
    for name, probability in ranking:
        st.write(f"**{name}:** {probability * 100:.2f}%")
        st.progress(float(probability))
