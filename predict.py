from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow import keras

IMAGE_SIZE = (224, 224)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict the fruit class for one image.")
    parser.add_argument("image", help="Path to an image")
    parser.add_argument("--model", default="models/fruit_classifier.keras")
    parser.add_argument("--classes", default="models/class_names.json")
    args = parser.parse_args()

    model = keras.models.load_model(args.model)
    names = json.loads(Path(args.classes).read_text(encoding="utf-8"))

    image = Image.open(args.image).convert("RGB").resize(IMAGE_SIZE)
    batch = np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)
    probabilities = model.predict(batch, verbose=0)[0]

    best = int(np.argmax(probabilities))
    print(f"Prediction: {names[best]}")
    print(f"Confidence: {probabilities[best] * 100:.2f}%")

    print("\nAll classes:")
    for index in np.argsort(probabilities)[::-1]:
        print(f"  {names[int(index)]:<12} {probabilities[int(index)] * 100:6.2f}%")


if __name__ == "__main__":
    main()
