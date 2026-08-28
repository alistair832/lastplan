from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from skimage.color import rgb2gray
from skimage.feature import hog

from dataset_utils import discover_dataset_root, image_files
from experiments.common import build_metrics, save_metrics


def load_split(split_dir: Path, class_names: list[str], image_size: int):
    features: list[np.ndarray] = []
    labels: list[int] = []
    for class_index, class_name in enumerate(class_names):
        files = sorted(image_files(split_dir / class_name))
        for path in files:
            with Image.open(path) as image:
                rgb = np.asarray(
                    image.convert("RGB").resize((image_size, image_size)),
                    dtype=np.float32,
                ) / 255.0
            gray = rgb2gray(rgb)
            vector = hog(
                gray,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm="L2-Hys",
                feature_vector=True,
            )
            features.append(vector.astype(np.float32, copy=False))
            labels.append(class_index)
    return np.stack(features), np.asarray(labels, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Member 3: HOG + Linear SVM fruit classifier.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("experiments/results/hog_svm.json"))
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--c-values", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    args = parser.parse_args()

    root = discover_dataset_root(args.data)
    class_names = sorted(path.name for path in (root / "train").iterdir() if path.is_dir())

    print("Extracting HOG features...")
    x_train, y_train = load_split(root / "train", class_names, args.image_size)
    x_valid, y_valid = load_split(root / "valid", class_names, args.image_size)
    x_test, y_test = load_split(root / "test", class_names, args.image_size)

    started = time.perf_counter()
    best_c = None
    best_valid_accuracy = -1.0
    for c_value in args.c_values:
        candidate = make_pipeline(
            StandardScaler(),
            LinearSVC(C=c_value, dual="auto", max_iter=10000, random_state=42),
        )
        candidate.fit(x_train, y_train)
        valid_accuracy = float((candidate.predict(x_valid) == y_valid).mean())
        print(f"C={c_value:g} validation accuracy: {valid_accuracy:.3%}")
        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            best_c = c_value

    if best_c is None:
        raise RuntimeError("SVM validation did not produce a model.")

    x_fit = np.concatenate([x_train, x_valid], axis=0)
    y_fit = np.concatenate([y_train, y_valid], axis=0)
    model = make_pipeline(
        StandardScaler(),
        LinearSVC(C=best_c, dual="auto", max_iter=10000, random_state=42),
    )
    model.fit(x_fit, y_fit)
    elapsed = time.perf_counter() - started
    y_pred = model.predict(x_test)

    metrics = build_metrics(
        y_test, y_pred, class_names, elapsed, "HOG + Linear SVM"
    )
    metrics["best_validation_accuracy"] = best_valid_accuracy
    metrics["best_c"] = best_c
    metrics["image_size"] = args.image_size
    metrics["hog"] = {
        "orientations": 9,
        "pixels_per_cell": [8, 8],
        "cells_per_block": [2, 2],
    }
    save_metrics(metrics, args.output)
    print(f"HOG + SVM test accuracy: {metrics['accuracy']:.3%}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
