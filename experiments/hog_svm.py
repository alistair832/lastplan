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
    """Extract one compact HOG vector for every image in a dataset split."""
    features: list[np.ndarray] = []
    labels: list[int] = []

    for class_index, class_name in enumerate(class_names):
        files = sorted(image_files(split_dir / class_name))
        print(f"  {split_dir.name}/{class_name}: {len(files)} images", flush=True)

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
    parser = argparse.ArgumentParser(
        description="Member 3: HOG + Linear SVM fruit classifier."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/hog_svm.json"),
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=64,
        help="Compact HOG input size. 64 keeps the experiment practical on CPU.",
    )
    parser.add_argument(
        "--c-value",
        type=float,
        default=1.0,
        help="Fixed Linear SVM regularization value. Kept fixed for a fair, fast baseline.",
    )
    args = parser.parse_args()

    root = discover_dataset_root(args.data)
    class_names = sorted(
        path.name for path in (root / "train").iterdir() if path.is_dir()
    )

    feature_started = time.perf_counter()
    print("Extracting compact HOG features from the full dataset...", flush=True)
    x_train, y_train = load_split(root / "train", class_names, args.image_size)
    x_valid, y_valid = load_split(root / "valid", class_names, args.image_size)
    x_test, y_test = load_split(root / "test", class_names, args.image_size)
    feature_seconds = time.perf_counter() - feature_started

    print(
        f"HOG extraction complete in {feature_seconds:.1f}s; "
        f"feature dimension = {x_train.shape[1]}",
        flush=True,
    )

    model = make_pipeline(
        StandardScaler(),
        LinearSVC(
            C=args.c_value,
            dual=False,
            max_iter=5000,
            random_state=42,
        ),
    )

    train_started = time.perf_counter()
    model.fit(x_train, y_train)
    training_seconds = time.perf_counter() - train_started

    valid_pred = model.predict(x_valid)
    valid_accuracy = float((valid_pred == y_valid).mean())
    y_pred = model.predict(x_test)

    metrics = build_metrics(
        y_test,
        y_pred,
        class_names,
        training_seconds,
        "HOG + Linear SVM",
    )
    metrics["best_validation_accuracy"] = valid_accuracy
    metrics["c_value"] = args.c_value
    metrics["image_size"] = args.image_size
    metrics["feature_extraction_seconds"] = feature_seconds
    metrics["feature_dimension"] = int(x_train.shape[1])
    metrics["hog"] = {
        "orientations": 9,
        "pixels_per_cell": [8, 8],
        "cells_per_block": [2, 2],
    }

    save_metrics(metrics, args.output)
    print(f"Validation accuracy: {valid_accuracy:.3%}")
    print(f"HOG + SVM test accuracy: {metrics['accuracy']:.3%}")
    print(f"SVM training time: {training_seconds:.1f}s")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
