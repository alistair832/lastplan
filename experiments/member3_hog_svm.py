from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import LinearSVC
from skimage.feature import hog

from dataset_utils import scan_dataset


def image_to_hog(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((image_size, image_size))
        array = np.asarray(image, dtype=np.float32) / 255.0
    return hog(
        array,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        channel_axis=-1,
        feature_vector=True,
    ).astype(np.float32)


def load_split(root: Path, split: str, classes: list[str], image_size: int):
    features = []
    labels = []
    for class_index, class_name in enumerate(classes):
        folder = root / split / class_name
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                continue
            features.append(image_to_hog(path, image_size))
            labels.append(class_index)
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Member 3: HOG + Linear SVM fruit classifier")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("experiments/results/hog_svm.json"))
    args = parser.parse_args()

    summary = scan_dataset(args.data, verify_images=False)
    root = Path(summary["dataset_root"])
    classes = list(summary["classes"])

    print("Extracting HOG features from training set...")
    x_train, y_train = load_split(root, "train", classes, args.image_size)
    print("Extracting HOG features from test set...")
    x_test, y_test = load_split(root, "test", classes, args.image_size)

    classifier = LinearSVC(C=1.0, dual="auto", max_iter=10000, random_state=42)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    report = classification_report(y_test, y_pred, target_names=classes, output_dict=True, zero_division=0)
    result = {
        "member": 3,
        "algorithm": "HOG + Linear SVM",
        "image_size": args.image_size,
        "test_accuracy": report["accuracy"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classes": classes,
        "feature_dimension": int(x_train.shape[1]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("algorithm", "test_accuracy", "macro_precision", "macro_recall", "macro_f1")}, indent=2))


if __name__ == "__main__":
    main()
