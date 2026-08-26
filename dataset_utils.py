from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
EXPECTED_SPLITS = ("train", "valid", "test")


def image_files(folder: Path) -> Iterable[Path]:
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def discover_dataset_root(base: Path) -> Path:
    """Find the directory containing train/valid/test folders."""
    base = base.expanduser().resolve()
    if all((base / split).is_dir() for split in EXPECTED_SPLITS):
        return base
    for candidate in base.rglob("*"):
        if candidate.is_dir() and all((candidate / split).is_dir() for split in EXPECTED_SPLITS):
            return candidate
    raise FileNotFoundError(
        f"Could not find a dataset root containing {EXPECTED_SPLITS} under: {base}"
    )


def scan_dataset(root: Path, verify_images: bool = True) -> Dict:
    root = discover_dataset_root(Path(root))
    summary: Dict = {
        "dataset_root": str(root),
        "splits": {},
        "classes": [],
        "total_images": 0,
        "invalid_images": [],
    }

    class_sets = []
    for split in EXPECTED_SPLITS:
        split_dir = root / split
        classes = sorted(path.name for path in split_dir.iterdir() if path.is_dir())
        class_sets.append(set(classes))
        split_counts = {}
        for class_name in classes:
            files = list(image_files(split_dir / class_name))
            split_counts[class_name] = len(files)
            summary["total_images"] += len(files)
            if verify_images:
                for file_path in files:
                    try:
                        with Image.open(file_path) as image:
                            image.verify()
                    except Exception as exc:
                        summary["invalid_images"].append(
                            {"path": str(file_path), "error": str(exc)}
                        )
        summary["splits"][split] = split_counts

    if not class_sets or any(classes != class_sets[0] for classes in class_sets[1:]):
        raise ValueError("Train, valid and test folders do not contain the same fruit classes.")

    summary["classes"] = sorted(class_sets[0])
    if not summary["classes"]:
        raise ValueError("No fruit classes were found in the dataset.")
    if summary["invalid_images"]:
        raise ValueError(
            f"Dataset contains {len(summary['invalid_images'])} invalid image(s). "
            "See the generated scan summary for details."
        )
    return summary


def save_summary(summary: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def format_counts(summary: Dict) -> List[Tuple[str, str, int]]:
    rows: List[Tuple[str, str, int]] = []
    for split, counts in summary.get("splits", {}).items():
        for class_name, count in counts.items():
            rows.append((split, class_name, int(count)))
    return rows
