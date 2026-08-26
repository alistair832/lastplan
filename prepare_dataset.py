from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

from dataset_utils import discover_dataset_root, save_summary, scan_dataset


def extract_dataset(zip_path: Path, output_dir: Path) -> Path:
    zip_path = zip_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    with tempfile.TemporaryDirectory(prefix="fruit_dataset_") as temp_dir:
        temp_path = Path(temp_dir)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(temp_path)
        source_root = discover_dataset_root(temp_path)

        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for split in ("train", "valid", "test"):
            shutil.copytree(source_root / split, output_dir / split)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and validate the fruits dataset.")
    parser.add_argument("--zip", required=True, type=Path, help="Path to the downloaded Kaggle ZIP.")
    parser.add_argument("--output", default=Path("data"), type=Path, help="Destination dataset folder.")
    parser.add_argument(
        "--skip-image-check",
        action="store_true",
        help="Only count files instead of opening every image for integrity checking.",
    )
    args = parser.parse_args()

    root = extract_dataset(args.zip, args.output)
    summary = scan_dataset(root, verify_images=not args.skip_image_check)
    save_summary(summary, Path("artifacts/dataset_summary.json"))

    print(f"Dataset ready: {root}")
    print(f"Classes: {', '.join(summary['classes'])}")
    print(f"Total images: {summary['total_images']}")
    for split, counts in summary["splits"].items():
        print(f"{split}: {sum(counts.values())} images -> {counts}")


if __name__ == "__main__":
    main()
