from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from dataset_utils import dataset_counts, find_dataset_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract the Fruits Classification Kaggle ZIP.")
    parser.add_argument("--zip", required=True, dest="zip_path", help="Path to the downloaded Kaggle ZIP file")
    parser.add_argument("--output", default="data", help="Extraction directory (default: data)")
    args = parser.parse_args()

    zip_path = Path(args.zip_path).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()

    if not zip_path.is_file():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output)

    root = find_dataset_root(output)
    counts = dataset_counts(root)

    print(f"Dataset ready at: {root}")
    for split, split_counts in counts.items():
        total = sum(split_counts.values())
        details = ", ".join(f"{name}={count}" for name, count in split_counts.items())
        print(f"{split:>5}: {total:>5} images | {details}")


if __name__ == "__main__":
    main()
