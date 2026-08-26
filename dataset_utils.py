from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

SPLITS = ("train", "valid", "test")


def find_dataset_root(start: str | Path) -> Path:
    """Return the directory that directly contains train/, valid/, and test/."""
    start_path = Path(start).expanduser().resolve()
    if not start_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {start_path}")

    candidates = [start_path]
    candidates.extend(p for p in start_path.rglob("*") if p.is_dir())

    for candidate in candidates:
        if all((candidate / split).is_dir() for split in SPLITS):
            return candidate

    raise FileNotFoundError(
        f"Could not find a dataset root containing {', '.join(SPLITS)} under {start_path}"
    )


def class_names(dataset_root: str | Path) -> List[str]:
    root = find_dataset_root(dataset_root)
    names = sorted(p.name for p in (root / "train").iterdir() if p.is_dir())
    if not names:
        raise ValueError(f"No class folders found in {root / 'train'}")
    return names


def dataset_counts(dataset_root: str | Path) -> Dict[str, Dict[str, int]]:
    root = find_dataset_root(dataset_root)
    names = class_names(root)
    result: Dict[str, Dict[str, int]] = {}

    for split in SPLITS:
        split_counts: Dict[str, int] = {}
        for name in names:
            folder = root / split / name
            if not folder.is_dir():
                split_counts[name] = 0
                continue
            split_counts[name] = sum(1 for p in folder.iterdir() if p.is_file())
        result[split] = split_counts

    return result


def save_class_names(names: List[str], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(names, indent=2), encoding="utf-8")
