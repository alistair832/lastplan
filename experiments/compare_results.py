from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def member1_result() -> dict:
    data = load_json(ROOT / "artifacts" / "metrics.json")
    macro = data["classification_report"]["macro avg"]
    return {
        "member": 1,
        "algorithm": "MobileNetV3-Small transfer learning",
        "test_accuracy": data["test_accuracy"],
        "macro_precision": macro["precision"],
        "macro_recall": macro["recall"],
        "macro_f1": macro["f1-score"],
        "confusion_matrix": data["confusion_matrix"],
        "classification_report": data["classification_report"],
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    models = [
        member1_result(),
        load_json(RESULTS / "custom_cnn.json"),
        load_json(RESULTS / "hog_svm.json"),
    ]

    columns = ["member", "algorithm", "test_accuracy", "macro_precision", "macro_recall", "macro_f1"]
    with (RESULTS / "model_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for model in models:
            writer.writerow({key: model[key] for key in columns})

    summary = {
        "models": models,
        "best_by_accuracy": max(models, key=lambda item: item["test_accuracy"])["algorithm"],
        "best_by_macro_f1": max(models, key=lambda item: item["macro_f1"])["algorithm"],
    }
    (RESULTS / "model_comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Three-Member Model Comparison",
        "",
        "All three approaches use the same five fruit classes and the same official test split.",
        "",
        "| Member | Algorithm | Accuracy | Precision | Recall | F1-score |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model in models:
        lines.append(
            f"| {model['member']} | {model['algorithm']} | {model['test_accuracy']:.2%} | "
            f"{model['macro_precision']:.2%} | {model['macro_recall']:.2%} | {model['macro_f1']:.2%} |"
        )
    lines.extend(
        [
            "",
            f"**Highest test accuracy:** {summary['best_by_accuracy']}",
            "",
            f"**Highest macro F1-score:** {summary['best_by_macro_f1']}",
        ]
    )
    (RESULTS / "COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
