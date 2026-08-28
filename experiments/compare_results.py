from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mobile_result(path: Path) -> dict:
    data = load_json(path)
    report = data["classification_report"]
    return {
        "algorithm": "MobileNetV3-Small Transfer Learning",
        "accuracy": float(data["test_accuracy"]),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "training_seconds": None,
        "classification_report": report,
        "confusion_matrix": data["confusion_matrix"],
        "class_names": list(data["dataset_summary"]["classes"]),
        "source": str(path),
    }


def experiment_result(path: Path) -> dict:
    data = load_json(path)
    return {
        "algorithm": data["algorithm"],
        "accuracy": float(data["accuracy"]),
        "macro_precision": float(data["macro_precision"]),
        "macro_recall": float(data["macro_recall"]),
        "macro_f1": float(data["macro_f1"]),
        "training_seconds": data.get("training_seconds"),
        "classification_report": data["classification_report"],
        "confusion_matrix": data["confusion_matrix"],
        "class_names": data["class_names"],
        "source": str(path),
    }


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def markdown_report(rows: list[dict]) -> str:
    lines = [
        "# Three-Algorithm Fruit Classification Results",
        "",
        "All methods use the same dataset classes and the provided train/validation/test folder split.",
        "",
        "## Overall comparison",
        "",
        "| Rank | Algorithm | Accuracy | Macro Precision | Macro Recall | Macro F1 | Training Time |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        seconds = row.get("training_seconds")
        time_text = "Not recorded" if seconds is None else f"{seconds:.1f} s"
        lines.append(
            f"| {row['rank']} | {row['algorithm']} | {pct(row['accuracy'])} | "
            f"{pct(row['macro_precision'])} | {pct(row['macro_recall'])} | "
            f"{pct(row['macro_f1'])} | {time_text} |"
        )

    for row in rows:
        lines += [
            "",
            f"## {row['algorithm']}",
            "",
            "### Per-class metrics",
            "",
            "| Class | Precision | Recall | F1-score | Support |",
            "|---|---:|---:|---:|---:|",
        ]
        report = row["classification_report"]
        for class_name in row["class_names"]:
            data = report[class_name]
            lines.append(
                f"| {class_name} | {pct(float(data['precision']))} | "
                f"{pct(float(data['recall']))} | {pct(float(data['f1-score']))} | "
                f"{int(data['support'])} |"
            )
        lines += [
            "",
            "### Confusion matrix",
            "",
            "Rows are actual classes and columns are predicted classes.",
            "",
            "| Actual \\ Predicted | " + " | ".join(row["class_names"]) + " |",
            "|---|" + "|".join(["---:" for _ in row["class_names"]]) + "|",
        ]
        for class_name, matrix_row in zip(row["class_names"], row["confusion_matrix"]):
            lines.append(
                f"| {class_name} | " + " | ".join(str(int(value)) for value in matrix_row) + " |"
            )

    lines += [
        "",
        "## Method characteristics for discussion",
        "",
        "| Method | Main advantages | Main disadvantages / characteristics |",
        "|---|---|---|",
        "| MobileNetV3-Small Transfer Learning | Lightweight pretrained visual features; strong deployment suitability; usually learns well with limited task-specific training | Depends on pretrained representations; more complex than traditional ML |",
        "| Custom CNN from Scratch | Learns task-specific features; architecture is easy to explain and modify | Starts with no pretrained knowledge; may need more training and may overfit |",
        "| HOG + Linear SVM | Traditional ML baseline; interpretable feature pipeline; comparatively simple classifier | Relies on handcrafted HOG features and may struggle with colour/texture or complex visual variation |",
        "",
        "> Select the final model using the measured results together with deployment suitability. Do not claim a model is best until all experiments have completed.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine the three member model results.")
    parser.add_argument("--mobilenet", type=Path, default=Path("artifacts/metrics.json"))
    parser.add_argument("--custom-cnn", type=Path, default=Path("experiments/results/custom_cnn.json"))
    parser.add_argument("--hog-svm", type=Path, default=Path("experiments/results/hog_svm.json"))
    parser.add_argument("--output", type=Path, default=Path("experiments/results/model_comparison.csv"))
    parser.add_argument("--summary", type=Path, default=Path("experiments/results/model_comparison.json"))
    parser.add_argument("--markdown", type=Path, default=Path("experiments/results/RESULTS.md"))
    args = parser.parse_args()

    rows = [
        mobile_result(args.mobilenet),
        experiment_result(args.custom_cnn),
        experiment_result(args.hog_svm),
    ]
    rows.sort(key=lambda row: row["accuracy"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank", "algorithm", "accuracy", "macro_precision",
        "macro_recall", "macro_f1", "training_seconds", "source"
    ]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})

    args.summary.write_text(json.dumps({"models": rows}, indent=2), encoding="utf-8")
    args.markdown.write_text(markdown_report(rows), encoding="utf-8")

    print("Final comparison:")
    for row in rows:
        print(
            f"{row['rank']}. {row['algorithm']}: "
            f"acc={row['accuracy']:.3%}, precision={row['macro_precision']:.3%}, "
            f"recall={row['macro_recall']:.3%}, f1={row['macro_f1']:.3%}"
        )
    print(f"Saved: {args.output}")
    print(f"Saved: {args.markdown}")


if __name__ == "__main__":
    main()
