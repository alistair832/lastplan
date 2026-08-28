from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from dataset_utils import scan_dataset


class CustomCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def make_loaders(root: Path, image_size: int, batch_size: int):
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    train_ds = datasets.ImageFolder(root / "train", transform=train_tf)
    valid_ds = datasets.ImageFolder(root / "valid", transform=eval_tf)
    test_ds = datasets.ImageFolder(root / "test", transform=eval_tf)
    if not (train_ds.classes == valid_ds.classes == test_ds.classes):
        raise ValueError("Class folders must match across train/valid/test.")
    workers = 2
    return (
        train_ds,
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers),
        DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=workers),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers),
    )


def run_epoch(model, loader, device, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        y_true.extend(labels.tolist())
        y_pred.extend(logits.argmax(1).cpu().tolist())
    return y_true, y_pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Member 2: custom CNN fruit classifier")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("experiments/results/custom_cnn.json"))
    args = parser.parse_args()

    torch.manual_seed(42)
    summary = scan_dataset(args.data, verify_images=False)
    root = Path(summary["dataset_root"])
    train_ds, train_loader, valid_loader, test_loader = make_loaders(root, args.image_size, args.batch_size)
    classes = train_ds.classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CustomCNN(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_state = None
    best_val = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, device, criterion, optimizer)
        valid_loss, valid_acc = run_epoch(model, valid_loader, device, criterion)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | train loss {train_loss:.4f} acc {train_acc:.3%} | "
            f"valid loss {valid_loss:.4f} acc {valid_acc:.3%}"
        )
        if valid_acc > best_val:
            best_val = valid_acc
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("Custom CNN training did not produce a model.")
    model.load_state_dict(best_state)
    y_true, y_pred = evaluate(model, test_loader, device)
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    result = {
        "member": 2,
        "algorithm": "Custom CNN from scratch",
        "image_size": args.image_size,
        "best_validation_accuracy": best_val,
        "test_accuracy": report["accuracy"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classes": classes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("algorithm", "test_accuracy", "macro_precision", "macro_recall", "macro_f1")}, indent=2))


if __name__ == "__main__":
    main()
