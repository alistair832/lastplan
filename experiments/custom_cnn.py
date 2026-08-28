from __future__ import annotations

import argparse
import copy
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from dataset_utils import discover_dataset_root
from experiments.common import build_metrics, save_metrics


class CustomFruitCNN(nn.Module):
    """Small CNN trained from scratch for the five-fruit classification task."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(192, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loaders(data_root: Path, image_size: int, batch_size: int, workers: int):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_ds = datasets.ImageFolder(data_root / "train", transform=train_tf)
    valid_ds = datasets.ImageFolder(data_root / "valid", transform=eval_tf)
    test_ds = datasets.ImageFolder(data_root / "test", transform=eval_tf)
    if not (train_ds.classes == valid_ds.classes == test_ds.classes):
        raise ValueError("Class folders must match across train/valid/test.")
    pin = torch.cuda.is_available()
    return (
        train_ds,
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=pin),
        DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin),
    )


def run_epoch(model, loader, device, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.inference_mode()
def predict_all(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        y_pred.extend(logits.argmax(1).cpu().tolist())
        y_true.extend(labels.tolist())
    return y_true, y_pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Member 2: custom CNN fruit classifier.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("experiments/results/custom_cnn.json"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    seed_everything()
    root = discover_dataset_root(args.data)
    train_ds, train_loader, valid_loader, test_loader = loaders(
        root, args.image_size, args.batch_size, args.workers
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomFruitCNN(len(train_ds.classes)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_state = None
    best_val = -1.0
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, valid_loader, device, criterion)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3%} | "
            f"valid loss {val_loss:.4f} acc {val_acc:.3%}"
        )
        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print("Early stopping.")
                break

    if best_state is None:
        raise RuntimeError("Custom CNN did not produce a valid model.")
    model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started
    y_true, y_pred = predict_all(model, test_loader, device)
    metrics = build_metrics(
        y_true, y_pred, train_ds.classes, elapsed, "Custom CNN from Scratch"
    )
    metrics["best_validation_accuracy"] = float(best_val)
    metrics["image_size"] = args.image_size
    save_metrics(metrics, args.output)
    print(f"Custom CNN test accuracy: {metrics['accuracy']:.3%}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
