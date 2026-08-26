from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from dataset_utils import save_summary, scan_dataset
from model import FruitClassifier


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_transforms(image_size: int):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.70, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.18),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize]
    )
    return train_transform, eval_transform


def make_loaders(data_root: Path, image_size: int, batch_size: int, workers: int):
    train_transform, eval_transform = make_transforms(image_size)
    train_ds = datasets.ImageFolder(data_root / "train", transform=train_transform)
    valid_ds = datasets.ImageFolder(data_root / "valid", transform=eval_transform)
    test_ds = datasets.ImageFolder(data_root / "test", transform=eval_transform)
    if train_ds.classes != valid_ds.classes or train_ds.classes != test_ds.classes:
        raise ValueError("Class folders must match across train/valid/test.")

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=pin
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin
    )
    return train_ds, valid_ds, test_ds, train_loader, valid_loader, test_loader


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
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.inference_mode()
def collect_outputs(model, loader, device):
    model.eval()
    logits_all, embeddings_all, labels_all = [], [], []
    for images, labels in loader:
        images = images.to(device)
        logits, embeddings = model(images, return_embedding=True)
        logits_all.append(logits.cpu())
        embeddings_all.append(F.normalize(embeddings, dim=1).cpu())
        labels_all.append(labels.cpu())
    return torch.cat(logits_all), torch.cat(embeddings_all), torch.cat(labels_all)


def calibrate_verification(
    model,
    train_loader,
    valid_loader,
    class_names: List[str],
    device,
) -> Dict:
    _, train_embeddings, train_labels = collect_outputs(model, train_loader, device)
    centroids = []
    for class_index in range(len(class_names)):
        class_embeddings = train_embeddings[train_labels == class_index]
        centroid = F.normalize(class_embeddings.mean(dim=0, keepdim=True), dim=1)[0]
        centroids.append(centroid)
    centroids_tensor = torch.stack(centroids)

    valid_logits, valid_embeddings, valid_labels = collect_outputs(model, valid_loader, device)
    valid_probabilities = torch.softmax(valid_logits, dim=1)
    predicted = valid_probabilities.argmax(dim=1)
    similarities = valid_embeddings @ centroids_tensor.T

    correct_mask = predicted == valid_labels
    correct_confidences = valid_probabilities.max(dim=1).values[correct_mask].numpy()
    if len(correct_confidences) == 0:
        raise RuntimeError("No correctly classified validation images; cannot calibrate verifier.")

    confidence_threshold = max(0.55, float(np.percentile(correct_confidences, 5) - 0.03))
    margins: List[float] = []
    similarity_thresholds: Dict[str, float] = {}

    for class_index, class_name in enumerate(class_names):
        mask = correct_mask & (valid_labels == class_index)
        class_sims = similarities[mask, class_index].numpy()
        if len(class_sims) == 0:
            similarity_thresholds[class_name] = 0.35
        else:
            similarity_thresholds[class_name] = max(
                0.30, float(np.percentile(class_sims, 5) - 0.02)
            )

        rows = similarities[mask]
        for row in rows:
            top_two = torch.topk(row, k=2).values
            margins.append(float(top_two[0] - top_two[1]))

    margin_threshold = max(0.015, float(np.percentile(margins, 5) - 0.01)) if margins else 0.03
    return {
        "centroids": centroids_tensor.tolist(),
        "confidence_threshold": confidence_threshold,
        "similarity_thresholds": similarity_thresholds,
        "margin_threshold": margin_threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and calibrate the fruit verification model.")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/fruit_classifier.pt"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    seed_everything()
    summary = scan_dataset(args.data, verify_images=False)
    save_summary(summary, Path("artifacts/dataset_summary.json"))
    data_root = Path(summary["dataset_root"])

    train_ds, _, _, train_loader, valid_loader, test_loader = make_loaders(
        data_root, args.image_size, args.batch_size, args.workers
    )
    class_names = train_ds.classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Classes: {class_names}")

    model = FruitClassifier(len(class_names), pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_state = None
    best_val_accuracy = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        if epoch <= args.warmup_epochs:
            model.freeze_backbone()
            learning_rate = 1e-3
        else:
            model.unfreeze_last_blocks(blocks=4)
            learning_rate = 2e-4

        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=learning_rate,
            weight_decay=1e-4,
        )
        train_loss, train_accuracy = run_epoch(
            model, train_loader, device, criterion, optimizer
        )
        valid_loss, valid_accuracy = run_epoch(
            model, valid_loader, device, criterion, optimizer=None
        )
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_accuracy:.3%} | "
            f"valid loss {valid_loss:.4f} acc {valid_accuracy:.3%}"
        )

        if valid_accuracy > best_val_accuracy:
            best_val_accuracy = valid_accuracy
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping triggered.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a model checkpoint.")
    model.load_state_dict(best_state)

    test_logits, _, test_labels = collect_outputs(model, test_loader, device)
    test_predictions = test_logits.argmax(dim=1)
    test_accuracy = float((test_predictions == test_labels).float().mean().item())
    report = classification_report(
        test_labels.numpy(),
        test_predictions.numpy(),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(test_labels.numpy(), test_predictions.numpy()).tolist()

    verification = calibrate_verification(
        model, train_loader, valid_loader, class_names, device
    )
    checkpoint = {
        "model_state": model.state_dict(),
        "class_names": class_names,
        "image_size": args.image_size,
        "verification": verification,
        "metadata": {
            "best_validation_accuracy": best_val_accuracy,
            "test_accuracy": test_accuracy,
            "classification_report": report,
            "confusion_matrix": matrix,
            "dataset_summary": summary,
            "architecture": "MobileNetV3-Small transfer learning + centroid verifier",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    Path("artifacts/metrics.json").write_text(
        json.dumps(checkpoint["metadata"], indent=2), encoding="utf-8"
    )
    print(f"Test accuracy: {test_accuracy:.3%}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
