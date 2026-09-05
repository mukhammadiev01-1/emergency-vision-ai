#!/usr/bin/env python3
"""Second-Stage Action Recognition Fine-Tuning on Production-Aligned Person Crops.

Trains/fine-tunes the R3D-18 video action recognition classifier using the exact
spatiotemporal representation observed at production inference:
    URFD Video -> YOLO11n + ByteTrack -> Person Crop (5% pad) -> 16-Frame Tube -> Normalized Tensor.

Preserves strict sequence-level isolation (Seed=42) across train, val, and test splits.
Initializes weights from the existing canonical R3D-18 checkpoint and outputs a new
specialized checkpoint for comparative evaluation:
    models/action_recognition/r3d18_urfd_person_crops.pth
"""
import argparse
from datetime import datetime, timezone
import json
import logging
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18

# Ensure workspace root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from apps.worker.app.datasets.person_crop_dataset import (
    PersonCropDataset,
    build_person_crop_splits,
)
from apps.worker.app.datasets.urfd_dataset import (
    CLASS_NAMES,
    LABEL_FALL,
    LABEL_NORMAL,
)
from apps.worker.app.models.yolo import YOLOModelWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_person_crops")


def resolve_device(requested_device: str) -> str:
    """Resolve compute accelerator with safe fallbacks."""
    req = requested_device.lower()
    if req == "cuda" and torch.cuda.is_available():
        return "cuda"
    elif req == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    elif req in ("cuda", "mps"):
        logger.warning("Requested device '%s' unavailable; falling back to CPU.", req)
        return "cpu"
    return "cpu"


def calculate_metrics(labels: np.ndarray, predictions: np.ndarray, num_classes: int = 2) -> Dict[str, Any]:
    """Calculate accuracy, precision, recall, F1, and confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for l, p in zip(labels, predictions):
        if 0 <= l < num_classes and 0 <= p < num_classes:
            cm[l, p] += 1

    total = np.sum(cm)
    correct = np.trace(cm)
    accuracy = float(correct / total) if total > 0 else 0.0

    precisions = []
    recalls = []
    f1s = []

    for c in range(num_classes):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    macro_f1 = float(np.mean(f1s))
    tn = cm[LABEL_NORMAL, LABEL_NORMAL]
    fp = cm[LABEL_NORMAL, LABEL_FALL]
    normal_fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "fall_precision": precisions[LABEL_FALL],
        "fall_recall": recalls[LABEL_FALL],
        "fall_f1": f1s[LABEL_FALL],
        "normal_precision": precisions[LABEL_NORMAL],
        "normal_recall": recalls[LABEL_NORMAL],
        "normal_f1": f1s[LABEL_NORMAL],
        "normal_fpr": normal_fpr,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
    }


def load_model_from_checkpoint(
    checkpoint_path: str,
    num_classes: int = 2,
    device: str = "cpu",
) -> nn.Module:
    """Instantiate R3D-18 and load weights from existing canonical checkpoint."""
    model = r3d_18()
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if os.path.exists(checkpoint_path):
        logger.info("Loading initial weights from canonical checkpoint: %s", checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        # Handle prefix if present
        clean_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(clean_state, strict=False)
        logger.info("Successfully transferred initial weights.")
    else:
        logger.warning("Checkpoint %s not found; initializing with default torchvision weights.", checkpoint_path)

    return model.to(device)


def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str = "cpu",
) -> Tuple[float, Dict[str, Any]]:
    """Evaluate model on a DataLoader split and return average loss and metrics."""
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for videos, labels in loader:
            videos = videos.to(device)
            labels = labels.to(device)

            outputs = model(videos)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(all_labels) if all_labels else 0.0
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
    return avg_loss, metrics


def train_person_crops_pipeline(
    dataset_root: str = "data/urfd",
    base_checkpoint: str = "models/action_recognition/r3d18_urfd_best.pth",
    yolo_model_path: str = "models/detection/yolo11n.pt",
    output_dir: str = "models/action_recognition",
    checkpoint_name: str = "r3d18_urfd_person_crops.pth",
    epochs: int = 15,
    batch_size: int = 8,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    device_str: str = "cuda",
    seed: int = 42,
    cache_dir: str = ".cache/person_crops",
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    """Execute second-stage fine-tuning on person crops and evaluate on test split."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = resolve_device(device_str)

    os.makedirs(output_dir, exist_ok=True)
    target_checkpoint = os.path.join(output_dir, checkpoint_name)

    print("=" * 85)
    print("EMERGENCY VISION AI — SECOND-STAGE PERSON-CROP ACTION TRAINING")
    print("=" * 85)
    print(f"Accelerator Device:        {device.upper()}")
    print(f"Base Checkpoint:           {base_checkpoint} (Exists: {os.path.exists(base_checkpoint)})")
    print(f"YOLO Detector:             {yolo_model_path}")
    print(f"Target Checkpoint:         {target_checkpoint}")
    print(f"Epochs:                    {epochs}")
    print(f"Batch Size:                {batch_size}")
    print(f"Learning Rate:             {lr}")
    print(f"Random Seed:               {seed} (Strict Sequence Isolation)")
    print("=" * 85)

    # 1. Build Production Person-Crop Dataset
    logger.info("Initializing YOLO model wrapper for tube extraction...")
    yolo_wrapper = YOLOModelWrapper(model_path=yolo_model_path, device=device)
    yolo_wrapper._ensure_loaded()

    train_ds, val_ds, test_ds = build_person_crop_splits(
        dataset_root=dataset_root,
        yolo_wrapper=yolo_wrapper,
        cache_dir=cache_dir,
        stride=4,
        seed=seed,
        force_rebuild=force_rebuild,
    )

    print(f"Extracted Person Tubes:")
    print(f"  - Train Split: {len(train_ds)} tubes (FALL: {sum(1 for s in train_ds.samples if s.label == 1)}, NORMAL: {sum(1 for s in train_ds.samples if s.label == 0)})")
    print(f"  - Val Split:   {len(val_ds)} tubes (FALL: {sum(1 for s in val_ds.samples if s.label == 1)}, NORMAL: {sum(1 for s in val_ds.samples if s.label == 0)})")
    print(f"  - Test Split:  {len(test_ds)} tubes (FALL: {sum(1 for s in test_ds.samples if s.label == 1)}, NORMAL: {sum(1 for s in test_ds.samples if s.label == 0)})")
    print("=" * 85)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 2. Build and Configure Model
    model = load_model_from_checkpoint(base_checkpoint, num_classes=2, device=device)

    # Unfreeze layer3, layer4, and fc head for second-stage fine-tuning
    for p in model.parameters():
        p.requires_grad = False
    for p in model.layer3.parameters():
        p.requires_grad = True
    for p in model.layer4.parameters():
        p.requires_grad = True
    for p in model.fc.parameters():
        p.requires_grad = True

    # Differential learning rates
    optimizer = torch.optim.AdamW(
        [
            {"params": model.layer3.parameters(), "lr": lr * 0.2},
            {"params": model.layer4.parameters(), "lr": lr * 0.5},
            {"params": model.fc.parameters(), "lr": lr},
        ],
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # 3. Training Loop
    best_val_f1 = -1.0
    best_val_metrics = {}
    history = []

    print("\nStarting second-stage person-crop fine-tuning...")
    print("-" * 95)
    print(f"{'Epoch':<6} | {'Train Loss':<12} | {'Val Loss':<10} | {'Val Acc':<10} | {'Fall Prec':<10} | {'Fall Rec':<10} | {'Fall F1':<10}")
    print("-" * 95)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_samples = 0

        for videos, labels in train_loader:
            videos = videos.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            n_samples += labels.size(0)

        scheduler.step()
        avg_train_loss = train_loss / n_samples if n_samples > 0 else 0.0
        val_loss, val_metrics = evaluate_split(model, val_loader, criterion, device=device)

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_metrics": val_metrics,
        })

        print(
            f"{epoch:<6} | {avg_train_loss:<12.4f} | {val_loss:<10.4f} | "
            f"{val_metrics['accuracy'] * 100:<9.2f}% | "
            f"{val_metrics['fall_precision'] * 100:<9.2f}% | "
            f"{val_metrics['fall_recall'] * 100:<9.2f}% | "
            f"{val_metrics['fall_f1'] * 100:<9.2f}%"
        )

        if val_metrics["fall_f1"] >= best_val_f1:
            best_val_f1 = val_metrics["fall_f1"]
            best_val_metrics = val_metrics
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_metrics": val_metrics,
                    "arch": "r3d_18",
                    "domain": "per_person_crops",
                    "training_timestamp": datetime.now(timezone.utc).isoformat(),
                },
                target_checkpoint,
            )

    print("-" * 95)
    print(f"Training completed. Best checkpoint saved to: {target_checkpoint}")

    # 4. Final Evaluation on Held-Out Test Split
    print("\n" + "=" * 85)
    print("EVALUATING BEST CHECKPOINT ON HELD-OUT TEST SPLIT (Seed=42)")
    print("=" * 85)

    best_ckpt = torch.load(target_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_loss, test_metrics = evaluate_split(model, test_loader, criterion, device=device)

    print(f"Test Loss:                   {test_loss:.4f}")
    print(f"Test Accuracy:               {test_metrics['accuracy'] * 100:.2f}%")
    print(f"FALL Precision:              {test_metrics['fall_precision'] * 100:.2f}%")
    print(f"FALL Recall (Sensitivity):   {test_metrics['fall_recall'] * 100:.2f}%")
    print(f"FALL F1-Score:               {test_metrics['fall_f1']:.4f}")
    print(f"NORMAL False Positive Rate:  {test_metrics['normal_fpr'] * 100:.2f}%")
    print(f"Macro F1-Score:              {test_metrics['macro_f1']:.4f}")
    print("\nTest Confusion Matrix:")
    cm = test_metrics["confusion_matrix"]
    print("+" + "-" * 22 + "+" + "-" * 16 + "+" + "-" * 16 + "+")
    print(f"| {'Actual \\ Predicted':<20} | {'Pred NORMAL':<14} | {'Pred FALL':<14} |")
    print("+" + "-" * 22 + "+" + "-" * 16 + "+" + "-" * 16 + "+")
    print(f"| {'Actual NORMAL':<20} | {cm[0, 0]:<14} | {cm[0, 1]:<14} |")
    print(f"| {'Actual FALL':<20} | {cm[1, 0]:<14} | {cm[1, 1]:<14} |")
    print("+" + "-" * 22 + "+" + "-" * 16 + "+" + "-" * 16 + "+")

    summary = {
        "best_checkpoint": target_checkpoint,
        "val_metrics": {k: float(v) if not isinstance(v, np.ndarray) else v.tolist() for k, v in best_val_metrics.items()},
        "test_metrics": {k: float(v) if not isinstance(v, np.ndarray) else v.tolist() for k, v in test_metrics.items()},
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train R3D-18 Action Model on Production Person Crops")
    parser.add_argument("--dataset-root", type=str, default="data/urfd", help="Path to URFD dataset")
    parser.add_argument("--base-checkpoint", type=str, default="models/action_recognition/r3d18_urfd_best.pth", help="Path to initial weights")
    parser.add_argument("--yolo-model", type=str, default="models/detection/yolo11n.pt", help="Path to YOLO11n weights")
    parser.add_argument("--output-dir", type=str, default="models/action_recognition", help="Checkpoint output directory")
    parser.add_argument("--checkpoint-name", type=str, default="r3d18_urfd_person_crops.pth", help="Checkpoint filename")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Accelerator (cuda, mps, cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sequence splits")
    parser.add_argument("--cache-dir", type=str, default=".cache/person_crops", help="Cache directory for tubes")
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuilding tube cache")

    args = parser.parse_args()

    train_person_crops_pipeline(
        dataset_root=args.dataset_root,
        base_checkpoint=args.base_checkpoint,
        yolo_model_path=args.yolo_model,
        output_dir=args.output_dir,
        checkpoint_name=args.checkpoint_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_str=args.device,
        seed=args.seed,
        cache_dir=args.cache_dir,
        force_rebuild=args.force_rebuild,
    )


if __name__ == "__main__":
    main()
