#!/usr/bin/env python3
"""Emergency Action Recognition Evaluation Script.

Evaluates a saved R3D-18 model checkpoint on the held-out test split,
computing accuracy, per-class precision/recall, F1-scores, and confusion matrix.
"""
import argparse
import logging
import os
import sys

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18

from apps.worker.app.datasets.urfd_dataset import (
    SyntheticURFDDataset,
    create_urfd_splits,
    LABEL_NORMAL,
    LABEL_FALL,
    CLASS_NAMES,
)
from scripts.train_action_model import calculate_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate_action_model")


def print_evaluation_report(metrics: Dict[str, float], num_classes: int = 2) -> None:
    """Format and print an evaluation report with confusion matrix."""
    print("\n" + "=" * 60)
    print("      EMERGENCY ACTION RECOGNITION EVALUATION REPORT")
    print("=" * 60)
    print(f"Overall Test Accuracy:    {metrics['accuracy'] * 100:.2f}%\n")

    print(f"{'Class':<15} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 55)
    for c in range(num_classes):
        c_name = CLASS_NAMES.get(c, f"Class {c}")
        if c == LABEL_FALL:
            prec, rec, f1 = metrics["fall_precision"], metrics["fall_recall"], metrics["fall_f1"]
        else:
            prec, rec, f1 = metrics["normal_precision"], metrics["normal_recall"], metrics["normal_f1"]
        print(f"{c_name:<15} | {prec * 100:>8.2f}% | {rec * 100:>8.2f}% | {f1 * 100:>8.2f}%")

    print("-" * 55)
    print(f"{'Macro Average':<15} | {'-':<10} | {'-':<10} | {metrics['macro_f1'] * 100:>8.2f}%\n")

    print("Confusion Matrix:")
    print("                   Predicted NORMAL   Predicted FALL")
    cm = metrics["confusion_matrix"]
    print(f"Actual NORMAL:        {cm[0, 0]:>8d}         {cm[0, 1]:>8d}")
    print(f"Actual FALL:          {cm[1, 0]:>8d}         {cm[1, 1]:>8d}")
    print("=" * 60 + "\n")


def evaluate_checkpoint(
    checkpoint_path: str,
    dataset_root: Optional[str] = "data/urfd",
    batch_size: int = 4,
    device_str: str = "auto",
    seed: int = 42,
    smoke_test: bool = False,
) -> Dict[str, float]:
    """Load model checkpoint and evaluate on the test split."""
    if device_str == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = device_str

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Load checkpoint
    logger.info("Loading model checkpoint from %s on %s...", checkpoint_path, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    num_classes = checkpoint.get("num_classes", 2)

    model = r3d_18()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Load test split
    if smoke_test:
        test_ds = SyntheticURFDDataset(num_samples=8, mode="test", seed=seed)
    else:
        if not dataset_root or not os.path.exists(dataset_root):
            raise FileNotFoundError(
                f"Dataset root not found: {dataset_root}. Use `--smoke-test` to test without dataset."
            )
        _, _, test_ds = create_urfd_splits(dataset_root, seed=seed)

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for videos, labels in test_loader:
            videos = videos.to(device)
            outputs = model(videos)
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())

    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds), num_classes=num_classes)
    print_evaluation_report(metrics, num_classes=num_classes)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate R3D-18 Action Recognition Model")
    parser.add_argument("--checkpoint", default="models/action_recognition/r3d18_urfd_best.pth", help="Checkpoint path")
    parser.add_argument("--dataset-root", default="data/urfd", help="Path to URFD dataset directory")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--device", default="auto", help="Device (cpu, cuda, mps, auto)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--smoke-test", action="store_true", help="Run evaluation smoke test on synthetic data")
    args = parser.parse_args()

    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        device_str=args.device,
        seed=args.seed,
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    main()
