#!/usr/bin/env python3
"""Emergency Action Recognition Training Script.

Trains a ResNet3D-18 (R3D-18) video model on UR Fall Detection (URFD)
using two-stage transfer learning initialized from official Kinetics-400 pretrained weights.

Classes:
  0: NORMAL (Activities of Daily Living)
  1: FALL (Acute human fall / collapse)
"""
import argparse
import logging
import os
import sys

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18, R3D_18_Weights

from apps.worker.app.datasets.urfd_dataset import (
    SyntheticURFDDataset,
    create_urfd_splits,
    LABEL_NORMAL,
    LABEL_FALL,
    CLASS_NAMES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_action_model")


def calculate_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    num_classes: int = 2,
) -> Dict[str, float]:
    """Calculate accuracy, per-class precision, per-class recall, macro F1, and confusion matrix."""
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

    return {
        "accuracy": accuracy,
        "fall_precision": precisions[LABEL_FALL],
        "fall_recall": recalls[LABEL_FALL],
        "fall_f1": f1s[LABEL_FALL],
        "normal_precision": precisions[LABEL_NORMAL],
        "normal_recall": recalls[LABEL_NORMAL],
        "normal_f1": f1s[LABEL_NORMAL],
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
    }


def build_r3d18_model(
    num_classes: int = 2,
    pretrained: bool = True,
    device: str = "cpu",
) -> nn.Module:
    """Build R3D-18 model initialized with Kinetics-400 pretrained weights and customized head."""
    if pretrained:
        try:
            weights = R3D_18_Weights.KINETICS400_V1
            model = r3d_18(weights=weights)
            logger.info("Loaded official torchvision Kinetics-400 pretrained R3D-18 weights.")
        except Exception as exc:
            logger.warning("Could not download online weights (%s), falling back to uninitialized model.", exc)
            model = r3d_18()
    else:
        model = r3d_18()

    # Replace classifier head with custom binary/multi-class linear layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model.to(device)


def set_stage1_parameter_freezing(model: nn.Module) -> None:
    """Stage 1: Freeze all backbone layers, train only classification head."""
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    logger.info("Stage 1: Backbone frozen. Only classifier head is trainable.")


def set_stage2_parameter_freezing(model: nn.Module) -> None:
    """Stage 2: Unfreeze layer3 and layer4 for differential spatiotemporal fine-tuning."""
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layer3, layer4, and fc head
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True
    logger.info("Stage 2: Layer3, Layer4, and classifier head unfreezed for fine-tuning.")


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str = "cpu",
    num_classes: int = 2,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate model on a DataLoader split and return average loss and metrics dict."""
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

    avg_loss = total_loss / max(1, len(loader.dataset))
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds), num_classes=num_classes)
    return avg_loss, metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
) -> float:
    """Execute one training epoch over the dataset loader."""
    model.train()
    running_loss = 0.0

    for videos, labels in loader:
        videos = videos.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

    return running_loss / max(1, len(loader.dataset))


def compute_class_weights(dataset: torch.utils.data.Dataset, num_classes: int = 2) -> torch.Tensor:
    """Compute balanced inverse class weights from training dataset."""
    counts = np.zeros(num_classes, dtype=np.float32)
    for sample in getattr(dataset, "samples", []):
        counts[sample.label] += 1
    total = np.sum(counts)
    if np.any(counts == 0) or total == 0:
        return torch.ones(num_classes, dtype=torch.float32)
    weights = total / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def run_training_pipeline(
    dataset_root: Optional[str] = "data/urfd",
    output_dir: str = "models/action_recognition",
    checkpoint_name: str = "r3d18_urfd_best.pth",
    stage1_epochs: int = 5,
    stage2_epochs: int = 20,
    batch_size: int = 4,
    device_str: str = "auto",
    seed: int = 42,
    smoke_test: bool = False,
) -> str:
    """Main training orchestrator for URFD action recognition model."""
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, checkpoint_name)

    # Select device
    if device_str == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = device_str

    logger.info("Using compute device: %s", device)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Prepare datasets
    if smoke_test:
        logger.info("[SMOKE TEST] Initializing synthetic in-memory datasets.")
        train_ds = SyntheticURFDDataset(num_samples=8, mode="train", seed=seed)
        val_ds = SyntheticURFDDataset(num_samples=4, mode="val", seed=seed)
        test_ds = SyntheticURFDDataset(num_samples=4, mode="test", seed=seed)
        stage1_epochs = 1
        stage2_epochs = 1
    else:
        if not dataset_root or not os.path.exists(dataset_root):
            raise FileNotFoundError(
                f"Dataset root directory not found: {dataset_root}. "
                "Run `python3 scripts/download_urfd.py` first, or use `--smoke-test`."
            )
        train_ds, val_ds, test_ds = create_urfd_splits(dataset_root, seed=seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize model
    model = build_r3d18_model(num_classes=2, pretrained=(not smoke_test), device=device)

    # Class-weighted loss
    class_weights = compute_class_weights(train_ds, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    logger.info("Loss function: CrossEntropyLoss with class weights %s", class_weights.tolist())

    best_val_f1 = -1.0
    best_val_loss = float("inf")

    # =========================================================================
    # STAGE 1: Train Head Only (Backbone Frozen)
    # =========================================================================
    logger.info("=== Starting Stage 1: Classifier Head Warm-Up (%d Epochs) ===", stage1_epochs)
    set_stage1_parameter_freezing(model)
    optimizer_s1 = torch.optim.AdamW(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(1, stage1_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer_s1, device=device)
        val_loss, val_metrics = evaluate_model(model, val_loader, criterion, device=device)

        logger.info(
            "Stage 1 [Epoch %d/%d] Train Loss: %.4f | Val Loss: %.4f | Val Acc: %.3f | Fall Rec: %.3f | Macro F1: %.3f",
            epoch,
            stage1_epochs,
            train_loss,
            val_loss,
            val_metrics["accuracy"],
            val_metrics["fall_recall"],
            val_metrics["macro_f1"],
        )

        if val_metrics["macro_f1"] > best_val_f1 or (val_metrics["macro_f1"] == best_val_f1 and val_loss < best_val_loss):
            best_val_f1 = val_metrics["macro_f1"]
            best_val_loss = val_loss
            # Create serializable copy of metrics
            serializable_metrics = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in val_metrics.items()}
            torch.save(
                {
                    "epoch": epoch,
                    "stage": 1,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_metrics": serializable_metrics,
                    "num_classes": 2,
                    "class_names": CLASS_NAMES,
                },
                checkpoint_path,
            )
            logger.info("Saved new best model checkpoint to %s (Val F1: %.3f)", checkpoint_path, best_val_f1)

    # =========================================================================
    # STAGE 2: Differential Deep Fine-Tuning
    # =========================================================================
    logger.info("=== Starting Stage 2: Differential Fine-Tuning (%d Epochs) ===", stage2_epochs)
    set_stage2_parameter_freezing(model)

    # Differential parameter groups
    backbone_params = list(model.layer3.parameters()) + list(model.layer4.parameters())
    head_params = list(model.fc.parameters())

    optimizer_s2 = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": 1e-5},
            {"params": head_params, "lr": 1e-4},
        ],
        weight_decay=1e-4,
    )
    scheduler_s2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_s2, T_max=max(1, stage2_epochs), eta_min=1e-6)

    for epoch in range(1, stage2_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer_s2, device=device)
        scheduler_s2.step()
        val_loss, val_metrics = evaluate_model(model, val_loader, criterion, device=device)

        logger.info(
            "Stage 2 [Epoch %d/%d] Train Loss: %.4f | Val Loss: %.4f | Val Acc: %.3f | Fall Rec: %.3f | Macro F1: %.3f",
            epoch,
            stage2_epochs,
            train_loss,
            val_loss,
            val_metrics["accuracy"],
            val_metrics["fall_recall"],
            val_metrics["macro_f1"],
        )

        if val_metrics["macro_f1"] > best_val_f1 or (val_metrics["macro_f1"] == best_val_f1 and val_loss < best_val_loss):
            best_val_f1 = val_metrics["macro_f1"]
            best_val_loss = val_loss
            serializable_metrics = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in val_metrics.items()}
            torch.save(
                {
                    "epoch": epoch,
                    "stage": 2,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_metrics": serializable_metrics,
                    "num_classes": 2,
                    "class_names": CLASS_NAMES,
                },
                checkpoint_path,
            )
            logger.info("Saved new best model checkpoint to %s (Val F1: %.3f)", checkpoint_path, best_val_f1)

    logger.info("Training complete. Best checkpoint saved at: %s", checkpoint_path)
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train R3D-18 Action Recognition Model on URFD")
    parser.add_argument("--dataset-root", default="data/urfd", help="Path to URFD dataset directory")
    parser.add_argument("--output-dir", default="models/action_recognition", help="Path to save checkpoints")
    parser.add_argument("--checkpoint-name", default="r3d18_urfd_best.pth", help="Name of saved checkpoint")
    parser.add_argument("--stage1-epochs", type=int, default=5, help="Epochs for Stage 1 head warm-up")
    parser.add_argument("--stage2-epochs", type=int, default=20, help="Epochs for Stage 2 fine-tuning")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--device", default="auto", help="Device (cpu, cuda, mps, auto)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast smoke test with synthetic tensors")
    args = parser.parse_args()

    run_training_pipeline(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        checkpoint_name=args.checkpoint_name,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        batch_size=args.batch_size,
        device_str=args.device,
        seed=args.seed,
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    main()
