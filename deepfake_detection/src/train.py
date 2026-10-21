"""
Two-Stage Fine-Tuning Pipeline for EfficientNet-B0 Deepfake Detection.

Features:
- Stage 1: Frozen backbone (train classifier head only with AdamW, lr=1e-3)
- Stage 2: Unfreeze top convolutional blocks (fine-tune with AdamW, lr=1e-5)
- Mixed Precision (AMP) for fast GPU execution
- CosineAnnealingLR scheduling
- Early stopping with patience
- Best model checkpointing based on Validation F1 and Loss
- Post-training probability calibration via Temperature Scaling
- Training history plotting (Loss and Accuracy curves)
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import EfficientNetDeepfakeClassifier
from .utils import TemperatureScaler, get_device, plot_training_history, set_seed


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Tuple[float, float]:
    """Execute one training epoch with optional mixed-precision."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    use_amp = scaler is not None and device.type == "cuda"

    for images, labels in tqdm(dataloader, desc="Training Batch", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += images.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, torch.Tensor, torch.Tensor]:
    """Evaluate model on validation/test dataloader."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds: list = []
    all_targets: list = []
    all_logits: list = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)

            correct += torch.sum(preds == labels.data).item()
            total += images.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_logits.append(outputs.cpu())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    epoch_f1 = float(f1_score(all_targets, all_preds, average="binary", zero_division=0))
    cat_logits = torch.cat(all_logits, dim=0)
    cat_targets = torch.tensor(all_targets, dtype=torch.long)

    return epoch_loss, epoch_acc, epoch_f1, cat_logits, cat_targets


def train_deepfake_classifier(
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: Optional[torch.Tensor] = None,
    stage1_epochs: int = 4,
    stage2_epochs: int = 8,
    stage1_lr: float = 1e-3,
    stage2_lr: float = 1e-5,
    patience: int = 5,
    save_dir: Union[str, Path] = "models",
    results_dir: Union[str, Path] = "results",
    seed: int = 42,
) -> Tuple[EfficientNetDeepfakeClassifier, float, Dict[str, List[float]]]:
    """
    Complete two-stage fine-tuning execution.
    """
    set_seed(seed)
    device = get_device()

    save_path = Path(save_dir) / "best_model.pth"
    results_path = Path(results_dir)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.mkdir(parents=True, exist_ok=True)

    # Initialize model
    model = EfficientNetDeepfakeClassifier(pretrained=True)
    model.to(device)

    # Loss function with class weight scaling
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
    }

    best_val_loss = float("inf")
    best_val_f1 = 0.0
    epochs_no_improve = 0

    total_epochs = stage1_epochs + stage2_epochs

    print("\n" + "=" * 70)
    print(f"  STARTING TWO-STAGE FINE-TUNING ({total_epochs} Total Epochs)")
    print(f"  • Stage 1 (Frozen Backbone, Train Head): {stage1_epochs} epochs @ lr={stage1_lr}")
    print(f"  • Stage 2 (Unfreeze Top Blocks 6 & 7):   {stage2_epochs} epochs @ lr={stage2_lr}")
    print(f"  • Device: {device} | Early Stopping Patience: {patience}")
    print("=" * 70 + "\n")

    current_epoch = 0

    # -------------------------------------------------------------------------
    # STAGE 1: TRAIN CLASSIFIER HEAD (Backbone Frozen)
    # -------------------------------------------------------------------------
    model.freeze_backbone()
    optimizer = optim.AdamW(model.classifier_head.parameters(), lr=stage1_lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, stage1_epochs))

    for epoch in range(stage1_epochs):
        current_epoch += 1
        start_t = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc, val_f1, _, _ = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - start_t

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(
            f"Stage 1 | Epoch [{current_epoch:02d}/{total_epochs:02d}] ({elapsed:.1f}s) "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_f1 = val_f1
            epochs_no_improve = 0
            model.save_checkpoint(save_path, current_epoch, val_loss, val_acc, val_f1)

    # -------------------------------------------------------------------------
    # STAGE 2: FINE-TUNE UNFROZEN TOP BLOCKS
    # -------------------------------------------------------------------------
    model.unfreeze_top_blocks(num_blocks=2)
    # Differential learning rates: smaller for backbone, slightly higher for head
    trainable_params = [
        {"params": [p for n, p in model.named_parameters() if "classifier_head" in n and p.requires_grad], "lr": stage2_lr * 2.0},
        {"params": [p for n, p in model.named_parameters() if "classifier_head" not in n and p.requires_grad], "lr": stage2_lr},
    ]
    optimizer = optim.AdamW(trainable_params, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, stage2_epochs), eta_min=1e-7)

    for epoch in range(stage2_epochs):
        current_epoch += 1
        start_t = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc, val_f1, _, _ = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - start_t

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(
            f"Stage 2 | Epoch [{current_epoch:02d}/{total_epochs:02d}] ({elapsed:.1f}s) "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_f1 = val_f1
            epochs_no_improve = 0
            model.save_checkpoint(save_path, current_epoch, val_loss, val_acc, val_f1)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n[Early Stopping] No validation loss improvement for {patience} consecutive epochs. Stopping.")
                break

    # -------------------------------------------------------------------------
    # POST-TRAINING: TEMPERATURE SCALING CALIBRATION
    # -------------------------------------------------------------------------
    print("\n[Post-Training] Fitting Temperature Scaling on Validation Set...")
    # Load best checkpoint before fitting calibration
    model.load_checkpoint(save_path)
    _, _, _, val_logits, val_targets = evaluate_epoch(model, val_loader, criterion, device)

    scaler_module = TemperatureScaler()
    best_temperature = scaler_module.fit(val_logits, val_targets)

    # Re-save best model with calibrated temperature
    model.save_checkpoint(
        save_path,
        epoch=current_epoch,
        val_loss=best_val_loss,
        val_acc=history["val_acc"][-1],
        val_f1=best_val_f1,
        temperature=best_temperature,
    )

    # Plot loss and accuracy curves
    plot_training_history(history, results_path)

    return model, best_temperature, history
