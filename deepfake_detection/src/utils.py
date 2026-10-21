"""
Utilities for Deepfake Detection Training Pipeline.
Includes:
- Reproducibility & Seed setting
- Device detection & CUDA status
- Temperature Scaling for probability calibration
- Evaluation visualization (Loss, Accuracy, Confusion Matrix, ROC curves)
"""

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    sns = None
    HAS_SEABORN = False

from sklearn.metrics import auc, confusion_matrix, roc_curve
import torch
import torch.nn as nn
import torch.optim as optim


def set_seed(seed: int = 42) -> None:
    """Set random seed across all libraries for deterministic reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[Setup] Reproducibility random seed set to: {seed}")


def get_device() -> torch.device:
    """Detect available compute device and log system GPU capabilities."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dev_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[Device] Using GPU: {dev_name} ({vram_gb:.2f} GB VRAM)")
    else:
        device = torch.device("cpu")
        print("[Device] CUDA not available. Running on CPU.")
    return device


class TemperatureScaler(nn.Module):
    """
    Post-hoc temperature scaling calibration for deep learning probabilities.
    Learns a single scalar temperature T > 0 on validation set logits via NLL.
    Preserves top-1 classification accuracy while producing well-calibrated confidence scores.
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature parameter T."""
        return logits / self.temperature

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor, lr: float = 0.01, max_iter: int = 50) -> float:
        """
        Optimize temperature parameter on validation logits.
        
        Args:
            val_logits: Tensor of shape (N, num_classes)
            val_labels: Tensor of shape (N,)
            
        Returns:
            Optimal temperature value (float)
        """
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_step():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(val_logits), val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        t_val = float(self.temperature.item())
        print(f"[Calibration] Temperature scaling fitted: T = {t_val:.4f}")
        return t_val


def plot_training_history(history: Dict[str, List[float]], save_dir: Union[str, Path]) -> Tuple[Path, Path]:
    """
    Plot and save training vs validation loss and accuracy curves.
    """
    p = Path(save_dir)
    p.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    # 1. Loss Curve
    loss_path = p / "training_loss.png"
    plt.figure(figsize=(8, 6), dpi=150)
    plt.plot(epochs, history["train_loss"], "b-o", label="Training Loss", linewidth=2)
    plt.plot(epochs, history["val_loss"], "r--s", label="Validation Loss", linewidth=2)
    plt.title("EfficientNet-B0 Fine-Tuning: Loss vs Epochs", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Cross-Entropy Loss", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(loss_path)
    plt.close()

    # 2. Accuracy Curve
    acc_path = p / "training_accuracy.png"
    plt.figure(figsize=(8, 6), dpi=150)
    plt.plot(epochs, [a * 100 for a in history["train_acc"]], "b-o", label="Training Accuracy (%)", linewidth=2)
    plt.plot(epochs, [a * 100 for a in history["val_acc"]], "g--s", label="Validation Accuracy (%)", linewidth=2)
    plt.title("EfficientNet-B0 Fine-Tuning: Accuracy vs Epochs", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(acc_path)
    plt.close()

    print(f"[Visuals] Saved training loss plot to: {loss_path}")
    print(f"[Visuals] Saved training accuracy plot to: {acc_path}")
    return loss_path, acc_path


def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Union[str, Path],
) -> Path:
    """
    Plot normalized and absolute confusion matrix with TP, TN, FP, FN annotations.
    """
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    annotations = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    if HAS_SEABORN and sns is not None:
        sns.heatmap(
            cm,
            annot=annotations,
            fmt="",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=True,
            annot_kws={"size": 12, "weight": "bold"},
            ax=ax,
        )
    else:
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm[i, j] > cm.max() / 2.0 else "black"
                ax.text(j, i, annotations[i, j], ha="center", va="center", color=color, fontweight="bold")
    ax.set_title("Test Set Confusion Matrix", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Class", fontsize=12, labelpad=8)
    ax.set_ylabel("Ground Truth Class", fontsize=12, labelpad=8)
    plt.tight_layout()
    plt.savefig(p)
    plt.close(fig)
    print(f"[Visuals] Saved confusion matrix heatmap to: {p}")
    return p


def plot_roc_curve_chart(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: Union[str, Path],
) -> Tuple[Path, float]:
    """
    Plot Receiver Operating Characteristic (ROC) curve and compute AUC.
    """
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = float(auc(fpr, tpr))

    plt.figure(figsize=(7, 6), dpi=150)
    plt.plot(fpr, tpr, color="darkorange", lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.50)")
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12)
    plt.title("Test Set ROC Curve (Deepfake Detection)", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    plt.savefig(p)
    plt.close()
    print(f"[Visuals] Saved ROC curve to: {p} (AUC = {roc_auc:.4f})")
    return p, roc_auc
