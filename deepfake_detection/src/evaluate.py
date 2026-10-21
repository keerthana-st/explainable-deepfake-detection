"""
Test Evaluation Suite for Deepfake Detection Model.

Computes:
- Confusion Matrix (TP, TN, FP, FN)
- Accuracy, Precision, Recall (Fake Sensitivity), Specificity, False Negative Rate (FNR), F1-Score, ROC-AUC
- Generates publication-ready Confusion Matrix heatmap and ROC curve plots
- Explains why accuracy alone is insufficient in security/forensic contexts
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import EfficientNetDeepfakeClassifier
from .utils import get_device, plot_confusion_matrix_heatmap, plot_roc_curve_chart


def evaluate_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    temperature: float = 1.0,
    results_dir: Union[str, Path] = "results",
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Run comprehensive evaluation strictly on unseen test set.
    """
    if device is None:
        device = get_device()
    model.to(device)
    model.eval()

    res_dir = Path(results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)

    all_preds: List[int] = []
    all_targets: List[int] = []
    all_fake_probs: List[float] = []

    fake_idx = getattr(model, "fake_idx", 1)
    real_idx = getattr(model, "real_idx", 0)

    print("\n[Evaluation] Running inference on unseen test set...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Test Batches"):
            images = images.to(device, non_blocking=True)
            logits = model(images)

            # Apply temperature scaling calibration
            calibrated_logits = logits / max(1e-4, temperature)
            probs = torch.softmax(calibrated_logits, dim=1)

            _, preds = torch.max(probs, 1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.numpy().tolist())
            all_fake_probs.extend(probs[:, fake_idx].cpu().numpy().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_scores = np.array(all_fake_probs)

    # -------------------------------------------------------------------------
    # COMPUTE METRICS
    # -------------------------------------------------------------------------
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, pos_label=fake_idx, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=fake_idx, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=fake_idx, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_scores))

    # Confusion Matrix breakdown: class 0 is Real, class 1 is Fake
    # cm[row, col] -> row = true, col = pred
    cm = confusion_matrix(y_true, y_pred, labels=[real_idx, fake_idx])
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # -------------------------------------------------------------------------
    # GENERATE PLOTS
    # -------------------------------------------------------------------------
    cm_path = res_dir / "confusion_matrix.png"
    plot_confusion_matrix_heatmap(cm, class_names=["REAL", "FAKE"], save_path=cm_path)

    roc_path = res_dir / "roc_curve.png"
    plot_roc_curve_chart(y_true, y_scores, save_path=roc_path)

    # -------------------------------------------------------------------------
    # PRINT FORENSIC REPORT
    # -------------------------------------------------------------------------
    bar = "=" * 70
    print("\n" + bar)
    print("                 FINAL UNSEEN TEST EVALUATION REPORT")
    print(bar)
    print(f"Total Test Samples:      {len(y_true):,}")
    print(f"  • Real Samples:        {np.sum(y_true == real_idx):,}")
    print(f"  • Fake Samples:        {np.sum(y_true == fake_idx):,}")
    print("-" * 70)
    print(f"Classification Accuracy: {acc * 100:.2f}%")
    print(f"ROC-AUC Score:           {roc_auc:.4f}")
    print(f"Fake Precision:          {prec * 100:.2f}%  (Purity of flagged fakes)")
    print(f"Fake Recall (Sensitivity):{rec * 100:.2f}%  (Ability to catch fakes)")
    print(f"F1-Score:                {f1:.4f}")
    print(f"Specificity (Real TNR):  {specificity * 100:.2f}%  (Protection of innocent real photos)")
    print(f"False Negative Rate (FNR):{fnr * 100:.2f}%  (CRITICAL: Undetected deepfakes)")
    print(f"False Positive Rate (FPR):{fpr * 100:.2f}%  (False alarms on authentic photos)")
    print("-" * 70)
    print("Confusion Matrix Counts:")
    print(f"  • True Positives  (Fake correctly predicted): {tp:,}")
    print(f"  • True Negatives  (Real correctly predicted): {tn:,}")
    print(f"  • False Positives (Real misflagged as Fake):  {fp:,}")
    print(f"  • False Negatives (Fake missed as Real):      {fn:,}")
    print(bar)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "specificity": specificity,
        "false_negative_rate": fnr,
        "false_positive_rate": fpr,
        "confusion_matrix": {
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
        },
        "plots": {
            "confusion_matrix": str(cm_path),
            "roc_curve": str(roc_path),
        },
    }
