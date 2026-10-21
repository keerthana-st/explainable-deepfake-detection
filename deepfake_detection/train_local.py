"""
Local Physical Machine Fine-Tuning Script for EfficientNet-B0 Deepfake Classifier.
Tailored for local AMD Ryzen 5 CPU multi-core execution using D:/archive.zip.

Features:
- Direct extraction of a clean, balanced subset from D:/archive.zip (Train, Val, Test).
- Multi-threaded CPU execution (torch.set_num_threads).
- Two-Stage Transfer Learning (Stage 1: Frozen backbone; Stage 2: Unfreeze blocks 6 & 7).
- Post-hoc Temperature Scaling calibration.
- Full test evaluation: Accuracy, Precision, Recall, Specificity, F1, ROC-AUC, Confusion Matrix, and ROC curve.
- Automatic deployment to models/ for live multi-agent system inference.
"""

import argparse
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from deepfake_detection.src.dataset import build_dataloaders
from deepfake_detection.src.evaluate import evaluate_test_set
from deepfake_detection.src.model import EfficientNetDeepfakeClassifier
from deepfake_detection.src.train import train_deepfake_classifier
from deepfake_detection.src.utils import TemperatureScaler, set_seed


def extract_archive_subset(
    zip_path: str = "D:/archive.zip",
    dest_dir: str = "deepfake_detection/dataset",
    train_per_class: int = 1200,
    val_per_class: int = 250,
    test_per_class: int = 250,
) -> Path:
    """
    Extracts a balanced, clean subset directly from D:/archive.zip into train/val/test splits.
    """
    dest = Path(dest_dir).resolve()
    train_real_dir = dest / "train" / "REAL"
    train_fake_dir = dest / "train" / "FAKE"
    val_real_dir = dest / "val" / "REAL"
    val_fake_dir = dest / "val" / "FAKE"
    test_real_dir = dest / "test" / "REAL"
    test_fake_dir = dest / "test" / "FAKE"

    # Check if already extracted
    if (
        train_real_dir.exists()
        and len(list(train_real_dir.glob("*.jpg"))) >= train_per_class
        and train_fake_dir.exists()
        and len(list(train_fake_dir.glob("*.jpg"))) >= train_per_class
    ):
        print(f"[Dataset] Existing dataset in {dest} already has required samples. Skipping extraction.")
        return dest

    print(f"\n============================================================")
    print(f"       EXTRACTING BALANCED DATASET FROM {zip_path}")
    print(f"============================================================")
    print(f"  • Destination: {dest}")
    print(f"  • Train count: {train_per_class} Real + {train_per_class} Fake ({train_per_class * 2} total)")
    print(f"  • Val count:   {val_per_class} Real + {val_per_class} Fake ({val_per_class * 2} total)")
    print(f"  • Test count:  {test_per_class} Real + {test_per_class} Fake ({test_per_class * 2} total)")

    for d in [train_real_dir, train_fake_dir, val_real_dir, val_fake_dir, test_real_dir, test_fake_dir]:
        d.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()

        # Gather source candidates
        raw_train_real = [n for n in names if n.startswith("train/REAL/") and n.lower().endswith((".jpg", ".png"))]
        raw_train_fake = [n for n in names if n.startswith("train/FAKE/") and n.lower().endswith((".jpg", ".png"))]
        raw_test_real = [n for n in names if n.startswith("test/REAL/") and n.lower().endswith((".jpg", ".png"))]
        raw_test_fake = [n for n in names if n.startswith("test/FAKE/") and n.lower().endswith((".jpg", ".png"))]

        print(f"[Archive Scan] Found {len(raw_train_real)} train/REAL, {len(raw_train_fake)} train/FAKE in zip.")

        def extract_items(item_list, target_dir, max_count):
            count = 0
            for name in item_list:
                if count >= max_count:
                    break
                fname = Path(name).name
                out_path = target_dir / fname
                if not out_path.exists():
                    data = z.read(name)
                    # Verify image validity before writing
                    if len(data) > 100:
                        out_path.write_bytes(data)
                        count += 1
                else:
                    count += 1
            return count

        # Extract Train
        n_tr_r = extract_items(raw_train_real[:train_per_class], train_real_dir, train_per_class)
        n_tr_f = extract_items(raw_train_fake[:train_per_class], train_fake_dir, train_per_class)

        # Extract Val (from remaining train or test)
        n_val_r = extract_items(raw_train_real[train_per_class : train_per_class + val_per_class], val_real_dir, val_per_class)
        n_val_f = extract_items(raw_train_fake[train_per_class : train_per_class + val_per_class], val_fake_dir, val_per_class)

        # Extract Test
        n_ts_r = extract_items(raw_test_real[:test_per_class], test_real_dir, test_per_class)
        n_ts_f = extract_items(raw_test_fake[:test_per_class], test_fake_dir, test_per_class)

    elapsed = time.time() - t0
    print(f"[Extraction Done] Completed in {elapsed:.2f}s:")
    print(f"  • Train: {n_tr_r} Real, {n_tr_f} Fake")
    print(f"  • Val:   {n_val_r} Real, {n_val_f} Fake")
    print(f"  • Test:  {n_ts_r} Real, {n_ts_f} Fake")
    return dest


def run_local_training(
    dataset_dir: Path,
    stage1_epochs: int = 2,
    stage2_epochs: int = 3,
    batch_size: int = 32,
    cpu_threads: int = 6,
):
    """
    Executes the two-stage transfer learning pipeline optimized for local physical CPU.
    """
    torch.set_num_threads(cpu_threads)
    print(f"[Hardware Setup] PyTorch CPU threads configured to: {torch.get_num_threads()} (AMD Ryzen multi-core active)")

    models_dir = PROJECT_ROOT / "deepfake_detection" / "models"
    results_dir = PROJECT_ROOT / "deepfake_detection" / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build DataLoaders (num_workers=0 for fast Windows CPU memory mapping)
    train_loader, val_loader, test_loader, class_weights = build_dataloaders(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        num_workers=0,
    )

    # 2. Two-Stage Fine-Tuning
    model, best_val_f1, history = train_deepfake_classifier(
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        stage1_epochs=stage1_epochs,
        stage2_epochs=stage2_epochs,
        stage1_lr=1e-3,
        stage2_lr=1e-5,
        patience=4,
        save_dir=models_dir,
        results_dir=results_dir,
    )

    # 4. Temperature Scaling Calibration (Optimized & stored during training)
    best_ckpt_path = models_dir / "best_model.pth"
    ckpt = torch.load(best_ckpt_path, map_location="cpu")
    calibrated_temp = float(ckpt.get("temperature", 1.0))
    print(f"\n[Calibration] Loaded post-hoc calibrated temperature: T = {calibrated_temp:.4f}")

    # 5. Evaluate on Unseen Test Set
    print("\n============================================================")
    print("             TEST SET FORENSIC EVALUATION")
    print("============================================================")
    metrics = evaluate_test_set(
        model=model,
        test_loader=test_loader,
        temperature=calibrated_temp,
        results_dir=results_dir,
    )

    # 6. Deploy to Multi-Agent System
    system_models_dir = PROJECT_ROOT / "models"
    system_models_dir.mkdir(parents=True, exist_ok=True)

    target_pt = system_models_dir / "classifier_deepfake_weights.pt"
    target_pth = system_models_dir / "efficientnet_deepfake.pth"

    shutil.copy2(best_ckpt_path, target_pt)
    shutil.copy2(best_ckpt_path, target_pth)

    print("\n============================================================")
    print("      DEPLOYMENT TO MULTI-AGENT SYSTEM COMPLETE")
    print("============================================================")
    print(f"  ✓ Fine-tuned checkpoint saved to: {best_ckpt_path}")
    print(f"  ✓ Live system weights updated at: {target_pt}")
    print(f"  ✓ Live system weights updated at: {target_pth}")
    print(f"  ✓ Test set Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"  ✓ Test set F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  ✓ Test set ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"  ✓ Test set Specificity: {metrics['specificity']*100:.2f}% (Real Image Correct Rate)")
    print(f"  ✓ Calibrated Temp:    {calibrated_temp:.4f}")
    print("============================================================\n")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Local Physical Machine Deepfake Fine-Tuning")
    parser.add_argument("--archive", type=str, default="D:/archive.zip", help="Path to D:/archive.zip")
    parser.add_argument("--train-per-class", type=int, default=1000, help="Number of Real and Fake images for training")
    parser.add_argument("--val-per-class", type=int, default=200, help="Number of Real and Fake images for validation")
    parser.add_argument("--test-per-class", type=int, default=200, help="Number of Real and Fake images for testing")
    parser.add_argument("--stage1-epochs", type=int, default=2, help="Stage 1 head warmup epochs")
    parser.add_argument("--stage2-epochs", type=int, default=3, help="Stage 2 fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for CPU processing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    set_seed(args.seed)

    # Step 1: Extract dataset
    dataset_dir = extract_archive_subset(
        zip_path=args.archive,
        dest_dir="deepfake_detection/dataset",
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
        test_per_class=args.test_per_class,
    )

    # Step 2: Run training and evaluation
    run_local_training(
        dataset_dir=dataset_dir,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
