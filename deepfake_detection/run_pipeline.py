"""
Master End-to-End Orchestrator for Deepfake Detection Training Pipeline.

Usage:
    # 1. Clean, split, train, and evaluate from raw images:
    python deepfake_detection/run_pipeline.py --raw-data "path/to/raw_kaggle_folder"

    # 2. Or if data is already split:
    python deepfake_detection/run_pipeline.py --data-dir "deepfake_detection/dataset" --skip-split

    # 3. Download directly from Kaggle and execute:
    python deepfake_detection/run_pipeline.py --download-kaggle "xhlulu/140k-real-and-fake-faces"
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import (
    build_dataloaders,
    create_leakage_free_splits,
    download_kaggle_dataset,
    inspect_and_clean_dataset,
)
from src.evaluate import evaluate_test_set
from src.train import train_deepfake_classifier
from src.utils import set_seed


def main():
    parser = argparse.ArgumentParser(description="End-to-End EfficientNet-B0 Deepfake Training & Evaluation Pipeline")

    # Dataset arguments
    parser.add_argument("--download-kaggle", type=str, default=None, help="Kaggle dataset slug (e.g. 'xhlulu/140k-real-and-fake-faces')")
    parser.add_argument("--raw-data", type=str, default=None, help="Path to raw unsplit dataset directory")
    parser.add_argument("--data-dir", type=str, default="deepfake_detection/dataset", help="Destination path for organized train/val/test splits")
    parser.add_argument("--skip-split", action="store_true", help="Skip inspection and splitting if dataset is already organized into train/val/test")

    # Training arguments
    parser.add_argument("--stage1-epochs", type=int, default=4, help="Epochs for Stage 1 (frozen backbone)")
    parser.add_argument("--stage2-epochs", type=int, default=8, help="Epochs for Stage 2 (unfrozen top blocks)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training and evaluation")
    parser.add_argument("--stage1-lr", type=float, default=1e-3, help="Learning rate for Stage 1")
    parser.add_argument("--stage2-lr", type=float, default=1e-5, help="Learning rate for Stage 2")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # Output directories
    parser.add_argument("--save-dir", type=str, default="deepfake_detection/models", help="Directory to save best model checkpoint")
    parser.add_argument("--results-dir", type=str, default="deepfake_detection/results", help="Directory to save plots and evaluation tables")

    args = parser.parse_args()
    set_seed(args.seed)

    raw_path = Path(args.raw_data) if args.raw_data else None
    split_dir = Path(args.data_dir)

    # Step 1: Optional Kaggle download
    if args.download_kaggle:
        dl_dir = Path("kaggle_download")
        raw_path = download_kaggle_dataset(args.download_kaggle, dl_dir)

    # Step 2: Inspection, cleaning, and leakage-free splitting
    if not args.skip_split:
        if raw_path is None or not raw_path.exists():
            # Check if dataset directory already has train/val/test
            if (split_dir / "train").exists():
                print(f"[Info] Found existing organized split at: {split_dir}. Proceeding to dataloaders.")
            else:
                print(f"[Error] Please provide --raw-data pointing to dataset or --download-kaggle.")
                sys.exit(1)
        else:
            inspection_results = inspect_and_clean_dataset(raw_path, check_duplicates=True)
            valid_records = inspection_results["valid_records"]

            if not valid_records:
                print("[Error] No valid images could be identified. Check folder structure.")
                sys.exit(1)

            create_leakage_free_splits(
                records=valid_records,
                output_dir=split_dir,
                train_ratio=0.70,
                val_ratio=0.15,
                test_ratio=0.15,
                random_seed=args.seed,
                copy_files=True,
            )

    # Step 3: Dataloader construction & class imbalance handling
    print("\n[Step 3] Initializing PyTorch DataLoaders with Conservative Augmentations...")
    train_loader, val_loader, test_loader, class_weights = build_dataloaders(
        dataset_dir=split_dir,
        batch_size=args.batch_size,
        num_workers=2,
        use_weighted_sampler=False,
    )

    # Step 4: Two-stage fine-tuning execution
    print("\n[Step 4] Launching Two-Stage Fine-Tuning...")
    model, temperature, history = train_deepfake_classifier(
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        stage1_lr=args.stage1_lr,
        stage2_lr=args.stage2_lr,
        patience=args.patience,
        save_dir=args.save_dir,
        results_dir=args.results_dir,
        seed=args.seed,
    )

    # Step 5: Test evaluation on unseen test set
    print("\n[Step 5] Evaluating Best Checkpoint on Unseen Test Set...")
    eval_results = evaluate_test_set(
        model=model,
        test_loader=test_loader,
        temperature=temperature,
        results_dir=args.results_dir,
    )

    print("\n" + "=" * 70)
    print("        DEEPFAKE FINE-TUNING PIPELINE SUCCESSFULLY COMPLETED")
    print(f"  • Best Checkpoint: {Path(args.save_dir) / 'best_model.pth'}")
    print(f"  • Test Accuracy:   {eval_results['accuracy']*100:.2f}%")
    print(f"  • Test ROC-AUC:    {eval_results['roc_auc']:.4f}")
    print(f"  • Test F1 Score:   {eval_results['f1_score']:.4f}")
    print(f"  • Visualizations:  {args.results_dir}/ (Loss, Accuracy, Confusion Matrix, ROC)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
