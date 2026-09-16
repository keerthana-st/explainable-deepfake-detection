"""
CLI Runner for Frequency Agent Empirical Calibration.

Runs raw DCT high-frequency scoring across validation images, tabulates score
distributions, calculates ROC curve & Youden's J statistic, and outputs the
empirically derived calibration divisor and decision threshold.
"""

import argparse
import json
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np

from modules.frequency.agent import FrequencyAgent
from modules.frequency.calibration import (
    calibrate_from_scores,
    compute_roc_and_optimal_threshold,
    generate_benchmark_validation_set,
)


def extract_raw_scores(image_dir: Path) -> list:
    """Extract raw high-frequency scores for all images in directory."""
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    image_paths = [p for p in image_dir.glob("*") if p.suffix.lower() in valid_exts]
    if not image_paths:
        raise ValueError(f"No valid image files found in {image_dir}")

    agent = FrequencyAgent()
    scores = []
    print(f"Analyzing {len(image_paths)} images from {image_dir.name}...")
    for p in image_paths:
        try:
            res = agent.analyze(p, keep_original=False)
            scores.append(res.raw_score)
        except Exception as e:
            print(f"Warning: skipped {p.name}: {e}")
    return scores


def plot_distributions_and_roc(
    real_scores: np.ndarray,
    fake_scores: np.ndarray,
    fpr: np.ndarray,
    tpr: np.ndarray,
    optimal_thresh: float,
    output_path: Path,
):
    """Plot score distribution histograms and ROC curve."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Score Distributions
    ax1 = axes[0]
    ax1.hist(real_scores, bins=25, alpha=0.6, color="navy", label=f"Real (N={len(real_scores)})", density=True)
    ax1.hist(fake_scores, bins=25, alpha=0.6, color="crimson", label=f"GAN Fake (N={len(fake_scores)})", density=True)
    ax1.axvline(optimal_thresh, color="black", linestyle="--", linewidth=2, label=f"Optimal Threshold ({optimal_thresh:.2f})")
    ax1.set_title("High-Frequency DCT Energy Distributions", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Raw High-Frequency Energy")
    ax1.set_ylabel("Probability Density")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # 2. ROC Curve
    ax2 = axes[1]
    # Sort for plotting line
    sort_idx = np.argsort(fpr)
    ax2.plot(fpr[sort_idx], tpr[sort_idx], color="darkorange", lw=2, label="ROC Curve")
    ax2.plot([0, 1], [0, 1], color="gray", linestyle=":", label="Random Chance")
    ax2.set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("False Positive Rate (FPR)")
    ax2.set_ylabel("True Positive Rate (TPR / Sensitivity)")
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150)
    plt.close()
    print(f"Calibration plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Empirical Calibration Runner for Frequency Agent")
    parser.add_argument("--real-dir", type=str, default=None, help="Directory containing real validation images")
    parser.add_argument("--fake-dir", type=str, default=None, help="Directory containing GAN fake validation images")
    parser.add_argument("--output-plot", type=str, default="calibration_report.png", help="Filepath for distribution/ROC plot")
    parser.add_argument("--export-json", type=str, default="calibration_config.json", help="Filepath to export calibration metadata")
    parser.add_argument("--dataset-name", type=str, default="Empirical-GAN-Artifact-Benchmark-v1", help="Provenance dataset name")
    args = parser.parse_args()

    # If no directories provided, generate empirical validation benchmark (100 real, 100 fake)
    if args.real_dir is None or args.fake_dir is None:
        print("No input directories provided. Generating benchmark validation set (100 real, 100 fake)...")
        bench_dir = PROJECT_ROOT / "data" / "calibration_benchmark"
        real_path, fake_path = generate_benchmark_validation_set(bench_dir, n_real=100, n_fake=100)
    else:
        real_path = Path(args.real_dir)
        fake_path = Path(args.fake_dir)

    real_scores = np.array(extract_raw_scores(real_path))
    fake_scores = np.array(extract_raw_scores(fake_path))

    # Derive calibration parameters
    cal_meta = calibrate_from_scores(real_scores, fake_scores, dataset_name=args.dataset_name)
    opt_thresh, max_j, auc_roc, fpr, tpr, _ = compute_roc_and_optimal_threshold(real_scores, fake_scores)

    # Print tabulation
    print("\n" + "=" * 65)
    print("      FREQUENCY AGENT EMPIRICAL CALIBRATION REPORT")
    print("=" * 65)
    print(f"Dataset Provenance:       {cal_meta.dataset_name}")
    print(f"Calibration Date:         {cal_meta.calibration_date}")
    print(f"Real Samples Evaluated:   {len(real_scores)}")
    print(f"Fake Samples Evaluated:   {len(fake_scores)}")
    print("-" * 65)
    print("Score Distribution Metrics:")
    print(f"  Real Images:  Mean = {cal_meta.real_mean:8.4f}, Std = {cal_meta.real_std:8.4f}, Range = [{np.min(real_scores):.4f}, {np.max(real_scores):.4f}]")
    print(f"  Fake Images:  Mean = {cal_meta.fake_mean:8.4f}, Std = {cal_meta.fake_std:8.4f}, Range = [{np.min(fake_scores):.4f}, {np.max(fake_scores):.4f}]")
    print("-" * 65)
    print("Empirical Separation Metrics:")
    print(f"  Optimal Raw Threshold (Youden J max): {opt_thresh:8.4f}")
    print(f"  Calibrated Divisor (2x Raw Thresh):   {cal_meta.divisor:8.4f}")
    print(f"  Normalized Decision Threshold:        {cal_meta.threshold:8.4f}")
    print(f"  Maximum Youden's J Statistic:         {cal_meta.youden_j:8.4f}")
    print(f"  Area Under ROC Curve (AUC-ROC):       {cal_meta.auc_roc:8.4f}")
    print("=" * 65)

    # Export Plot
    plot_path = Path(args.output_plot)
    plot_distributions_and_roc(real_scores, fake_scores, fpr, tpr, opt_thresh, plot_path)

    # Export JSON
    json_path = Path(args.export_json)
    json_path.write_text(json.dumps(cal_meta.to_dict(), indent=2), encoding="utf-8")
    print(f"Calibration metadata exported to: {json_path}")


if __name__ == "__main__":
    main()
