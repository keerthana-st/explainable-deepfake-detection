"""
Empirical Calibration Engine for the Frequency Agent.

Derives the normalization divisor and decision threshold empirically using
ROC curve analysis and Youden's J statistic (J = Sensitivity + Specificity - 1)
on labeled validation datasets of Real vs GAN-manipulated images.
"""

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


@dataclass
class CalibrationMetadata:
    """Stores calibration constants and validation provenance."""
    divisor: float
    threshold: float
    raw_threshold: float
    dataset_name: str
    calibration_date: str
    auc_roc: float
    youden_j: float
    real_mean: float
    real_std: float
    fake_mean: float
    fake_std: float

    def to_dict(self) -> Dict[str, Any]:
        """Format calibration metadata as structured dictionary."""
        return {
            "divisor": round(self.divisor, 4),
            "threshold": round(self.threshold, 4),
            "raw_threshold": round(self.raw_threshold, 4),
            "calibration_dataset": self.dataset_name,
            "calibration_date": self.calibration_date,
            "auc_roc": round(self.auc_roc, 4),
            "youden_j": round(self.youden_j, 4),
            "distribution_summary": {
                "real_mean": round(self.real_mean, 4),
                "real_std": round(self.real_std, 4),
                "fake_mean": round(self.fake_mean, 4),
                "fake_std": round(self.fake_std, 4),
            },
        }


def compute_roc_and_optimal_threshold(
    real_scores: Union[List[float], np.ndarray],
    fake_scores: Union[List[float], np.ndarray],
) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve, AUC, and optimal threshold maximizing Youden's J.

    Parameters
    ----------
    real_scores : list or np.ndarray
        Raw frequency scores from known real images (ground truth = 0).
    fake_scores : list or np.ndarray
        Raw frequency scores from known GAN fake images (ground truth = 1).

    Returns
    -------
    optimal_raw_threshold : float
    max_youden_j : float
    auc_roc : float
    fpr_array : np.ndarray
    tpr_array : np.ndarray
    thresholds : np.ndarray
    """
    real = np.asarray(real_scores, dtype=np.float64)
    fake = np.asarray(fake_scores, dtype=np.float64)

    if len(real) == 0 or len(fake) == 0:
        raise ValueError("Both real and fake score collections must be non-empty.")

    # Candidate thresholds across full observed range
    all_scores = np.concatenate([real, fake])
    thresholds = np.unique(all_scores)
    thresholds = np.sort(thresholds)
    # Add boundary points
    thresholds = np.concatenate([[thresholds[0] - 1e-5], thresholds, [thresholds[-1] + 1e-5]])

    n_real = len(real)
    n_fake = len(fake)

    tpr_list = []
    fpr_list = []
    j_list = []

    for t in thresholds:
        # Fakes classified as fake (True Positives)
        tpr = np.sum(fake >= t) / n_fake
        # Reals classified as fake (False Positives)
        fpr = np.sum(real >= t) / n_real
        j = tpr - fpr

        tpr_list.append(tpr)
        fpr_list.append(fpr)
        j_list.append(j)

    tpr_arr = np.array(tpr_list)
    fpr_arr = np.array(fpr_list)
    j_arr = np.array(j_list)

    # Optimal threshold by Youden's J statistic: max(TPR - FPR)
    best_idx = int(np.argmax(j_arr))
    optimal_raw_threshold = float(thresholds[best_idx])
    max_youden_j = float(j_arr[best_idx])

    # Compute AUC-ROC via numerical trapezoidal integration
    # Sort by FPR ascending for integration
    sort_order = np.argsort(fpr_arr)
    sorted_fpr = fpr_arr[sort_order]
    sorted_tpr = tpr_arr[sort_order]

    # Trapezoid rule: integral of TPR over d(FPR)
    auc_roc = float(np.sum((sorted_fpr[1:] - sorted_fpr[:-1]) * (sorted_tpr[1:] + sorted_tpr[:-1]) / 2.0))
    auc_roc = abs(auc_roc)

    return optimal_raw_threshold, max_youden_j, auc_roc, fpr_arr, tpr_arr, thresholds


def calibrate_from_scores(
    real_scores: Union[List[float], np.ndarray],
    fake_scores: Union[List[float], np.ndarray],
    dataset_name: str = "Empirical-GAN-Validation-Set",
    calibration_date_str: Optional[str] = None,
) -> CalibrationMetadata:
    """
    Derive calibration divisor and threshold from score collections.

    To ensure standard interpretation, the calibration divisor is set to
    (2.0 * optimal_raw_threshold). Under this calibration:
    - raw_score == optimal_raw_threshold maps to normalized frequency_score == 0.50
    - raw_score < optimal_raw_threshold maps to frequency_score < 0.50 (REAL)
    - raw_score > optimal_raw_threshold maps to frequency_score > 0.50 (SUSPICIOUS)
    """
    real = np.asarray(real_scores, dtype=np.float64)
    fake = np.asarray(fake_scores, dtype=np.float64)

    raw_thresh, youden_j, auc_roc, _, _, _ = compute_roc_and_optimal_threshold(real, fake)

    # Divisor mapping: T_raw / divisor = 0.50 => divisor = 2.0 * T_raw
    divisor = float(2.0 * raw_thresh) if raw_thresh > 0 else 1.0
    normalized_threshold = 0.50

    cal_date = calibration_date_str or date.today().isoformat()

    return CalibrationMetadata(
        divisor=divisor,
        threshold=normalized_threshold,
        raw_threshold=raw_thresh,
        dataset_name=dataset_name,
        calibration_date=cal_date,
        auc_roc=auc_roc,
        youden_j=youden_j,
        real_mean=float(np.mean(real)),
        real_std=float(np.std(real)),
        fake_mean=float(np.mean(fake)),
        fake_std=float(np.std(fake)),
    )


def generate_benchmark_validation_set(
    output_dir: Union[str, Path],
    n_real: int = 100,
    n_fake: int = 100,
    img_size: int = 256,
    random_seed: int = 42,
) -> Tuple[Path, Path]:
    """
    Generate an empirical benchmark dataset of 100 real-like images and 100
    GAN-checkerboard fake images for rigorous reproducible calibration.
    """
    rng = np.random.default_rng(random_seed)
    out_path = Path(output_dir)
    real_dir = out_path / "real"
    fake_dir = out_path / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate 100 Real-like images (smooth gradients, natural photo textures, varied lighting)
    for i in range(n_real):
        # Base natural gradient
        x = np.linspace(0, rng.uniform(1.0, 4.0) * np.pi, img_size)
        y = np.linspace(0, rng.uniform(1.0, 4.0) * np.pi, img_size)
        xx, yy = np.meshgrid(x, y)
        base = np.sin(xx) * np.cos(yy) * 127.0 + 128.0

        # Add natural soft low-frequency variations
        low_noise = cv2.GaussianBlur(
            rng.normal(0, 30, (img_size, img_size)), (15, 15), 0
        )
        img_val = np.clip(base + low_noise, 0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_val, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(real_dir / f"real_{i:03d}.png"), img_bgr)

    # 2. Generate 100 GAN-like images (transposed convolution checkerboard artifacts)
    # Generative models leave periodic spatial grid harmonics at stride 4 & 8
    for i in range(n_fake):
        # Base image similar to natural photos
        x = np.linspace(0, rng.uniform(1.0, 4.0) * np.pi, img_size)
        y = np.linspace(0, rng.uniform(1.0, 4.0) * np.pi, img_size)
        xx, yy = np.meshgrid(x, y)
        base = np.sin(xx) * np.cos(yy) * 127.0 + 128.0

        # Periodic checkerboard upsampling grid (simulating transposed-convolution strided deconv)
        grid_pattern = np.zeros((img_size, img_size), dtype=np.float32)
        freq_stride = rng.choice([2, 4])
        grid_pattern[::freq_stride, ::freq_stride] = rng.uniform(25.0, 50.0)
        grid_pattern[1::freq_stride, 1::freq_stride] = -rng.uniform(25.0, 50.0)

        # Combine base + periodic high-frequency upsampling artifacts
        synthetic_img = np.clip(base + grid_pattern, 0, 255).astype(np.uint8)
        synthetic_bgr = cv2.cvtColor(synthetic_img, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(fake_dir / f"fake_gan_{i:03d}.png"), synthetic_bgr)

    return real_dir, fake_dir
