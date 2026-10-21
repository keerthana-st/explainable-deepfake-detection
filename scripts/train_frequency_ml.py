"""
Training & Calibration Runner for Frequency Agent ML Model.

Generates a balanced multi-generator and multi-manipulation training set:
- Authentic Camera (uncompressed optical shot noise)
- Real Compressed (JPEG Q=35 to 75)
- Real Social Media (downscaled & re-encoded)
- GAN Fakes (checkerboard Nyquist upsampling)
- Diffusion Fakes (smooth latent VAE decoders)
- Inpainting Clothing Swaps (hybrid face + synthetic clothing)
- Inpainting Background Edits (hybrid subject + synthetic background)
- Face-Preserving Expression Edits

Extracts 28-dimensional frequency feature vectors, fits the Cost-Sensitive
ElasticNet Logistic Regression model, and serializes to data/frequency_ml_model.json.
"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.frequency.extractor import FrequencyFeatureExtractor, FEATURE_NAMES
from modules.frequency.ml_classifier import FrequencyMLClassifier


def generate_diverse_frequency_dataset(
    output_dir: Path,
    samples_per_category: int = 40,
    random_seed: int = 42,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """Generate multi-category dataset for frequency ML training and validation."""
    rng = np.random.default_rng(random_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    size = 256

    all_samples: List[Tuple[Path, int]] = []

    def make_base_photo():
        x = np.linspace(0, rng.uniform(2.0, 4.0) * np.pi, size)
        y = np.linspace(0, rng.uniform(2.0, 4.0) * np.pi, size)
        xx, yy = np.meshgrid(x, y)
        return np.sin(xx) * np.cos(yy) * 80.0 + 128.0

    # 1. Real: Authentic Camera (Label 0)
    for i in range(samples_per_category):
        base = make_base_photo()
        shot_noise = rng.normal(0, rng.uniform(2.0, 5.0), (size, size))
        img = np.clip(base + shot_noise, 0, 255).astype(np.uint8)
        p = output_dir / f"real_cam_{i:03d}.png"
        cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        all_samples.append((p, 0))

    # 2. Real: Compressed Photos (Label 0)
    for i in range(samples_per_category):
        base = make_base_photo()
        shot_noise = rng.normal(0, 3.0, (size, size))
        img = np.clip(base + shot_noise, 0, 255).astype(np.uint8)
        p = output_dir / f"real_comp_{i:03d}.jpg"
        q = int(rng.uniform(35, 75))
        cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
        all_samples.append((p, 0))

    # 3. Real: Social Media Resized (Label 0)
    for i in range(samples_per_category):
        base = make_base_photo()
        img = np.clip(base + rng.normal(0, 3.5, (size, size)), 0, 255).astype(np.uint8)
        # Downscale and upscale to simulate social media compression
        small = cv2.resize(img, (size // 2, size // 2), interpolation=cv2.INTER_AREA)
        restored = cv2.resize(small, (size, size), interpolation=cv2.INTER_CUBIC)
        p = output_dir / f"real_social_{i:03d}.jpg"
        cv2.imwrite(str(p), cv2.cvtColor(restored, cv2.COLOR_GRAY2BGR), [cv2.IMWRITE_JPEG_QUALITY, 55])
        all_samples.append((p, 0))

    # 4. Fake: GAN Checkerboard (Label 1)
    for i in range(samples_per_category):
        base = make_base_photo()
        grid = rng.choice([2, 4])
        base[::grid, ::grid] += rng.uniform(30.0, 55.0)
        base[1::grid, 1::grid] -= rng.uniform(30.0, 55.0)
        img = np.clip(base, 0, 255).astype(np.uint8)
        p = output_dir / f"fake_gan_{i:03d}.png"
        cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        all_samples.append((p, 1))

    # 5. Fake: Modern Diffusion (Label 1)
    for i in range(samples_per_category):
        base = make_base_photo()
        # Smooth latent VAE decodings + high-contrast edge dispersion
        diff = cv2.GaussianBlur(base, (7, 7), 0)
        diff[::2, :] += rng.uniform(15.0, 30.0)
        img = np.clip(diff, 0, 255).astype(np.uint8)
        p = output_dir / f"fake_diff_{i:03d}.png"
        cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        all_samples.append((p, 1))

    # 6. Fake: Inpainting Clothing Swap (Label 1)
    for i in range(samples_per_category):
        base = make_base_photo()
        hybrid = base.copy()
        h_mid = size // 2
        # Synthetic fabric grid in lower half
        hybrid[h_mid:, :] = cv2.GaussianBlur(hybrid[h_mid:, :], (5, 5), 0)
        hybrid[h_mid::2, ::2] += rng.uniform(25.0, 45.0)
        # Seam blending
        hybrid[h_mid - 3 : h_mid + 3, :] = cv2.GaussianBlur(hybrid[h_mid - 3 : h_mid + 3, :], (3, 3), 0)
        img = np.clip(hybrid, 0, 255).astype(np.uint8)
        p = output_dir / f"fake_clothing_{i:03d}.png"
        cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        all_samples.append((p, 1))

    # 7. Fake: Inpainting Background (Label 1)
    for i in range(samples_per_category):
        base = make_base_photo()
        # Smooth diffusion background with sharp real foreground patch
        bg = cv2.GaussianBlur(base, (9, 9), 0)
        # Real subject in center
        cx, cy = size // 2, size // 2
        bg[cy - 40 : cy + 40, cx - 40 : cx + 40] = base[cy - 40 : cy + 40, cx - 40 : cx + 40] + rng.normal(0, 4.0, (80, 80))
        img = np.clip(bg, 0, 255).astype(np.uint8)
        p = output_dir / f"fake_bg_{i:03d}.png"
        cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        all_samples.append((p, 1))

    # Split 80/20 train/val
    rng.shuffle(all_samples)
    split = int(len(all_samples) * 0.80)
    return all_samples[:split], all_samples[split:]


def train_frequency_model():
    print("=" * 65)
    print("   TRAINING DATA-DRIVEN FREQUENCY AGENT ML CLASSIFIER")
    print("=" * 65)

    data_dir = PROJECT_ROOT / "data" / "frequency_training"
    train_samples, val_samples = generate_diverse_frequency_dataset(data_dir, samples_per_category=35)
    print(f"Dataset generated: {len(train_samples)} training samples, {len(val_samples)} validation samples.")

    extractor = FrequencyFeatureExtractor(target_size=256)

    print("Extracting 28-dimensional frequency feature vectors...")
    X_train, y_train = [], []
    for p, label in train_samples:
        feats = extractor.extract_from_file(p)
        X_train.append(feats.vector)
        y_train.append(label)
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)

    X_val, y_val = [], []
    for p, label in val_samples:
        feats = extractor.extract_from_file(p)
        X_val.append(feats.vector)
        y_val.append(label)
    X_val = np.array(X_val, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.float32)

    # Train Cost-Sensitive ElasticNet Model
    clf = FrequencyMLClassifier(indifference_low=0.35, indifference_high=0.65, threshold=0.50)
    clf.fit(
        X_train,
        y_train,
        feature_names=FEATURE_NAMES,
        l1_ratio=0.25,
        alpha=0.015,
        learning_rate=0.04,
        max_iter=1200,
        fake_weight=1.0,
        real_weight=2.5,  # Higher penalty on false alarms on real photos
    )

    # Evaluate on validation split
    val_preds = []
    inconclusive_count = 0
    for x in X_val:
        exp = clf.explain(x)
        if exp["verdict"] == "INCONCLUSIVE_FREQUENCY":
            inconclusive_count += 1
            val_preds.append(0.5)
        elif exp["verdict"] == "MANIPULATED_FREQUENCY":
            val_preds.append(1)
        else:
            val_preds.append(0)

    # Decided accuracy (excluding inconclusive)
    decided_idx = [i for i, v in enumerate(val_preds) if v != 0.5]
    if decided_idx:
        acc = np.mean([val_preds[i] == y_val[i] for i in decided_idx])
    else:
        acc = 0.0

    print(f"\nValidation Results:")
    print(f"• Decided Samples Accuracy: {acc * 100:.2f}% ({len(decided_idx)}/{len(val_samples)})")
    print(f"• Inconclusive Safeguard Rate: {inconclusive_count / len(val_samples) * 100:.2f}% ({inconclusive_count} samples)")

    # Save trained model
    out_model_path = PROJECT_ROOT / "data" / "frequency_ml_model.json"
    clf.save_model(out_model_path)
    print(f"\nModel successfully saved to: {out_model_path}")
    print("=" * 65)


if __name__ == "__main__":
    train_frequency_model()
