"""
Dataset Management, Inspection, Cleaning, and DataLoader Pipeline for Deepfake Detection.

Features:
- Kaggle dataset integration (download & extraction)
- Corrupt/unreadable image filtering (Pillow verification)
- Duplicate detection via MD5 byte hash and 64-bit dHash perceptual hashing
- Leakage-free 70/15/15 train/val/test splitting
- Conservative image augmentation tailored for subtle deepfake artifact retention
- Class imbalance calculation and weighted sampling
- Dynamic label synchronization preventing inverted class mapping
"""

import hashlib
import json
import os
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm


import re

# Standard class keywords
REAL_KEYWORDS = ["real", "authentic", "natural", "original", "pristine", "0_real", "real_faces"]
FAKE_KEYWORDS = ["fake", "synthetic", "ai", "manipulated", "deepfake", "generated", "1_fake", "fake_faces"]


def resolve_class_label(folder_name: str) -> Optional[int]:
    """
    Map folder name or label string to standard binary index: 0 for REAL, 1 for FAKE.
    Prevents false substring matches (e.g., 'ai' inside 'portraits').
    """
    clean = folder_name.lower().strip()
    # Split into clean alphanumeric tokens
    tokens = [t for t in re.split(r"[\s_\-\/\\]+", clean) if t]

    # Priority 1: Exact token matching
    for token in tokens:
        if token in ["fake", "synthetic", "ai", "manipulated", "deepfake", "generated", "1_fake", "fake_faces"]:
            return 1
        if token in ["real", "authentic", "natural", "original", "pristine", "0_real", "real_faces"]:
            return 0

    # Priority 2: Substring matching for longer descriptive roots (exclude ambiguous 2-letter tokens like 'ai')
    for fk in ["fake", "synthetic", "manipulat", "deepfake", "generat"]:
        if fk in clean:
            return 1
    for rk in ["real", "authentic", "natural", "original", "pristine"]:
        if rk in clean:
            return 0

    return None


def download_kaggle_dataset(dataset_slug: str, target_dir: Union[str, Path]) -> Path:
    """
    Download and extract a dataset from Kaggle via the Kaggle API.
    
    Args:
        dataset_slug: e.g. 'xhlulu/140k-real-and-fake-faces' or 'davitshabgard/deepfake-faces'
        target_dir: Directory where the unzipped files should reside
    """
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    print(f"[Kaggle] Initiating download for '{dataset_slug}' into {target}...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(dataset_slug, path=str(target), unzip=True, quiet=False)
        print(f"[Kaggle] Successfully downloaded and extracted dataset to: {target}")
    except Exception as e:
        print(f"[Kaggle Error] Failed to download automatically: {e}")
        print("[Kaggle Instructions]:")
        print("  1. Ensure kaggle is installed: pip install kaggle")
        print("  2. Place your kaggle.json file in ~/.kaggle/ (Linux/Colab) or %USERPROFILE%\\.kaggle\\ (Windows)")
        print(f"  3. Or run manually in terminal: kaggle datasets download -d {dataset_slug} -p {target} --unzip")
        raise
    return target


def calculate_dhash(image: Image.Image, hash_size: int = 8) -> int:
    """
    Compute 64-bit difference hash (dHash) to detect duplicate / resaved images.
    Robust to slight compression and small color variations.
    """
    # Resize to (width + 1, height) grayscale
    resized = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BICUBIC)
    pixels = np.array(resized)
    # Compare adjacent pixels
    difference = pixels[:, 1:] > pixels[:, :-1]
    # Convert binary array to integer hash
    decimal_val = 0
    for idx, val in enumerate(difference.flatten()):
        if val:
            decimal_val += 1 << idx
    return decimal_val


def inspect_and_clean_dataset(
    source_dir: Union[str, Path],
    min_size: int = 16,
    check_duplicates: bool = True,
) -> Dict[str, Any]:
    """
    Scan, validate, and clean raw dataset directory.
    - Discards 0-byte, truncated, or unreadable images.
    - Identifies duplicates using MD5 file hashing and perceptual dHash.
    - Categorizes images into binary Real (0) and Fake (1).
    
    Returns:
        Dictionary containing verified image paths, labels, and statistics.
    """
    source_path = Path(source_dir)
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

    valid_records: List[Dict[str, Any]] = []
    corrupted_files: List[Path] = []
    duplicate_files: List[Path] = []
    unmapped_files: List[Path] = []

    seen_md5: Set[str] = set()
    seen_dhash: Set[int] = set()

    print(f"[Dataset Cleaning] Scanning directory: {source_path}")
    all_files = [p for p in source_path.rglob("*") if p.is_file() and p.suffix.lower() in valid_extensions]
    print(f"[Dataset Cleaning] Found {len(all_files)} candidate image files. Inspecting integrity...")

    for file_path in tqdm(all_files, desc="Validating & Hashing"):
        # 1. Check size
        if file_path.stat().st_size == 0:
            corrupted_files.append(file_path)
            continue

        # 2. Map class label from folder structure
        # Check parents from closest to farthest
        label_idx = None
        for part in reversed(file_path.parts[:-1]):
            res = resolve_class_label(part)
            if res is not None:
                label_idx = res
                break

        if label_idx is None:
            unmapped_files.append(file_path)
            continue

        # 3. Verify image readability and dimensions using Pillow
        try:
            with open(file_path, "rb") as f:
                content = f.read()
                md5_hash = hashlib.md5(content).hexdigest()

            with Image.open(file_path) as img:
                img.verify()

            # Re-open after verify to check dimensions
            with Image.open(file_path) as img:
                w, h = img.size
                if w < min_size or h < min_size:
                    corrupted_files.append(file_path)
                    continue

                # 4. Check duplicates if enabled
                if check_duplicates:
                    if md5_hash in seen_md5:
                        duplicate_files.append(file_path)
                        continue
                    seen_md5.add(md5_hash)

                    dh = calculate_dhash(img)
                    if dh in seen_dhash:
                        duplicate_files.append(file_path)
                        continue
                    seen_dhash.add(dh)

        except Exception:
            corrupted_files.append(file_path)
            continue

        valid_records.append({
            "path": file_path,
            "label": label_idx,
            "class_name": "FAKE" if label_idx == 1 else "REAL",
            "width": w,
            "height": h,
            "stem": file_path.stem,
        })

    real_count = sum(1 for r in valid_records if r["label"] == 0)
    fake_count = sum(1 for r in valid_records if r["label"] == 1)

    print("\n" + "=" * 60)
    print("           DATASET INSPECTION & CLEANING REPORT")
    print("=" * 60)
    print(f"Total files scanned:    {len(all_files)}")
    print(f"Valid verified images:  {len(valid_records)}")
    print(f"  • Real images:        {real_count}")
    print(f"  • Fake images:        {fake_count}")
    print(f"  • Class ratio:        {real_count / max(fake_count, 1):.2f}:1")
    print(f"Corrupted / truncated:  {len(corrupted_files)}")
    print(f"Duplicates removed:     {len(duplicate_files)}")
    print(f"Unmapped / skipped:     {len(unmapped_files)}")
    print("=" * 60 + "\n")

    return {
        "valid_records": valid_records,
        "real_count": real_count,
        "fake_count": fake_count,
        "corrupted_count": len(corrupted_files),
        "duplicate_count": len(duplicate_files),
    }


def create_leakage_free_splits(
    records: List[Dict[str, Any]],
    output_dir: Union[str, Path],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    copy_files: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Split verified records into Train, Validation, and Test sets without data leakage.
    Ensures:
    - Balanced stratification across classes.
    - Identity / Group isolation: images sharing identical base prefix/identity are kept in the same split.
    - Physical directory structure created: output_dir/{train,val,test}/{real,fake}/
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-4, "Splits must sum to 1.0"
    rng = np.random.default_rng(random_seed)
    out_path = Path(output_dir)

    # Group by identity/prefix if present to prevent leakage between frames of the same origin
    def extract_group_id(record: Dict[str, Any]) -> str:
        stem = record["stem"]
        # If filename follows pattern like id001_frame02.jpg or personA_003.jpg
        parts = stem.split("_")
        if len(parts) > 1 and len(parts[0]) > 2:
            return f"{record['label']}_{parts[0]}"
        return f"{record['label']}_{stem}"

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        gid = extract_group_id(r)
        if gid not in groups:
            groups[gid] = []
        groups[gid].append(r)

    # Separate groups by class
    real_gids = [gid for gid in groups if gid.startswith("0_")]
    fake_gids = [gid for gid in groups if gid.startswith("1_")]

    rng.shuffle(real_gids)
    rng.shuffle(fake_gids)

    def split_gids(gids: List[str]) -> Tuple[List[str], List[str], List[str]]:
        n = len(gids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_g = gids[:n_train]
        val_g = gids[n_train : n_train + n_val]
        test_g = gids[n_train + n_val :]
        return train_g, val_g, test_g

    train_real_g, val_real_g, test_real_g = split_gids(real_gids)
    train_fake_g, val_fake_g, test_fake_g = split_gids(fake_gids)

    splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}

    for gid in train_real_g + train_fake_g:
        splits["train"].extend(groups[gid])
    for gid in val_real_g + val_fake_g:
        splits["val"].extend(groups[gid])
    for gid in test_real_g + test_fake_g:
        splits["test"].extend(groups[gid])

    # Shuffle each split
    for s in splits:
        rng.shuffle(splits[s])

    print("[Dataset Splitting] Stratified Leakage-Free Split Results:")
    for s_name in ["train", "val", "test"]:
        reals = sum(1 for r in splits[s_name] if r["label"] == 0)
        fakes = sum(1 for r in splits[s_name] if r["label"] == 1)
        pct = len(splits[s_name]) / max(len(records), 1) * 100
        print(f"  • {s_name.upper():<5}: {len(splits[s_name]):>6} samples ({pct:.1f}%) | Real: {reals} | Fake: {fakes}")

    # Copy files into physical structure if requested
    if copy_files:
        print(f"[Dataset Splitting] Organizing directory structure at: {out_path}")
        for s_name, rec_list in splits.items():
            for r in tqdm(rec_list, desc=f"Writing {s_name} files"):
                sub_folder = "real" if r["label"] == 0 else "fake"
                dest_dir = out_path / s_name / sub_folder
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_file = dest_dir / Path(r["path"]).name
                if not dest_file.exists():
                    shutil.copy2(r["path"], dest_file)
                r["split_path"] = dest_file

    return splits


class DeepfakeDataset(Dataset):
    """
    Standard PyTorch Dataset for EfficientNet-B0 Deepfake Classification.
    Ensures safe RGB conversion and applies designated transforms.
    """

    def __init__(
        self,
        samples: List[Tuple[Path, int]],
        transform: Optional[Callable] = None,
    ):
        self.samples = samples
        self.transform = transform
        self.classes = ["REAL", "FAKE"]
        self.class_to_idx = {"REAL": 0, "FAKE": 1}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        file_path, label = self.samples[idx]
        try:
            with Image.open(file_path) as img:
                img = ImageOps.exif_transpose(img)
                rgb_img = img.convert("RGB")
        except Exception:
            # Fallback to zero tensor on rare read glitch
            rgb_img = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform is not None:
            tensor = self.transform(rgb_img)
        else:
            tensor = transforms.ToTensor()(rgb_img)

        return tensor, label


def get_data_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns train and evaluation transforms.
    
    FORENSIC JUSTIFICATION FOR AUGMENTATION:
    - RandomHorizontalFlip(p=0.5): Human faces and general scenes are laterally symmetric.
    - RandomRotation(degrees=(-10, 10)): Models subtle camera tilt without distorting pixel structures.
    - ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1): Simulates varying ambient illumination.
    - PROHIBITED: Aggressive Gaussian blurring, heavy compression distortion, CutMix, or affine shear,
      as these actively destroy the high-frequency pixel lattices and boundary seams that reveal deepfakes.
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-10, 10), interpolation=InterpolationMode.BILINEAR),
        transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_test_transform


def build_dataloaders(
    dataset_dir: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 2,
    use_weighted_sampler: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Construct Train, Validation, and Test PyTorch DataLoaders from organized split directory.
    Calculates class imbalance and returns loss weights.
    """
    root = Path(dataset_dir)
    train_transform, val_test_transform = get_data_transforms()

    def load_split_samples(split_name: str) -> List[Tuple[Path, int]]:
        samples = []
        for class_candidates, label_idx in [(["real", "REAL", "0_real"], 0), (["fake", "FAKE", "1_fake"], 1)]:
            for c_name in class_candidates:
                d = root / split_name / c_name
                if d.exists() and d.is_dir():
                    for f in d.iterdir():
                        if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                            samples.append((f, label_idx))
                    break
        return samples

    train_samples = load_split_samples("train")
    val_samples = load_split_samples("val")
    test_samples = load_split_samples("test")

    if not train_samples:
        raise ValueError(f"No training samples found in {root / 'train'}. Run splitting first.")

    train_dataset = DeepfakeDataset(train_samples, transform=train_transform)
    val_dataset = DeepfakeDataset(val_samples, transform=val_test_transform)
    test_dataset = DeepfakeDataset(test_samples, transform=val_test_transform)

    # Class distribution
    train_labels = [s[1] for s in train_samples]
    counts = Counter(train_labels)
    n_real = counts.get(0, 1)
    n_fake = counts.get(1, 1)
    total = n_real + n_fake

    # Inverse frequency weights for CrossEntropyLoss
    class_weights = torch.tensor([total / (2.0 * n_real), total / (2.0 * n_fake)], dtype=torch.float32)
    print(f"[DataLoader] Train Class Counts -> Real: {n_real}, Fake: {n_fake} | Computed Loss Weights: {class_weights.tolist()}")

    sampler = None
    shuffle = True
    if use_weighted_sampler:
        sample_weights = [class_weights[label] for label in train_labels]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False
        print("[DataLoader] WeightedRandomSampler enabled for class balancing.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, class_weights
