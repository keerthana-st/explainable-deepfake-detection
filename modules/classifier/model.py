"""
Neural Model Architecture for Classifier Agent (Module 2).

Uses an EfficientNet-B0 backbone fine-tuned/adapted for deepfake and modern
diffusion artifact detection. Implements dynamic label mapping verification to
eliminate index mismatch bugs.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


def resolve_label_mapping(id2label: Dict[int, str]) -> Tuple[int, int]:
    """
    Dynamically identify which index corresponds to 'FAKE' and which to 'REAL'.

    CRITICAL FIX: Never hardcode index 0 or index 1. Dataset loaders (like
    imagefolder) sort folder names alphabetically ('fake' vs 'real' or '0_real'
    vs '1_fake'), which previously caused complete label inversion bugs.

    Parameters
    ----------
    id2label : Dict[int, str]
        Mapping from class index to label string name.

    Returns
    -------
    Tuple[int, int]
        (fake_idx, real_idx)
    """
    fake_keywords = ["fake", "synthetic", "ai", "manipulated", "deepfake", "generated"]
    real_keywords = ["real", "authentic", "natural", "original", "pristine"]

    fake_idx = None
    real_idx = None

    for idx, name in id2label.items():
        clean_name = str(name).strip().lower()
        if any(kw in clean_name for kw in fake_keywords):
            if fake_idx is not None:
                raise ValueError(f"Ambiguous label mapping: multiple classes match fake ({fake_idx}, {idx})")
            fake_idx = int(idx)
        elif any(kw in clean_name for kw in real_keywords):
            if real_idx is not None:
                raise ValueError(f"Ambiguous label mapping: multiple classes match real ({real_idx}, {idx})")
            real_idx = int(idx)

    if fake_idx is None or real_idx is None:
        raise ValueError(
            f"Unable to automatically resolve label mapping from id2label: {id2label}. "
            f"One class must contain a fake keyword {fake_keywords} and one a real keyword {real_keywords}."
        )

    if fake_idx == real_idx:
        raise ValueError(f"Conflicting label mapping: fake_idx and real_idx are identical ({fake_idx}).")

    return fake_idx, real_idx


class DeepfakeClassifierModel(nn.Module):
    """
    EfficientNet-B0 based deepfake and AI-generation classifier.
    Combines deep semantic convolutional representations with spatial artifact pooling.
    """

    def __init__(
        self,
        id2label: Optional[Dict[int, str]] = None,
        pretrained: bool = True,
    ):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base = models.efficientnet_b0(weights=weights)

        self.features = base.features
        self.avgpool = base.avgpool

        # 1280 feature dimension from EfficientNet-B0
        self.classifier_head = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(1280, 256),
            nn.SiLU(),
            nn.Dropout(p=0.15),
            nn.Linear(256, 2),
        )

        # Standard label mapping
        self.id2label = id2label or {0: "REAL", 1: "FAKE"}
        self.label2id = {v: k for k, v in self.id2label.items()}

        # Verify mapping immediately upon initialization
        self.fake_idx, self.real_idx = resolve_label_mapping(self.id2label)

    def synchronize_labels_with_dataset(self, dataset_classes: list):
        """
        Verify and synchronize model label configuration with dataset classes.

        Parameters
        ----------
        dataset_classes : list
            Class names as assigned by dataset loader (e.g. ['fake', 'real']).
        """
        new_id2label = {i: name for i, name in enumerate(dataset_classes)}
        new_fake_idx, new_real_idx = resolve_label_mapping(new_id2label)

        self.id2label = new_id2label
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.fake_idx = new_fake_idx
        self.real_idx = new_real_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits of shape (B, 2)."""
        f = self.features(x)
        f = self.avgpool(f)
        f = torch.flatten(f, 1)
        logits = self.classifier_head(f)
        return logits


def get_default_transform():
    """Standard EfficientNet-B0 inference preprocessing transform."""
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
