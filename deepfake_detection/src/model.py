"""
EfficientNet-B0 Transfer Learning Architecture for Deepfake Detection.

Features:
- Pretrained ImageNet weights initialization
- Custom binary classification head with dropout regularization
- Two-stage training API:
  * Stage 1: Frozen backbone (train classifier head only)
  * Stage 2: Gradual unfreezing of top convolutional blocks
- Dynamic label synchronization preventing inverted index mapping
- Checkpoint persistence and metadata tracking
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchvision.models as models


def resolve_label_mapping(id2label: Dict[int, str]) -> Tuple[int, int]:
    """
    Dynamically verify class index mapping.
    Ensures index for 'FAKE' and 'REAL' are definitively resolved,
    eliminating alphabetical folder inversion bugs.
    """
    fake_keywords = ["fake", "synthetic", "ai", "manipulated", "deepfake", "generated", "1_fake"]
    real_keywords = ["real", "authentic", "natural", "original", "pristine", "0_real"]

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
            f"Unable to resolve label mapping from id2label: {id2label}. "
            "Must contain one real and one fake class."
        )

    return fake_idx, real_idx


class EfficientNetDeepfakeClassifier(nn.Module):
    """
    EfficientNet-B0 Backbone with Two-Stage Transfer Learning and Binary Classification Head.
    """

    def __init__(
        self,
        pretrained: bool = True,
        dropout_rate: float = 0.30,
        id2label: Optional[Dict[int, str]] = None,
    ):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base = models.efficientnet_b0(weights=weights)

        # 1. Feature Extractor (EfficientNet-B0 has 8 sequential feature blocks)
        self.features = base.features
        self.avgpool = base.avgpool

        # 2. Custom Binary Classifier Head
        self.classifier_head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(1280, 256),
            nn.SiLU(),
            nn.Dropout(p=dropout_rate * 0.67),
            nn.Linear(256, 2),
        )

        # 3. Label configuration
        self.id2label = id2label or {0: "REAL", 1: "FAKE"}
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.fake_idx, self.real_idx = resolve_label_mapping(self.id2label)

        # Default to Stage 1: freeze backbone
        self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """
        Stage 1: Freeze all parameters in feature extractor backbone.
        Only the custom classification head is trainable.
        """
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.classifier_head.parameters():
            param.requires_grad = True
        print("[Model] Stage 1 Configured: EfficientNet-B0 backbone FROZEN. Classifier head TRAINABLE.")
        self._print_param_stats()

    def unfreeze_top_blocks(self, num_blocks: int = 2) -> None:
        """
        Stage 2: Unfreeze the final convolutional stages (blocks 6 and 7 of EfficientNet-B0)
        for delicate fine-tuning of domain-specific deepfake features.
        
        Args:
            num_blocks: Number of final blocks to unfreeze (default: 2, corresponds to blocks 6 and 7)
        """
        # Ensure head is trainable
        for param in self.classifier_head.parameters():
            param.requires_grad = True

        total_blocks = len(self.features)
        unfreeze_start_idx = max(0, total_blocks - num_blocks)

        for i, block in enumerate(self.features):
            is_trainable = i >= unfreeze_start_idx
            for param in block.parameters():
                param.requires_grad = is_trainable

        print(f"[Model] Stage 2 Configured: Unfroze top {num_blocks} blocks (blocks {unfreeze_start_idx} to {total_blocks-1}).")
        self._print_param_stats()

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters across the entire network."""
        for param in self.parameters():
            param.requires_grad = True
        print("[Model] Fully Unfrozen: All backbone and head layers are TRAINABLE.")
        self._print_param_stats()

    def _print_param_stats(self) -> None:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"        Trainable Params: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (B, 3, 224, 224)
        Returns:
            Logits tensor of shape (B, 2)
        """
        feat = self.features(x)
        pooled = self.avgpool(feat)
        flattened = torch.flatten(pooled, 1)
        logits = self.classifier_head(flattened)
        return logits

    def save_checkpoint(
        self,
        save_path: Union[str, Path],
        epoch: int,
        val_loss: float,
        val_acc: float,
        val_f1: float,
        temperature: float = 1.0,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save weights, training state, and metadata."""
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "state_dict": self.state_dict(),
            "id2label": self.id2label,
            "fake_idx": self.fake_idx,
            "real_idx": self.real_idx,
            "epoch": epoch,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "temperature": temperature,
            "architecture": "EfficientNet-B0",
            "metadata": extra_meta or {},
        }
        torch.save(payload, str(p))
        print(f"[Checkpoint] Saved best model to: {p} (Epoch {epoch} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f})")
        return p

    def load_checkpoint(self, checkpoint_path: Union[str, Path], map_location: Optional[str] = None) -> Dict[str, Any]:
        """Load weights and verify label mappings."""
        p = Path(checkpoint_path)
        if not p.exists():
            raise FileNotFoundError(f"Model checkpoint not found at: {p}")

        device = map_location or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(str(p), map_location=device)

        if "state_dict" in checkpoint:
            self.load_state_dict(checkpoint["state_dict"])
            if "id2label" in checkpoint:
                self.id2label = checkpoint["id2label"]
                self.label2id = {v: k for k, v in self.id2label.items()}
                self.fake_idx, self.real_idx = resolve_label_mapping(self.id2label)
        else:
            # Direct state_dict fallback
            self.load_state_dict(checkpoint)

        print(f"[Model] Successfully loaded weights from {p} on {device}")
        return checkpoint
