"""
Grad-CAM (Gradient-Weighted Class Activation Mapping) Explainability for EfficientNet-B0.

Produces visual attribution heatmaps highlighting the spatial regions that
most strongly activated the deepfake detection model.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn


class EfficientNetGradCAM:
    """
    Grad-CAM implementation hooked to EfficientNet-B0 final convolutional layer.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()

        # EfficientNet-B0 final convolutional block is features[8]
        if target_layer is not None:
            self.target_layer = target_layer
        elif hasattr(model, "features"):
            self.target_layer = model.features[-1]
        elif hasattr(model, "backbone") and hasattr(model.backbone, "features"):
            self.target_layer = model.backbone.features[-1]
        else:
            raise ValueError("Target layer could not be automatically resolved.")

        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        self._hook_handles = [
            self.target_layer.register_forward_hook(self._forward_hook),
        ]

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()
        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad.detach()

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: int = 1,  # Default to FAKE
    ) -> Tuple[np.ndarray, Optional[dict]]:
        """
        Generate normalized Grad-CAM saliency map and peak bounding box.

        Args:
            input_tensor: Shape (1, 3, 224, 224)
            target_class_idx: Class index to explain (0: REAL, 1: FAKE)

        Returns:
            cam_norm: 2D array of shape (224, 224) normalized to [0, 1]
            peak_bbox: Dictionary {ymin, ymax, xmin, xmax} or None
        """
        # Ensure target layer parameters temporarily require grad so backward passes through
        orig_grad_states = [p.requires_grad for p in self.target_layer.parameters()]
        for p in self.target_layer.parameters():
            p.requires_grad = True

        self.gradients = None
        self.activations = None

        try:
            self.model.zero_grad()
            inp = input_tensor.clone().detach().requires_grad_(True)
            logits = self.model(inp)
            score = logits[0, target_class_idx]
            score.backward()
        finally:
            for p, state in zip(self.target_layer.parameters(), orig_grad_states):
                p.requires_grad = state

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations were not captured. Check hooks.")

        # Channel-wise pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, H, W)
        cam = torch.relu(cam)

        cam_np = cam.squeeze().cpu().numpy()
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam_resized = cv2.resize(cam_np, (w, h), interpolation=cv2.INTER_CUBIC)

        max_val = np.max(cam_resized)
        min_val = np.min(cam_resized)
        if max_val > min_val:
            cam_norm = (cam_resized - min_val) / (max_val - min_val)
        else:
            cam_norm = np.zeros_like(cam_resized)

        # Extract peak bounding box (top 20% activation area)
        peak_bbox = None
        thresh = (cam_norm >= 0.80).astype(np.uint8)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(c)
            peak_bbox = {"ymin": int(by), "ymax": int(by + bh), "xmin": int(bx), "xmax": int(bx + bw)}

        return cam_norm, peak_bbox

    def generate_overlay(
        self,
        original_bgr: np.ndarray,
        cam_norm: np.ndarray,
        alpha: float = 0.55,
        colormap: int = cv2.COLORMAP_JET,
        output_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """
        Create a color-blended heatmap overlay onto the original photographic image.
        """
        h, w = original_bgr.shape[:2]
        cam_resized = cv2.resize(cam_norm, (w, h), interpolation=cv2.INTER_CUBIC)
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)

        blended = cv2.addWeighted(original_bgr, 1.0 - alpha, heatmap_colored, alpha, 0)

        # Add explainability banner
        disclaimer = "Grad-CAM Attribution (Saliency Evidence, not absolute proof)"
        cv2.putText(blended, disclaimer, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        if output_path is not None:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), blended)
            print(f"[Grad-CAM] Saved overlay heatmap to: {p}")

        return blended

    def remove_hooks(self):
        """Clean up hooks to prevent memory retention."""
        for h in self._hook_handles:
            h.remove()
