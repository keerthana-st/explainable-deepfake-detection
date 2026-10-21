"""
Grad-CAM & Layer-CAM Visual Explainability for EfficientNet-B0 Classifier Agent.

Hooks into the final convolutional layer of EfficientNet-B0 (model.features[-1])
to extract spatial gradient activations, producing:
1. Normalized spatial heatmap showing exact regions contributing to the FAKE verdict.
2. Blended RGB visualization.
3. Bounding box coordinates of peak activation (localizing inpainting / clothing / background edits).
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class EfficientNetGradCAM:
    """
    Grad-CAM explainability extractor for EfficientNet-B0.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()

        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        # Hook into the last convolutional block of EfficientNet features
        target_layer = self.model.features[-1]
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: int = 1,
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Generate Grad-CAM heatmap and bounding box of peak activation.

        Args:
            input_tensor: (1, 3, 224, 224) normalized input image tensor.
            target_class_idx: Class index to explain (default 1 for FAKE).

        Returns:
            heatmap_norm: (224, 224) float32 array in [0.0, 1.0].
            peak_bbox: Dict with ymin, ymax, xmin, xmax.
        """
        self.model.zero_grad()

        # Forward pass
        logits = self.model(input_tensor)
        score = logits[0, target_class_idx]

        # Backward pass
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            # Fallback uniform heatmap if gradients unavailable
            return np.zeros((224, 224), dtype=np.float32), {"ymin": 0, "ymax": 224, "xmin": 0, "xmax": 224}

        # Global average pool the gradients to get channel weights
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, H, W)
        cam = torch.relu(cam)  # Positive influences only

        cam_np = cam.squeeze().cpu().numpy()
        # Resize to 224x224
        cam_resized = cv2.resize(cam_np, (224, 224), interpolation=cv2.INTER_CUBIC)

        # Normalize to [0, 1]
        cam_min, cam_max = np.min(cam_resized), np.max(cam_resized)
        if cam_max > cam_min:
            cam_norm = (cam_resized - cam_min) / (cam_max - cam_min)
        else:
            cam_norm = np.zeros_like(cam_resized)

        # Extract peak bounding box (regions >= 70% peak intensity)
        thresh = (cam_norm >= 0.70).astype(np.uint8)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Take largest contour
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            bbox = {"ymin": int(y), "ymax": int(y + h), "xmin": int(x), "xmax": int(x + w)}
        else:
            bbox = {"ymin": 0, "ymax": 224, "xmin": 0, "xmax": 224}

        return cam_norm.astype(np.float32), bbox

    def overlay_heatmap(
        self,
        original_bgr: np.ndarray,
        cam_norm: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
        output_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """
        Blend Grad-CAM heatmap onto the original image.
        """
        h, w = original_bgr.shape[:2]
        cam_full = cv2.resize(cam_norm, (w, h), interpolation=cv2.INTER_CUBIC)
        heatmap_color = cv2.applyColorMap((cam_full * 255).astype(np.uint8), colormap)

        blended = cv2.addWeighted(original_bgr, 1.0 - alpha, heatmap_color, alpha, 0)

        if output_path is not None:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), blended)

        return blended
