"""
Region Preprocessing Module for the Secondary System.
Ensures that all extracted regions strictly adhere to the exact same normalization,
resizing, and color space transforms that the Classifier Agent was trained with.
"""

from typing import Optional, Tuple
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from .models import Region, RegionVerdict


def get_region_transforms(target_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """
    Standard PyTorch transforms strictly matching the fine-tuned EfficientNet-B0.
    """
    return transforms.Compose([
        transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class RegionPreprocessResult(tuple):
    """Result object supporting both 3-tuple unpacking and named attributes."""
    def __new__(cls, rgb_tensor, gray_array, error_message):
        return super().__new__(cls, (rgb_tensor, gray_array, error_message))

    @property
    def rgb_tensor(self) -> Optional[torch.Tensor]:
        return self[0]

    @property
    def gray_array(self) -> Optional[np.ndarray]:
        return self[1]

    @property
    def error_message(self) -> Optional[str]:
        return self[2]

    @property
    def is_valid(self) -> bool:
        return self[2] is None


class RegionPreprocessor:
    """
    Preprocesses isolated spatial regions for downstream agent evaluation.
    """

    def __init__(self, target_size: Tuple[int, int] = (224, 224), min_dim: int = 16):
        self.target_size = target_size
        self.min_dim = min_dim
        self.transform = get_region_transforms(target_size)

    def preprocess_region(
        self,
        region: Region,
        np_img: np.ndarray,
        apply_mask_to_pixels: bool = False,
    ) -> RegionPreprocessResult:
        """
        Preprocesses a region into matched representations:
        1. RGB Tensor (1, 3, 224, 224) ImageNet-normalized for Classifier Agent.
        2. Grayscale 2D array for Frequency Agent.

        Args:
            region: Region data model
            np_img: Full RGB image array of shape (H, W, 3)
            apply_mask_to_pixels: If True, zeroes out non-mask pixels

        Returns:
            RegionPreprocessResult with (rgb_tensor, gray_array, error_status)
        """
        h_full, w_full = np_img.shape[:2]
        ymin, xmin, ymax, xmax = region.bbox

        # Clamp bounding box
        ymin = max(0, min(ymin, h_full - 1))
        ymax = max(ymin + 1, min(ymax, h_full))
        xmin = max(0, min(xmin, w_full - 1))
        xmax = max(xmin + 1, min(xmax, w_full))

        rw = xmax - xmin
        rh = ymax - ymin

        # Check minimum size requirement (Part 26 error handling)
        if rw < self.min_dim or rh < self.min_dim:
            return RegionPreprocessResult(None, None, RegionVerdict.INSUFFICIENT_SIZE.value)

        # Crop region from image
        cropped_rgb = np_img[ymin:ymax, xmin:xmax].copy()

        # Apply binary mask if requested
        if apply_mask_to_pixels and region.mask is not None:
            region_mask_crop = region.mask[ymin:ymax, xmin:xmax]
            cropped_rgb[~region_mask_crop] = 0

        # Convert to PIL and apply exact ImageNet transforms
        pil_crop = Image.fromarray(cropped_rgb)
        rgb_tensor = self.transform(pil_crop).unsqueeze(0)  # Shape (1, 3, 224, 224)

        # Build grayscale representation for Frequency Agent
        gray_array = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2GRAY)
        gray_array = cv2.resize(gray_array, self.target_size, interpolation=cv2.INTER_AREA)

        return RegionPreprocessResult(rgb_tensor, gray_array, None)



def preprocess_region(
    region: Region,
    np_img: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    apply_mask_to_pixels: bool = False,
    min_dim: int = 16,
) -> RegionPreprocessResult:
    """Convenience functional wrapper for region preprocessing."""
    preprocessor = RegionPreprocessor(target_size=target_size, min_dim=min_dim)
    return preprocessor.preprocess_region(region, np_img, apply_mask_to_pixels=apply_mask_to_pixels)

