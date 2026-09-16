"""
Heatmap visualization for the Frequency Agent.

Generates visual heatmaps by normalizing 2D high-frequency coefficient energy
maps and blending them with the underlying cropped/resized image at ~50% opacity.
This spatial representation clearly indicates WHERE in the image high-frequency
energy is localized (e.g. upsampling checkerboard artifacts), rather than
relying solely on a single global score.
"""

from pathlib import Path
from typing import Optional, Union
import cv2
import numpy as np


def generate_frequency_heatmap(
    image: np.ndarray,
    coefficient_map: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
    output_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """
    Generate an explainable heatmap visualization overlaying high-frequency energy.

    Parameters
    ----------
    image : np.ndarray
        Original input image (grayscale or color BGR/RGB).
    coefficient_map : np.ndarray
        2D high-frequency energy coefficient map with shape (H_crop, W_crop).
    alpha : float, default=0.5
        Opacity weight of the colored heatmap overlay (0.0 to 1.0).
    colormap : int, default=cv2.COLORMAP_JET
        OpenCV colormap identifier (e.g. cv2.COLORMAP_JET, cv2.COLORMAP_TURBO).
    output_path : Optional[Union[str, Path]], default=None
        If provided, saves the blended image to disk.

    Returns
    -------
    np.ndarray
        Blended BGR heatmap image of shape (H_crop, W_crop, 3).
    """
    if coefficient_map is None or coefficient_map.size == 0:
        raise ValueError("Coefficient map cannot be empty or None.")

    target_h, target_w = coefficient_map.shape[:2]

    # Convert single-channel or grayscale image to 3-channel BGR for blending
    if len(image.shape) == 2:
        base_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        base_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        base_img = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        base_img = image.copy()

    # Align base image dimensions to match coefficient map (H_crop, W_crop)
    h_orig, w_orig = base_img.shape[:2]
    if h_orig >= target_h and w_orig >= target_w:
        # Crop to matching pixel bounds (avoids interpolation distortion)
        base_matched = base_img[:target_h, :target_w]
    else:
        base_matched = cv2.resize(
            base_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR
        )

    # Normalize coefficient map to [0, 255]
    map_min = float(np.min(coefficient_map))
    map_max = float(np.max(coefficient_map))
    if map_max > map_min:
        norm_map = np.uint8(255.0 * (coefficient_map - map_min) / (map_max - map_min))
    else:
        norm_map = np.zeros((target_h, target_w), dtype=np.uint8)

    # Apply colormap to normalized energy values
    heatmap_colored = cv2.applyColorMap(norm_map, colormap)

    # Blend original image and heatmap at ~50% opacity
    blended = cv2.addWeighted(
        src1=base_matched,
        alpha=(1.0 - alpha),
        src2=heatmap_colored,
        beta=alpha,
        gamma=0.0,
    )

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), blended)

    return blended
