"""
Visualization Tools for the Secondary System.
Generates:
1. Color-coded Region Bounding Box & Mask Overlay (Red: AI-EDITED, Green: REAL, Yellow: UNCERTAIN).
2. Spatial Regional Suspicion Heatmap (Smooth 2D evidence density across image coordinates).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .models import RegionEvidence, RegionVerdict


# Forensic Color Scheme (BGR for OpenCV, RGB for PIL)
COLOR_MAP_BGR = {
    RegionVerdict.AI_EDITED.value: (40, 40, 220),       # Red
    RegionVerdict.REAL.value: (40, 180, 40),            # Green
    RegionVerdict.UNCERTAIN.value: (40, 210, 240),      # Yellow / Amber
    RegionVerdict.INSUFFICIENT_SIZE.value: (140, 140, 140), # Gray
}

COLOR_MAP_RGB = {
    RegionVerdict.AI_EDITED.value: (220, 40, 40),
    RegionVerdict.REAL.value: (40, 180, 40),
    RegionVerdict.UNCERTAIN.value: (240, 210, 40),
    RegionVerdict.INSUFFICIENT_SIZE.value: (140, 140, 140),
}


def draw_region_overlay(
    np_img: np.ndarray,
    region_evidences: List[RegionEvidence],
    save_path: Optional[Union[str, Path]] = None,
    alpha: float = 0.25,
) -> np.ndarray:
    """
    Draws a color-coded forensic region overlay on the original image.

    Args:
        np_img: Original RGB numpy array (H, W, 3)
        region_evidences: List of evaluated RegionEvidence
        save_path: Optional output path to save PNG
        alpha: Transparency factor for shaded masks/boxes

    Returns:
        Annotated RGB image numpy array
    """
    h, w = np_img.shape[:2]
    overlay_img = np_img.copy()

    # Draw semi-transparent fills first
    for r in region_evidences:
        if r.region_type == "background":
            continue  # Don't shade entire background to keep image clear
        ymin, xmin, ymax, xmax = r.bbox
        col_rgb = COLOR_MAP_RGB.get(r.region_verdict, (200, 200, 200))
        sub_crop = overlay_img[ymin:ymax, xmin:xmax].astype(np.float32)
        colored = np.full_like(sub_crop, col_rgb, dtype=np.float32)
        blended = (1.0 - alpha) * sub_crop + alpha * colored
        overlay_img[ymin:ymax, xmin:xmax] = np.clip(blended, 0, 255).astype(np.uint8)

    # Draw sharp bounding boxes and label badges
    pil_img = Image.fromarray(overlay_img)
    draw = ImageDraw.Draw(pil_img)

    for r in region_evidences:
        ymin, xmin, ymax, xmax = r.bbox
        col_rgb = COLOR_MAP_RGB.get(r.region_verdict, (200, 200, 200))

        # Box border
        border_width = 3 if r.region_type != "background" else 1
        draw.rectangle([xmin, ymin, xmax, ymax], outline=col_rgb, width=border_width)

        # Label badge text
        lbl = f"{r.region_id.upper()} | {r.region_verdict} ({r.fake_probability*100:.0f}%)"
        
        # Position label inside or above box
        lbl_y = max(4, ymin - 20) if ymin >= 24 else ymin + 4
        lbl_x = max(4, xmin + 4)

        # Draw badge background
        bbox_text = draw.textbbox((lbl_x, lbl_y), lbl)
        draw.rectangle(
            [bbox_text[0] - 3, bbox_text[1] - 2, bbox_text[2] + 3, bbox_text[3] + 2],
            fill=(20, 20, 25),
            outline=col_rgb,
            width=1,
        )
        draw.text((lbl_x, lbl_y), lbl, fill=col_rgb)

    result_rgb = np.array(pil_img)

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result_rgb).save(p)

    return result_rgb


def generate_spatial_suspicion_heatmap(
    np_img: np.ndarray,
    region_evidences: List[RegionEvidence],
    save_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """
    Constructs a smooth continuous 2D spatial heatmap of forensic suspicion density.

    Args:
        np_img: Original RGB numpy array (H, W, 3)
        region_evidences: Evaluated regional evidence
        save_path: Optional output path

    Returns:
        Heatmap overlay blended onto the original image (H, W, 3)
    """
    h, w = np_img.shape[:2]
    heat_accum = np.zeros((h, w), dtype=np.float32)

    # Base background score
    bg_ev = next((r for r in region_evidences if r.region_type == "background"), None)
    base_val = bg_ev.combined_score if bg_ev else 0.20
    heat_accum.fill(base_val)

    # Add localized scores
    for r in region_evidences:
        if r.region_type == "background":
            continue
        ymin, xmin, ymax, xmax = r.bbox
        rw = xmax - xmin
        rh = ymax - ymin
        if rw <= 0 or rh <= 0:
            continue

        # Create smooth 2D Gaussian bell curve inside the region box
        y = np.linspace(-2, 2, rh)[:, None]
        x = np.linspace(-2, 2, rw)[None, :]
        gauss = np.exp(-(x**2 + y**2) / 2.0)
        gauss = (gauss - gauss.min()) / (gauss.max() - gauss.min() + 1e-6)

        target_score = r.combined_score
        curr = heat_accum[ymin:ymax, xmin:xmax]
        # Smoothly blend regional peak
        heat_accum[ymin:ymax, xmin:xmax] = np.maximum(curr, target_score * gauss + curr * (1.0 - gauss))

    # Normalize to [0, 255]
    heat_norm = cv2.normalize(heat_accum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    color_map = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
    color_map_rgb = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)

    # Blend 50% with original image
    blended = cv2.addWeighted(np_img, 0.60, color_map_rgb, 0.40, 0)

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(blended).save(p)

    return blended
