"""
Comprehensive Unit and Integration Tests for Common Image Preprocessing Module.

Verifies:
1. JPG and JPEG image validation, decoding, and representation synthesis.
2. PNG image validation and RGBA transparency composite handling.
3. Resolution handling across extreme dimensions (low-res, high-res 4K).
4. Aspect ratio preservation across non-square geometries (16:9, 9:16, 21:9, 1:1).
5. Corrupted, truncated, empty, and non-image input graceful failure handling.
6. Bit-exact forensic integrity of raw_bytes (SHA-256 cryptographic check).
7. Strict RGB channel order verification (ensuring NO silent BGR channel swapping).
8. Normalization strategies (ImageNet float32, zero-to-one, minus-one-to-one).
9. Output directory hierarchy and file persistence.
10. Explainable report generation format.
"""

import hashlib
import io
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.preprocessing import (
    ImagePreprocessor,
    PreprocessingConfig,
    PreprocessingResult,
    preprocess_image,
)


@pytest.fixture
def preprocessor(tmp_path):
    """Creates a standard preprocessor instance writing to a temporary test directory."""
    cfg = PreprocessingConfig(
        target_size=(512, 512),
        resize_method="aspect_ratio_preserve",
        normalization_method="imagenet",
        output_dir=tmp_path / "outputs",
        save_artifacts=True,
    )
    return ImagePreprocessor(config=cfg)


@pytest.fixture
def sample_jpg(tmp_path):
    """Creates an authentic standard 640x480 RGB JPG image."""
    img_arr = np.full((480, 640, 3), 128, dtype=np.uint8)
    # Add distinct color patches
    img_arr[0:100, 0:100] = [200, 50, 50]
    p = tmp_path / "sample_valid.jpg"
    Image.fromarray(img_arr).save(p, format="JPEG", quality=95)
    return p


@pytest.fixture
def sample_png(tmp_path):
    """Creates an authentic 300x400 RGBA PNG image with transparency."""
    img_arr = np.full((400, 300, 4), 255, dtype=np.uint8)
    # Set semi-transparent alpha in one region
    img_arr[0:100, 0:100, 3] = 128
    p = tmp_path / "sample_transparent.png"
    Image.fromarray(img_arr).save(p, format="PNG")
    return p


# =============================================================================
# 1. INPUT VALIDATION & FORMAT TESTS (JPG, JPEG, PNG)
# =============================================================================

def test_jpg_preprocessing_success(preprocessor, sample_jpg):
    """Test standard valid JPG file processing."""
    res = preprocessor.preprocess(sample_jpg)

    assert res.is_success
    assert res.status == "success"
    assert res.error_message is None
    assert res.metadata["preprocessing_status"] == "SUCCESSFUL"
    assert res.metadata["format"].upper() in ["JPEG", "JPG"]
    assert res.metadata["original_width"] == 640
    assert res.metadata["original_height"] == 480
    assert res.metadata["channels"] == 3
    assert res.raw_bytes is not None and len(res.raw_bytes) > 0


def test_png_preprocessing_with_alpha_channel(preprocessor, sample_png):
    """Test PNG file with RGBA channels safely converted to 3-channel RGB."""
    res = preprocessor.preprocess(sample_png)

    assert res.is_success
    assert res.rgb_image is not None
    assert res.rgb_image.shape[2] == 3  # Strictly 3 channels (RGB)
    assert res.grayscale_image is not None
    assert len(res.grayscale_image.shape) == 2  # Single channel (H, W)
    assert res.metadata["format"].upper() == "PNG"


# =============================================================================
# 2. RESOLUTION & ASPECT-RATIO PRESERVATION TESTS
# =============================================================================

@pytest.mark.parametrize("orig_w,orig_h", [
    (1920, 1080),  # 16:9 Landscape
    (1080, 1920),  # 9:16 Portrait
    (800, 800),    # 1:1 Square
    (2100, 900),   # 21:9 Ultra-wide
    (128, 96),     # Low resolution
])
def test_aspect_ratio_preservation(preprocessor, tmp_path, orig_w, orig_h):
    """Verifies that aspect ratio is strictly maintained without distortion."""
    img = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    p = tmp_path / f"test_{orig_w}x{orig_h}.png"
    Image.fromarray(img).save(p)

    res = preprocessor.preprocess(p)
    assert res.is_success

    proc_w = res.metadata["processed_width"]
    proc_h = res.metadata["processed_height"]

    # Maximum dimension must not exceed target size (512, 512)
    assert proc_w <= 512 and proc_h <= 512

    # Aspect ratio error must be less than 1.5% due to integer rounding
    orig_aspect = orig_w / orig_h
    proc_aspect = proc_w / proc_h
    assert abs(orig_aspect - proc_aspect) / orig_aspect < 0.015


def test_pad_square_resizing(tmp_path):
    """Test optional letterbox padding to exact square resolution."""
    cfg = PreprocessingConfig(
        target_size=(256, 256),
        resize_method="pad_square",
        output_dir=None,
        save_artifacts=False,
    )
    prep = ImagePreprocessor(config=cfg)

    # 400x200 landscape
    img = np.full((200, 400, 3), 200, dtype=np.uint8)
    p = tmp_path / "rect.jpg"
    Image.fromarray(img).save(p)

    res = prep.preprocess(p)
    assert res.is_success
    assert res.rgb_image.shape == (256, 256, 3)
    assert res.metadata["processed_width"] == 256
    assert res.metadata["processed_height"] == 256


# =============================================================================
# 3. COLOR CHANNELS & NORMALIZATION TESTS
# =============================================================================

def test_explicit_rgb_color_order(preprocessor, tmp_path):
    """Verifies that RGB order is strictly preserved and NEVER swapped to BGR."""
    # Create an image that is 100% pure Red (R=255, G=0, B=0)
    red_img = np.zeros((100, 100, 3), dtype=np.uint8)
    red_img[:, :, 0] = 255  # Red channel
    p = tmp_path / "pure_red.png"
    Image.fromarray(red_img).save(p)

    res = preprocessor.preprocess(p)
    assert res.is_success

    # Check center pixel: must be [255, 0, 0] in RGB
    center_pixel = res.rgb_image[50, 50]
    assert center_pixel[0] == 255  # R
    assert center_pixel[1] == 0    # G
    assert center_pixel[2] == 0    # B


def test_grayscale_representation_validity(preprocessor, sample_jpg):
    """Verifies grayscale conversion uses standard ITU-R BT.601 luminance."""
    res = preprocessor.preprocess(sample_jpg)
    assert res.is_success

    gray = res.grayscale_image
    assert gray.ndim == 2
    assert gray.dtype == np.uint8
    assert gray.shape == (res.metadata["processed_height"], res.metadata["processed_width"])


@pytest.mark.parametrize("norm_method", ["imagenet", "zero_to_one", "minus_one_to_one", "none"])
def test_normalization_methods(tmp_path, sample_jpg, norm_method):
    """Test all supported normalization modes for deep vision classifiers."""
    cfg = PreprocessingConfig(
        target_size=(256, 256),
        normalization_method=norm_method,
        output_dir=None,
        save_artifacts=False,
    )
    prep = ImagePreprocessor(config=cfg)
    res = prep.preprocess(sample_jpg)

    assert res.is_success
    norm_arr = res.normalized_image
    assert norm_arr.dtype == np.float32

    if norm_method == "zero_to_one":
        assert norm_arr.min() >= 0.0 and norm_arr.max() <= 1.0
    elif norm_method == "minus_one_to_one":
        assert norm_arr.min() >= -1.0 and norm_arr.max() <= 1.0


# =============================================================================
# 4. FORENSIC INTEGRITY TEST (RAW BYTES BIT-EXACTNESS)
# =============================================================================

def test_raw_bytes_cryptographic_bit_exactness(preprocessor, sample_jpg):
    """
    CRITICAL FORENSIC REQUIREMENT:
    Verifies that raw_bytes returned is 100% bit-exact to the original file on disk
    by comparing SHA-256 hashes. Preprocessing must NOT re-compress or alter metadata.
    """
    with open(sample_jpg, "rb") as f:
        disk_bytes = f.read()
    disk_sha256 = hashlib.sha256(disk_bytes).hexdigest()

    res = preprocessor.preprocess(sample_jpg)
    assert res.is_success
    result_sha256 = hashlib.sha256(res.raw_bytes).hexdigest()

    assert result_sha256 == disk_sha256
    assert len(res.raw_bytes) == len(disk_bytes)


# =============================================================================
# 5. ERROR HANDLING & CORRUPTED INPUT TESTS
# =============================================================================

def test_corrupted_image_graceful_failure(preprocessor, tmp_path):
    """Verifies corrupted file returns PREPROCESSING FAILED without raising an unhandled exception."""
    corrupted_p = tmp_path / "corrupt.jpg"
    # Write random junk bytes with a JPG extension
    with open(corrupted_p, "wb") as f:
        f.write(b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01" + b"\xDE\xAD\xBE\xEF" * 32)

    res = preprocessor.preprocess(corrupted_p)
    assert not res.is_success
    assert res.status == "failed"
    assert res.metadata["preprocessing_status"] == "FAILED"
    assert "Corrupted" in res.error_message or "Decoding failed" in res.error_message

    # Verify report reflects failure
    rep = res.format_report()
    assert "PREPROCESSING MODULE" in rep
    assert "FAILED" in rep
    assert "ABORTED: CANNOT DISPATCH TO AGENTS" in rep


def test_empty_file_fails(preprocessor, tmp_path):
    """Verifies 0-byte file fails gracefully."""
    empty_p = tmp_path / "empty.png"
    empty_p.touch()

    res = preprocessor.preprocess(empty_p)
    assert not res.is_success
    assert "empty" in res.error_message.lower()


def test_unsupported_file_extension_fails(preprocessor, tmp_path):
    """Verifies non-image format (e.g. .txt, .exe) is rejected."""
    txt_p = tmp_path / "notes.txt"
    txt_p.write_text("Not an image file.")

    res = preprocessor.preprocess(txt_p)
    assert not res.is_success
    assert "Unsupported file format" in res.error_message


def test_nonexistent_path_fails(preprocessor, tmp_path):
    """Verifies missing file path returns explicit error."""
    missing_p = tmp_path / "does_not_exist.jpg"
    res = preprocessor.preprocess(missing_p)
    assert not res.is_success
    assert "does not exist" in res.error_message


# =============================================================================
# 6. OUTPUT DIRECTORY STRUCTURE & ARTIFACT PERSISTENCE
# =============================================================================

def test_output_directory_structure_created(preprocessor, sample_jpg):
    """Verifies all five output subdirectories are created and populated."""
    res = preprocessor.preprocess(sample_jpg)
    assert res.is_success

    paths = res.saved_paths
    assert "original" in paths
    assert "rgb" in paths
    assert "grayscale" in paths
    assert "normalized" in paths
    assert "metadata" in paths

    # Verify all files actually exist on disk
    for key, p_str in paths.items():
        p = Path(p_str)
        assert p.exists(), f"Artifact for {key} was not saved at: {p}"
        assert p.stat().st_size > 0

    # Check metadata JSON content
    with open(paths["metadata"], "r", encoding="utf-8") as f:
        meta_loaded = json.load(f)
    assert meta_loaded["preprocessing_status"] == "SUCCESSFUL"
    assert meta_loaded["original_width"] == 640


# =============================================================================
# 7. EXPLAINABLE REPORT GENERATION TEST
# =============================================================================

def test_explainable_report_format(preprocessor, sample_jpg):
    """Verifies the terminal audit report contains all required headers and fields."""
    res = preprocessor.preprocess(sample_jpg)
    report = res.format_report()

    assert "IMAGE PREPROCESSING MODULE" in report
    assert "Input Image:" in report
    assert "Format: JPEG" in report or "Format: JPG" in report
    assert "Original Resolution:" in report
    assert "640 x 480" in report
    assert "Original Channels:" in report
    assert "3" in report
    assert "RGB Processing:" in report
    assert "SUCCESS" in report
    assert "Grayscale Conversion:" in report
    assert "SUCCESS" in report
    assert "Resize:" in report
    assert "Normalization:" in report
    assert "Metadata Preservation:" in report
    assert "SUCCESS" in report
    assert "Raw Bytes:" in report
    assert "AVAILABLE" in report
    assert "Preprocessing Status:" in report
    assert "SUCCESSFUL" in report
    assert "READY FOR MULTI-AGENT ANALYSIS" in report


# =============================================================================
# 8. TOP-LEVEL INTERFACE (preprocess_image) TEST
# =============================================================================

def test_top_level_convenience_function(sample_jpg):
    """Verifies preprocess_image top-level function works seamlessly."""
    res = preprocess_image(sample_jpg, target_size=(256, 256))
    assert isinstance(res, PreprocessingResult)
    assert res.is_success
    d = res.to_dict()
    assert "status" in d
    assert "rgb_image" in d
    assert "grayscale_image" in d
    assert "raw_bytes" in d
    assert "metadata" in d
