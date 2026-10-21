"""
Common Image Preprocessing Module for Explainable Deepfake Detection.

This module acts as the single decoupled ingestion pipeline for the system:
1. Validates inputs (JPG, JPEG, PNG) and catches corrupted/truncated data gracefully.
2. Decodes images safely to true RGB without silent channel inversion.
3. Computes objective quality and sharpness metrics.
4. Performs aspect-ratio-preserving resizing for computational consistency without destroying forensic cues.
5. Generates three decoupled representations:
   - Path 1: Intact raw bytes & original file for Metadata Agent (C2PA/EXIF/DQT preservation).
   - Path 2: Processed RGB image & normalized float32 tensor for Classifier Agent (EfficientNet-B0).
   - Path 3: Processed Grayscale image for Frequency Agent (8x8 2D-DCT spectral energy).
6. Persists artifacts to outputs/preprocessing/ with unique collision-free naming.
7. Renders an explainable terminal preprocessing audit report.
"""

from dataclasses import dataclass, field, asdict
import io
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import cv2
import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

# Supported image formats
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SUPPORTED_FORMATS = {"JPEG", "JPG", "PNG"}

# ImageNet normalization constants (standard for PyTorch EfficientNet)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class PreprocessingConfig:
    """Configuration settings for the image preprocessing pipeline."""
    target_size: Optional[Tuple[int, int]] = (512, 512)
    resize_method: str = "aspect_ratio_preserve"  # "aspect_ratio_preserve", "pad_square", "direct", "none"
    normalization_method: str = "imagenet"         # "imagenet", "zero_to_one", "minus_one_to_one", "none"
    output_dir: Optional[Union[str, Path]] = "outputs/preprocessing"
    save_artifacts: bool = True
    generate_unique_id: bool = True


@dataclass
class PreprocessingResult:
    """Structured container holding all preprocessed outputs and forensic representations."""
    status: str                                     # "success" or "failed"
    original_path: Optional[str] = None
    raw_bytes: Optional[bytes] = None
    original_image: Optional[np.ndarray] = None     # Unmodified RGB array
    rgb_image: Optional[np.ndarray] = None          # Processed RGB array (H, W, 3)
    grayscale_image: Optional[np.ndarray] = None    # Processed Grayscale array (H, W)
    normalized_image: Optional[np.ndarray] = None   # Normalized float32 array
    saved_paths: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    @property
    def is_success(self) -> bool:
        return self.status == "success"

    def format_report(self) -> str:
        """Constructs the exact explainable preprocessing report for console output."""
        m = self.metadata
        bar = "=" * 50
        if not self.is_success:
            return (
                f"{bar}\n"
                "IMAGE PREPROCESSING MODULE\n"
                f"{bar}\n\n"
                f"Input Image:\n"
                f"Filename: {m.get('filename', 'Unknown')}\n"
                f"Format: {m.get('format', 'UNKNOWN / CORRUPTED')}\n\n"
                "Preprocessing Status:\n"
                "FAILED\n\n"
                "Reason:\n"
                f"{self.error_message or 'Corrupted image or unsupported format.'}\n\n"
                f"{bar}\n"
                "ABORTED: CANNOT DISPATCH TO AGENTS\n"
                f"{bar}"
            )

        orig_w = m.get("original_width", 0)
        orig_h = m.get("original_height", 0)
        proc_w = m.get("processed_width", 0)
        proc_h = m.get("processed_height", 0)
        method = m.get("resize_method", "aspect_ratio_preserve")

        resize_text = (
            f"{orig_w} x {orig_h} px (Unchanged)"
            if (orig_w == proc_w and orig_h == proc_h)
            else f"{orig_w} x {orig_h} -> {proc_w} x {proc_h} px ({method.replace('_', ' ').title()})"
        )

        norm_status = m.get("normalization_status", "SUCCESS")
        raw_size = len(self.raw_bytes) if self.raw_bytes else 0

        return (
            f"{bar}\n"
            "IMAGE PREPROCESSING MODULE\n"
            f"{bar}\n\n"
            "Input Image:\n"
            f"Filename: {m.get('filename', 'image')}\n"
            f"Format: {m.get('format', 'UNKNOWN')}\n\n"
            "Original Resolution:\n"
            f"{orig_w} x {orig_h}\n\n"
            "Original Channels:\n"
            f"{m.get('channels', 3)}\n\n"
            "RGB Processing:\n"
            "SUCCESS\n\n"
            "Grayscale Conversion:\n"
            "SUCCESS\n\n"
            "Resize:\n"
            f"{resize_text}\n\n"
            "Normalization:\n"
            f"{norm_status}\n\n"
            "Metadata Preservation:\n"
            "SUCCESS\n\n"
            "Raw Bytes:\n"
            f"AVAILABLE ({raw_size:,} bytes)\n\n"
            "Preprocessing Status:\n"
            "SUCCESSFUL\n\n"
            f"{bar}\n"
            "READY FOR MULTI-AGENT ANALYSIS\n"
            f"{bar}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the result to a clean dictionary matching section 13 interface."""
        return {
            "status": self.status,
            "original_path": self.original_path,
            "raw_bytes": self.raw_bytes,
            "rgb_image": self.rgb_image,
            "grayscale_image": self.grayscale_image,
            "normalized_image": self.normalized_image,
            "saved_paths": self.saved_paths,
            "metadata": self.metadata,
            "error_message": self.error_message,
        }


class ImagePreprocessor:
    """
    Common Image Preprocessor executing validation, forensic isolation,
    aspect-preserving resizing, and multi-domain representation synthesis.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        if self.config.output_dir:
            self._ensure_output_dirs(Path(self.config.output_dir))

    def _ensure_output_dirs(self, base_dir: Path):
        """Creates the required organized output directory hierarchy."""
        subdirs = ["original", "rgb", "grayscale", "normalized", "metadata"]
        for s in subdirs:
            (base_dir / s).mkdir(parents=True, exist_ok=True)

    def preprocess(
        self,
        image_input: Union[str, Path, bytes, io.BytesIO, np.ndarray],
        filename: Optional[str] = None,
    ) -> PreprocessingResult:
        """
        Executes the end-to-end preprocessing pipeline on any supported input.

        Args:
            image_input: File path, byte sequence, BytesIO, or NumPy array.
            filename: Optional source filename override for reporting.

        Returns:
            PreprocessingResult containing all representations or failure metadata.
        """
        raw_bytes: Optional[bytes] = None
        orig_path_str: Optional[str] = None
        resolved_filename = filename or "image"

        # ---------------------------------------------------------------------
        # STEP 1: INPUT VALIDATION & RAW BYTES ISOLATION
        # ---------------------------------------------------------------------
        try:
            if isinstance(image_input, (str, Path)):
                file_path = Path(image_input)
                resolved_filename = filename or file_path.name
                orig_path_str = str(file_path.resolve()) if file_path.exists() else str(file_path)

                if not file_path.exists():
                    return self._failure(
                        f"Image file does not exist at path: {file_path}",
                        filename=resolved_filename,
                        path=orig_path_str,
                    )

                ext = file_path.suffix.lower()
                if ext not in SUPPORTED_EXTENSIONS:
                    return self._failure(
                        f"Unsupported file format '{ext}'. Must be one of: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
                        filename=resolved_filename,
                        path=orig_path_str,
                    )

                with open(file_path, "rb") as f:
                    raw_bytes = f.read()

                if len(raw_bytes) == 0:
                    return self._failure(
                        "File is completely empty (0 bytes).",
                        filename=resolved_filename,
                        path=orig_path_str,
                    )

            elif isinstance(image_input, bytes):
                raw_bytes = image_input
                resolved_filename = filename or "byte_stream.bin"
                if len(raw_bytes) == 0:
                    return self._failure(
                        "Empty byte sequence provided.",
                        filename=resolved_filename,
                    )

            elif isinstance(image_input, io.BytesIO):
                raw_bytes = image_input.getvalue()
                resolved_filename = filename or "buffer_stream.bin"
                if len(raw_bytes) == 0:
                    return self._failure(
                        "Empty BytesIO stream provided.",
                        filename=resolved_filename,
                    )

            elif isinstance(image_input, np.ndarray):
                if image_input.size == 0:
                    return self._failure(
                        "Empty NumPy image array provided.",
                        filename=resolved_filename,
                    )
                # Encode numpy array to lossless PNG bytes to guarantee raw_bytes path
                resolved_filename = filename or "numpy_input.png"
                success, encoded = cv2.imencode(".png", cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR) if len(image_input.shape) == 3 else image_input)
                if not success:
                    return self._failure(
                        "Failed to serialize input NumPy array to raw bytes.",
                        filename=resolved_filename,
                    )
                raw_bytes = encoded.tobytes()

            else:
                return self._failure(
                    f"Unsupported input type '{type(image_input).__name__}'.",
                    filename=resolved_filename,
                )

        except Exception as e:
            return self._failure(f"Error during input validation: {str(e)}", filename=resolved_filename)

        # ---------------------------------------------------------------------
        # STEP 2: IMAGE DECODING & CHANNEL VERIFICATION
        # ---------------------------------------------------------------------
        try:
            # Pillow verification to ensure file is non-corrupt
            pil_buf = io.BytesIO(raw_bytes)
            try:
                with Image.open(pil_buf) as pil_probe:
                    pil_probe.verify()  # Verifies integrity without decoding entire stream
            except Exception as e:
                return self._failure(
                    f"Corrupted or invalid image stream: {str(e)}",
                    filename=resolved_filename,
                    raw_bytes=raw_bytes,
                    path=orig_path_str,
                )

            # Re-open after verify() (per Pillow requirement)
            pil_buf.seek(0)
            with Image.open(pil_buf) as pil_img:
                img_format = pil_img.format or "UNKNOWN"
                if img_format.upper() not in SUPPORTED_FORMATS:
                    return self._failure(
                        f"Unsupported decoded image format '{img_format}'.",
                        filename=resolved_filename,
                        raw_bytes=raw_bytes,
                        path=orig_path_str,
                    )

                # Handle EXIF orientation if present, but DO NOT modify raw_bytes
                pil_img = ImageOps.exif_transpose(pil_img)

                # Safely convert color modes to strict RGB (preventing silent channel inversion)
                if pil_img.mode == "RGBA":
                    # Alpha blend onto a neutral pure white background
                    bg = Image.new("RGB", pil_img.size, (255, 255, 255))
                    bg.paste(pil_img, mask=pil_img.split()[3])
                    pil_rgb = bg
                elif pil_img.mode != "RGB":
                    pil_rgb = pil_img.convert("RGB")
                else:
                    pil_rgb = pil_img.copy()

                orig_width, orig_height = pil_rgb.size
                orig_rgb_arr = np.array(pil_rgb, dtype=np.uint8)

        except Exception as e:
            return self._failure(
                f"Decoding failed: {str(e)}",
                filename=resolved_filename,
                raw_bytes=raw_bytes,
                path=orig_path_str,
            )

        # ---------------------------------------------------------------------
        # STEP 3: IMAGE QUALITY CHECK & FORENSIC METRICS
        # ---------------------------------------------------------------------
        try:
            # Compute Laplacian variance for blur/sharpness estimation
            gray_scratch = cv2.cvtColor(orig_rgb_arr, cv2.COLOR_RGB2GRAY)
            laplacian_var = float(cv2.Laplacian(gray_scratch, cv2.CV_64F).var())

            # Mean brightness and contrast
            mean_brightness = float(np.mean(gray_scratch))
            contrast_std = float(np.std(gray_scratch))

            # Aspect ratio
            aspect_ratio = float(orig_width / max(orig_height, 1))
            aspect_ratio_str = f"{orig_width}:{orig_height}"

            file_size_bytes = len(raw_bytes)
            file_size_formatted = (
                f"{file_size_bytes / (1024 * 1024):.2f} MB"
                if file_size_bytes >= 1024 * 1024
                else f"{file_size_bytes / 1024:.2f} KB"
            )

        except Exception as e:
            logger.warning(f"Quality metric computation non-fatal warning: {e}")
            laplacian_var = 0.0
            mean_brightness = 0.0
            contrast_std = 0.0
            aspect_ratio = 1.0
            aspect_ratio_str = "1:1"
            file_size_bytes = len(raw_bytes)
            file_size_formatted = f"{file_size_bytes} B"

        # ---------------------------------------------------------------------
        # STEP 4: RESOLUTION HANDLING & ASPECT-PRESERVING RESIZING
        # ---------------------------------------------------------------------
        try:
            proc_rgb, proc_w, proc_h = self._apply_resizing(
                orig_rgb_arr,
                self.config.target_size,
                self.config.resize_method,
            )
        except Exception as e:
            return self._failure(
                f"Resizing failed: {str(e)}",
                filename=resolved_filename,
                raw_bytes=raw_bytes,
                path=orig_path_str,
            )

        # ---------------------------------------------------------------------
        # STEP 5: COLOR REPRESENTATIONS FOR AGENTS
        # ---------------------------------------------------------------------
        # Path 2 (Classifier): RGB array (H, W, 3)
        # Path 3 (Frequency): Grayscale array (H, W) using ITU-R BT.601 luminance
        proc_gray = cv2.cvtColor(proc_rgb, cv2.COLOR_RGB2GRAY)

        # ---------------------------------------------------------------------
        # STEP 6: NORMALIZATION (FOR CLASSIFIER AGENT)
        # ---------------------------------------------------------------------
        try:
            norm_arr, norm_status_str = self._apply_normalization(
                proc_rgb,
                self.config.normalization_method,
            )
        except Exception as e:
            return self._failure(
                f"Normalization failed: {str(e)}",
                filename=resolved_filename,
                raw_bytes=raw_bytes,
                path=orig_path_str,
            )

        # ---------------------------------------------------------------------
        # STEP 7: ASSEMBLE METADATA OBJECT
        # ---------------------------------------------------------------------
        meta_dict = {
            "filename": resolved_filename,
            "format": img_format,
            "original_width": orig_width,
            "original_height": orig_height,
            "processed_width": proc_w,
            "processed_height": proc_h,
            "channels": 3,
            "color_space": "RGB",
            "file_size_bytes": file_size_bytes,
            "file_size_formatted": file_size_formatted,
            "aspect_ratio": round(aspect_ratio, 4),
            "aspect_ratio_str": aspect_ratio_str,
            "blur_score": round(laplacian_var, 2),
            "mean_brightness": round(mean_brightness, 2),
            "contrast": round(contrast_std, 2),
            "resize_method": self.config.resize_method,
            "normalization": self.config.normalization_method,
            "normalization_status": norm_status_str,
            "preprocessing_status": "SUCCESSFUL",
        }

        # ---------------------------------------------------------------------
        # STEP 8: SAVE ARTIFACTS TO ORGANIZED OUTPUT DIRECTORY
        # ---------------------------------------------------------------------
        saved_paths: Dict[str, str] = {}
        if self.config.save_artifacts and self.config.output_dir:
            try:
                saved_paths = self._save_all_artifacts(
                    resolved_filename=resolved_filename,
                    img_format=img_format,
                    raw_bytes=raw_bytes,
                    orig_rgb=orig_rgb_arr,
                    proc_rgb=proc_rgb,
                    proc_gray=proc_gray,
                    norm_arr=norm_arr,
                    metadata=meta_dict,
                )
            except Exception as e:
                logger.warning(f"Failed to persist some preprocessing artifacts: {e}")

        # ---------------------------------------------------------------------
        # STEP 9: RETURN COMPLETE PREPROCESSING RESULT
        # ---------------------------------------------------------------------
        return PreprocessingResult(
            status="success",
            original_path=orig_path_str,
            raw_bytes=raw_bytes,
            original_image=orig_rgb_arr,
            rgb_image=proc_rgb,
            grayscale_image=proc_gray,
            normalized_image=norm_arr,
            saved_paths=saved_paths,
            metadata=meta_dict,
            error_message=None,
        )

    def _apply_resizing(
        self,
        img_rgb: np.ndarray,
        target_size: Optional[Tuple[int, int]],
        method: str,
    ) -> Tuple[np.ndarray, int, int]:
        """Performs resolution scaling while strictly preserving aspect ratio."""
        h, w = img_rgb.shape[:2]
        if method == "none" or target_size is None:
            return img_rgb, w, h

        target_w, target_h = target_size

        if method == "aspect_ratio_preserve":
            # Scale proportionally so neither dimension exceeds target_size
            scale = min(target_w / w, target_h / h)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))

            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
            resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=interp)
            return resized, new_w, new_h

        elif method == "pad_square":
            # Proportional scale + letterbox padding to exact target square
            scale = min(target_w / w, target_h / h)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
            scaled = cv2.resize(img_rgb, (new_w, new_h), interpolation=interp)

            # Pad evenly
            pad_top = (target_h - new_h) // 2
            pad_bottom = target_h - new_h - pad_top
            pad_left = (target_w - new_w) // 2
            pad_right = target_w - new_w - pad_left

            padded = cv2.copyMakeBorder(
                scaled,
                pad_top, pad_bottom, pad_left, pad_right,
                borderType=cv2.BORDER_CONSTANT,
                value=[0, 0, 0],
            )
            return padded, target_w, target_h

        elif method == "direct":
            # Direct stretch to target_w, target_h (used only if explicitly requested)
            interp = cv2.INTER_AREA if (target_w < w and target_h < h) else cv2.INTER_LINEAR
            resized = cv2.resize(img_rgb, (target_w, target_h), interpolation=interp)
            return resized, target_w, target_h

        else:
            raise ValueError(f"Unknown resize_method '{method}'.")

    def _apply_normalization(
        self,
        img_rgb: np.ndarray,
        method: str,
    ) -> Tuple[np.ndarray, str]:
        """Generates the normalized float32 representation for deep neural networks."""
        float_img = img_rgb.astype(np.float32) / 255.0

        if method == "imagenet":
            # (image - mean) / std
            norm_img = (float_img - IMAGENET_MEAN) / IMAGENET_STD
            status_str = "SUCCESS"

        elif method == "zero_to_one":
            norm_img = float_img
            status_str = "SUCCESS ([0.0, 1.0])"

        elif method == "minus_one_to_one":
            norm_img = (img_rgb.astype(np.float32) / 127.5) - 1.0
            status_str = "SUCCESS ([-1.0, 1.0])"

        elif method == "none":
            norm_img = img_rgb.astype(np.float32)
            status_str = "SKIPPED (None)"

        else:
            raise ValueError(f"Unknown normalization_method '{method}'.")

        return norm_img, status_str

    def _save_all_artifacts(
        self,
        resolved_filename: str,
        img_format: str,
        raw_bytes: bytes,
        orig_rgb: np.ndarray,
        proc_rgb: np.ndarray,
        proc_gray: np.ndarray,
        norm_arr: np.ndarray,
        metadata: Dict[str, Any],
    ) -> Dict[str, str]:
        """Saves decoupled representations to outputs/preprocessing/ subdirectories."""
        base_dir = Path(self.config.output_dir or "outputs/preprocessing")
        stem = Path(resolved_filename).stem
        unique_suffix = f"_{uuid.uuid4().hex[:8]}" if self.config.generate_unique_id else ""
        file_id = f"{stem}{unique_suffix}"

        # 1. Original file / raw bytes (preserve original extension)
        ext = f".{img_format.lower()}" if not Path(resolved_filename).suffix else Path(resolved_filename).suffix
        orig_out = base_dir / "original" / f"{file_id}{ext}"
        with open(orig_out, "wb") as f:
            f.write(raw_bytes)

        # 2. RGB processed image (lossless PNG to prevent re-compression artifacts)
        rgb_out = base_dir / "rgb" / f"{file_id}_rgb.png"
        cv2.imwrite(str(rgb_out), cv2.cvtColor(proc_rgb, cv2.COLOR_RGB2BGR))

        # 3. Grayscale processed image
        gray_out = base_dir / "grayscale" / f"{file_id}_gray.png"
        cv2.imwrite(str(gray_out), proc_gray)

        # 4. Resized preview image (224 x 224)
        resized_dir = base_dir / "resized"
        resized_dir.mkdir(parents=True, exist_ok=True)
        resized_img = cv2.resize(proc_rgb, (224, 224), interpolation=cv2.INTER_AREA)
        resized_out = resized_dir / f"{file_id}_resized.png"
        cv2.imwrite(str(resized_out), cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR))

        # 5. Normalized tensor (saved as NumPy binary array)
        norm_out = base_dir / "normalized" / f"{file_id}_normalized.npy"
        np.save(str(norm_out), norm_arr)

        # 6. Preprocessing metadata JSON
        meta_out = base_dir / "metadata" / f"{file_id}_metadata.json"
        with open(meta_out, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return {
            "original": str(orig_out.resolve()),
            "rgb": str(rgb_out.resolve()),
            "grayscale": str(gray_out.resolve()),
            "resized": str(resized_out.resolve()),
            "normalized": str(norm_out.resolve()),
            "metadata": str(meta_out.resolve()),
        }

    def _failure(
        self,
        error_message: str,
        filename: str = "image",
        raw_bytes: Optional[bytes] = None,
        path: Optional[str] = None,
    ) -> PreprocessingResult:
        """Constructs an explicit failure container that halts agent dispatch."""
        logger.error(f"Image Preprocessing Failed: {error_message}")
        meta = {
            "filename": filename,
            "format": "UNKNOWN / CORRUPTED",
            "preprocessing_status": "FAILED",
            "error": error_message,
        }
        return PreprocessingResult(
            status="failed",
            original_path=path,
            raw_bytes=raw_bytes,
            error_message=error_message,
            metadata=meta,
        )


def preprocess_image(
    image_input: Union[str, Path, bytes, io.BytesIO, np.ndarray],
    target_size: Optional[Tuple[int, int]] = (512, 512),
    resize_method: str = "aspect_ratio_preserve",
    normalization_method: str = "imagenet",
    output_dir: Optional[Union[str, Path]] = "outputs/preprocessing",
    save_artifacts: bool = True,
    filename: Optional[str] = None,
) -> PreprocessingResult:
    """
    Convenience top-level function to preprocess an uploaded image.

    Args:
        image_input: Path, bytes, stream, or NumPy array.
        target_size: Desired (width, height) bounding resolution.
        resize_method: "aspect_ratio_preserve", "pad_square", "direct", or "none".
        normalization_method: "imagenet", "zero_to_one", "minus_one_to_one", or "none".
        output_dir: Directory where artifacts are persisted.
        save_artifacts: Whether to write outputs to disk.
        filename: Optional source filename.

    Returns:
        PreprocessingResult containing all three agent representations.
    """
    cfg = PreprocessingConfig(
        target_size=target_size,
        resize_method=resize_method,
        normalization_method=normalization_method,
        output_dir=output_dir,
        save_artifacts=save_artifacts,
    )
    preprocessor = ImagePreprocessor(config=cfg)
    return preprocessor.preprocess(image_input, filename=filename)
