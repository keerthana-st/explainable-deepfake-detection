"""
Frequency Agent (Module 1) - Explainable Deepfake Detection.

Performs multi-scale frequency-domain analysis:
1. Classical 2D-DCT 8x8 block-level energy mapping (spatial checkerboard detection).
2. 28-Dimensional explainable frequency feature extraction (azimuthal radial decay,
   quadrant energy distribution, harmonic periodic spikes, moments, rolloff, ELA).
3. Cost-Sensitive regularized tabular ML classification with calibrated uncertainty
   and explicit INCONCLUSIVE_FREQUENCY safeguard for compression/texture ambiguity.
4. Generates visual explainability artifacts:
   - 2D-DCT Log-Power Heatmap
   - Radial Spectral Decay Curve vs. Natural Optical 1/f^2 slope
   - Top contributing feature attributions with directional z-scores.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from .visualization import generate_frequency_heatmap
from .extractor import FrequencyFeatureExtractor, FrequencyFeatures
from .ml_classifier import FrequencyMLClassifier
from .explain import (
    plot_radial_spectral_decay,
    plot_annotated_dct_spectrum,
    compute_centered_fft_spectrum,
    plot_centered_fft_spectrum,
)

# =============================================================================
# EMPIRICALLY DERIVED CALIBRATION CONSTANTS
# =============================================================================
DEFAULT_CALIBRATION_DIVISOR = 20.5264
DEFAULT_DECISION_THRESHOLD = 0.5000
DEFAULT_CALIBRATION_DATASET = "Empirical-GAN-Artifact-Benchmark-v1 (100 Real vs 100 GAN-Checkerboard)"
DEFAULT_CALIBRATION_DATE = "2026-09-10"

KNOWN_FALSE_POSITIVE_SOURCES: List[str] = [
    "JPEG compression artifacts (common in shared, social-media, or re-saved photos)",
    "Fine natural textures (hair, dense fabrics, skin pores, grass, foliage)",
    "Edge sharpening filters applied automatically by camera sensors or phone software",
    "Image resizing, bicubic resampling, or downscaling before upload",
]

DIFFUSION_BLIND_SPOT_DISCLOSURE: str = (
    "KNOWN BLIND SPOT: This agent is substantially less reliable against modern "
    "latent diffusion models (e.g., DALL-E 3, Midjourney v6, Stable Diffusion XL), "
    "which decode smoothly through variational autoencoders and do not produce the "
    "periodic checkerboard artifacts characteristic of transposed-convolution GAN upsampling. "
    "A low frequency score does NOT confirm an image is real when diffusion generation is suspected."
)


@dataclass
class FrequencyAnalysisResult:
    """Structured output container for Frequency Agent analysis."""

    frequency_score: float
    raw_score: float
    is_suspicious: bool
    coefficient_map: np.ndarray
    cropped_shape: Tuple[int, int]
    calibration_metadata: Dict[str, Any]
    reasoning: str
    false_positive_risks: List[str]
    blind_spot_warning: str
    original_image: Optional[np.ndarray] = None

    # Enhanced Data-Driven & Explainability Fields
    features_28d: Optional[Dict[str, float]] = None
    ml_verdict: Optional[str] = None
    ml_probability: Optional[float] = None
    ml_attributions: Optional[List[Dict[str, Any]]] = None
    radial_profile: Optional[np.ndarray] = None
    log_spectrum: Optional[np.ndarray] = None
    fft_data: Optional[Dict[str, Any]] = None

    def get_heatmap(
        self,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
        output_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """Generate blended heatmap showing where high-frequency energy is concentrated."""
        if self.original_image is None:
            raise ValueError(
                "Original image is not attached to this result. Provide an image "
                "or use generate_frequency_heatmap(image, result.coefficient_map)."
            )
        return generate_frequency_heatmap(
            image=self.original_image,
            coefficient_map=self.coefficient_map,
            alpha=alpha,
            colormap=colormap,
            output_path=output_path,
        )

    def get_radial_plot(
        self,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Optional[np.ndarray]:
        """Generate radial spectral falloff curve vs natural optical slope."""
        if self.radial_profile is None:
            return None
        return plot_radial_spectral_decay(self.radial_profile, output_path=output_path)

    def get_spectrum_plot(
        self,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Optional[np.ndarray]:
        """Generate annotated 2D frequency spectrum plot."""
        if self.fft_data is not None:
            return plot_centered_fft_spectrum(self.fft_data, output_path=output_path)
        if self.log_spectrum is not None:
            return plot_annotated_dct_spectrum(self.log_spectrum, output_path=output_path)
        return None

    def format_console_output(self) -> str:
        """
        Formats clean, classical DCT-based Frequency Agent report for Command Prompt display.
        Focuses strictly on classical DCT frequency analysis, scores, thresholds, resolution,
        and explainable reasoning without tabular ML details.
        """
        verdict_str = "AUTHENTIC / NATURAL" if not self.is_suspicious else "SUSPICIOUS / MANIPULATED"
        sep_thresh = self.calibration_metadata.get("threshold", DEFAULT_DECISION_THRESHOLD)

        lines = [
            "=" * 60,
            "MODULE 1: FREQUENCY AGENT",
            "=" * 60,
            "",
            f"• Classical Verdict: {verdict_str}",
            f"• Frequency Score: {self.frequency_score:.4f} (Range [0.0, 1.0])",
            f"• Raw Energy Score: {self.raw_score:.4f}",
            f"• Separation Threshold: {sep_thresh:.2f}",
            f"• Cropped Resolution: {self.cropped_shape[0]} x {self.cropped_shape[1]} px",
            "",
            "Explainable Reasoning:",
            self.reasoning,
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result summary to a serializable dictionary."""
        out = {
            "frequency_score": round(self.frequency_score, 4),
            "raw_score": round(self.raw_score, 4),
            "is_suspicious": self.is_suspicious,
            "cropped_shape": list(self.cropped_shape),
            "calibration_metadata": self.calibration_metadata,
            "reasoning": self.reasoning,
            "false_positive_risks": self.false_positive_risks,
            "blind_spot_warning": self.blind_spot_warning,
        }
        if self.fft_data is not None:
            out["low_freq_energy_pct"] = round(self.fft_data.get("low_pct", 0.0), 2)
            out["mid_freq_energy_pct"] = round(self.fft_data.get("mid_pct", 0.0), 2)
            out["high_freq_energy_pct"] = round(self.fft_data.get("high_pct", 0.0), 4)
            out["energy_ratio_hf"] = round(self.fft_data.get("energy_ratio_hf", 0.0), 6)
            out["energy_ratio_hl"] = round(self.fft_data.get("energy_ratio_hl", 0.0), 6)
            out["spectral_entropy"] = round(self.fft_data.get("spectral_entropy", 0.0), 4)
        if self.ml_verdict is not None:
            out["ml_verdict"] = self.ml_verdict
        if self.ml_probability is not None:
            out["ml_probability"] = round(self.ml_probability, 4)
        if self.ml_attributions is not None:
            out["ml_attributions"] = self.ml_attributions
        return out


class FrequencyAgent:
    """
    Enhanced Frequency Agent combining classical 2D-DCT signal processing with
    a 28-dimensional explainable tabular ML classifier.
    """

    def __init__(
        self,
        calibration_divisor: float = DEFAULT_CALIBRATION_DIVISOR,
        decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
        calibration_dataset: str = DEFAULT_CALIBRATION_DATASET,
        calibration_date: str = DEFAULT_CALIBRATION_DATE,
        ml_model_path: Optional[Union[str, Path]] = None,
    ):
        if calibration_divisor <= 0.0:
            raise ValueError("calibration_divisor must be strictly positive.")
        if not (0.0 <= decision_threshold <= 1.0):
            raise ValueError("decision_threshold must reside within [0.0, 1.0].")

        self.calibration_divisor = float(calibration_divisor)
        self.decision_threshold = float(decision_threshold)
        self.calibration_dataset = calibration_dataset
        self.calibration_date = calibration_date

        self.extractor = FrequencyFeatureExtractor(target_size=512)
        self.ml_classifier = FrequencyMLClassifier()

        # Load trained ML model if available
        base_dir = Path(__file__).resolve().parent.parent.parent
        default_model = base_dir / "data" / "frequency_ml_model.json"
        target_path = Path(ml_model_path) if ml_model_path else default_model

        if target_path.exists():
            try:
                self.ml_classifier.load_model(target_path)
            except Exception as e:
                # Keep uninitialized fallback
                pass

    def _load_and_validate_image(self, image_input: Union[str, Path, np.ndarray]) -> np.ndarray:
        if image_input is None:
            raise ValueError("Input image cannot be None.")
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise ValueError(f"Image file does not exist at path: {path}")
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"Failed to read image at: {path}")
            return img
        elif isinstance(image_input, np.ndarray):
            if image_input.size == 0:
                raise ValueError("Received empty image array.")
            return image_input
        else:
            raise TypeError(f"Expected path or np.ndarray, got {type(image_input)}")

    def analyze(
        self,
        image_input: Union[str, Path, np.ndarray],
        keep_original: bool = True,
    ) -> FrequencyAnalysisResult:
        """
        Executes full multi-scale frequency analysis on the target image.
        """
        img = self._load_and_validate_image(image_input)

        # 1. Grayscale conversion
        if len(img.shape) == 2:
            gray = img.copy()
        elif len(img.shape) == 3 and img.shape[2] == 1:
            gray = img[:, :, 0].copy()
        elif len(img.shape) == 3 and img.shape[2] >= 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Unsupported image channel shape: {img.shape}")

        # 2. Crop to nearest multiple of 8 pixels for 8x8 block DCT
        h, w = gray.shape[:2]
        h_crop = h - (h % 8)
        w_crop = w - (w % 8)

        if h_crop < 8 or w_crop < 8:
            raise ValueError(f"Cropped image dimensions ({h_crop}x{w_crop}) are too small for 8x8 DCT.")

        cropped_gray = gray[:h_crop, :w_crop]

        # 3. 8x8 Block 2D-DCT & Full-Resolution Coefficient Map Construction
        num_blocks_r = h_crop // 8
        num_blocks_c = w_crop // 8

        coefficient_map = np.zeros((h_crop, w_crop), dtype=np.float32)
        block_energies = np.zeros((num_blocks_r, num_blocks_c), dtype=np.float32)

        for r in range(num_blocks_r):
            r_start = r * 8
            r_end = r_start + 8
            for c in range(num_blocks_c):
                c_start = c * 8
                c_end = c_start + 8

                block = cropped_gray[r_start:r_end, c_start:c_end].astype(np.float32)
                block_dct = cv2.dct(block)
                hf_quadrant = block_dct[4:, 4:]
                energy = float(np.sum(np.abs(hf_quadrant)))

                block_energies[r, c] = energy
                coefficient_map[r_start:r_end, c_start:c_end] = energy

        raw_score = float(np.mean(block_energies))
        frequency_score = float(np.clip(raw_score / self.calibration_divisor, 0.0, 1.0))
        is_suspicious = bool(frequency_score >= self.decision_threshold)

        # 4. Extract 28-D Frequency Features & Run ML Classifier
        features_28d_dict = None
        ml_verdict = None
        ml_prob = None
        ml_attributions = None
        radial_profile = None
        log_spectrum = None

        try:
            bgr_img = img if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            feat_obj = self.extractor.extract_from_image(bgr_img)
            features_28d_dict = feat_obj.to_dict()
            radial_profile = feat_obj.radial_profile
            log_spectrum = feat_obj.dct_spectrum

            ml_exp = self.ml_classifier.explain(feat_obj.vector)
            ml_verdict = ml_exp["verdict"]
            ml_prob = ml_exp["probability"]
            ml_attributions = ml_exp["attributions"]
        except Exception:
            pass

        # Cross-validate classical energy with 28-D ML spectral analysis:
        # If 28-D ML confirms authentic 1/f^2 spectral decay without periodic GAN lattice spikes,
        # and high-frequency diagonal quadrant energy is minimal (quadrant_energy_hh < 5e-5),
        # elevated classical energy is authentic fine texture (hair, fabric, camera DSP edge sharpening).
        quad_hh = features_28d_dict.get("quadrant_energy_hh", 0.0) if features_28d_dict else 0.0
        if (
            ml_verdict == "AUTHENTIC_FREQUENCY"
            and ml_prob is not None
            and ml_prob < 0.20
            and quad_hh < 5e-5
            and frequency_score < 0.65
        ):
            is_suspicious = False
            frequency_score = float(min(frequency_score, 0.35))

        # 4b. Compute 2D Centered FFT Spectrum & Radial Frequency Metrics
        fft_data = None
        try:
            fft_data = compute_centered_fft_spectrum(cropped_gray)
        except Exception:
            pass

        # 5. Explainable reasoning generation
        reasoning, active_risks = self._generate_reasoning(
            raw_score=raw_score,
            frequency_score=frequency_score,
            is_suspicious=is_suspicious,
            ml_verdict=ml_verdict,
            ml_prob=ml_prob,
            ml_attributions=ml_attributions,
            fft_data=fft_data,
        )

        cal_metadata = {
            "divisor": self.calibration_divisor,
            "threshold": self.decision_threshold,
            "calibration_dataset": self.calibration_dataset,
            "calibration_date": self.calibration_date,
        }

        return FrequencyAnalysisResult(
            frequency_score=frequency_score,
            raw_score=raw_score,
            is_suspicious=is_suspicious,
            coefficient_map=coefficient_map,
            cropped_shape=(h_crop, w_crop),
            calibration_metadata=cal_metadata,
            reasoning=reasoning,
            false_positive_risks=active_risks,
            blind_spot_warning=DIFFUSION_BLIND_SPOT_DISCLOSURE,
            original_image=img if keep_original else None,
            features_28d=features_28d_dict,
            ml_verdict=ml_verdict,
            ml_probability=ml_prob,
            ml_attributions=ml_attributions,
            radial_profile=radial_profile,
            log_spectrum=log_spectrum,
            fft_data=fft_data,
        )

    def _generate_reasoning(
        self,
        raw_score: float,
        frequency_score: float,
        is_suspicious: bool,
        ml_verdict: Optional[str] = None,
        ml_prob: Optional[float] = None,
        ml_attributions: Optional[List[Dict[str, Any]]] = None,
        fft_data: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, List[str]]:
        active_risks = list(KNOWN_FALSE_POSITIVE_SOURCES)

        if is_suspicious:
            reasoning = (
                "ELEVATED FREQUENCY ARTIFACTS DETECTED\n\n"
                "The frequency profile exhibits elevated high-frequency energy "
                "consistent with periodic checkerboard artifacts produced by GAN "
                "transposed-convolution upsampling. High-frequency energy across "
                f"8×8 DCT blocks exceeds the calibrated separation threshold ({raw_score:.4f} vs {self.decision_threshold:.2f})."
            )
        else:
            reasoning = (
                "NATURAL FREQUENCY PROFILE\n\n"
                "The frequency profile is consistent with smooth, naturally\n"
                "captured imagery. High-frequency energy across 8×8 DCT\n"
                "blocks is consistent with naturally captured imagery."
            )

        if fft_data:
            low_p = fft_data.get("low_pct", 0.0)
            mid_p = fft_data.get("mid_pct", 0.0)
            high_p = fft_data.get("high_pct", 0.0)
            entropy = fft_data.get("spectral_entropy", 0.0)
            reasoning += (
                f"\n\nSpectral Energy Distribution:\n"
                f"  • Low-Frequency Power (<0.15 Nyquist): {low_p:.1f}%\n"
                f"  • Mid-Frequency Power (0.15-0.50 Nyquist): {mid_p:.2f}%\n"
                f"  • High-Frequency Power (≥0.50 Nyquist): {high_p:.3f}%\n"
                f"  • Spectral Entropy: {entropy:.4f}"
            )

        return reasoning, active_risks
