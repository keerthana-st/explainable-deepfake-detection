"""
28-Dimensional Explainable Frequency Feature Extractor.

Extracts frequency-domain features from image luminance using 2D Discrete Cosine Transform:
1. Azimuthal Radial Power Spectral Density (8 concentric bands)
2. Quadrant Energy Ratios (LL, LH, HL, HH)
3. Harmonic Grid Periodic Spikes (strides 2, 4, 8, 16 - transposed-conv check)
4. Statistical Moments (Spectral Kurtosis and Skewness)
5. Spectral Rolloff (f_85%)
6. ELA-DCT Cross-Divergence (JPEG recompression grid discrepancy)
7. Spatial Patch Spectral Variance (16 disjoint patches for inpainting detection)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from scipy import fftpack


FEATURE_NAMES: List[str] = [
    # 0-7: Radial Power Spectral Density (8 concentric bands)
    "radial_power_band_0",
    "radial_power_band_1",
    "radial_power_band_2",
    "radial_power_band_3",
    "radial_power_band_4",
    "radial_power_band_5",
    "radial_power_band_6",
    "radial_power_band_7",
    # 8-11: Quadrant Energy Ratios
    "quadrant_energy_ll",
    "quadrant_energy_lh",
    "quadrant_energy_hl",
    "quadrant_energy_hh",
    # 12-15: Harmonic Grid Periodic Spikes
    "grid_harmonic_stride_2",
    "grid_harmonic_stride_4",
    "grid_harmonic_stride_8",
    "grid_harmonic_stride_16",
    # 16-17: Statistical Moments
    "spectral_kurtosis",
    "spectral_skewness",
    # 18: Spectral Rolloff
    "spectral_rolloff_85",
    # 19-22: ELA-DCT Cross Divergence (4 quadrant divergences)
    "ela_divergence_ll",
    "ela_divergence_lh",
    "ela_divergence_hl",
    "ela_divergence_hh",
    # 23-27: Patch Spectral Variance (mean, std, min, max, iqr of radial slope)
    "patch_slope_mean",
    "patch_slope_std",
    "patch_slope_min",
    "patch_slope_max",
    "patch_slope_iqr",
]


@dataclass
class FrequencyFeatures:
    """Container for extracted 28-dimensional frequency feature vector."""

    vector: np.ndarray  # Shape: (28,) float32
    feature_names: List[str]
    dct_spectrum: np.ndarray  # Log-magnitude 2D-DCT spectrum
    radial_profile: np.ndarray  # Azimuthal 1D radial decay profile
    patch_slopes: np.ndarray  # Radial slopes per 64x64 sub-patch

    def to_dict(self) -> Dict[str, float]:
        return {name: float(val) for name, val in zip(self.feature_names, self.vector)}


class FrequencyFeatureExtractor:
    """
    Extracts 28 explainable frequency features from image arrays or files.
    """

    def __init__(self, target_size: int = 512):
        self.target_size = target_size

    def _compute_2d_dct(self, gray_img: np.ndarray) -> np.ndarray:
        """Compute 2D Discrete Cosine Transform (ortho-normalized)."""
        return fftpack.dct(fftpack.dct(gray_img.T, norm="ortho").T, norm="ortho")

    def _compute_azimuthal_radial_profile(
        self, log_dct: np.ndarray, num_bins: int = 8
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute azimuthally integrated radial power spectral profile across concentric rings.
        """
        h, w = log_dct.shape
        y, x = np.indices((h, w))
        r = np.sqrt(x**2 + y**2)
        r_max = np.sqrt(h**2 + w**2)

        bin_edges = np.linspace(0, r_max, num_bins + 1)
        radial_profile = np.zeros(num_bins, dtype=np.float32)

        for i in range(num_bins):
            mask = (r >= bin_edges[i]) & (r < bin_edges[i + 1])
            if np.any(mask):
                radial_profile[i] = float(np.mean(log_dct[mask]))
            else:
                radial_profile[i] = 0.0

        return radial_profile, bin_edges

    def _compute_harmonic_spikes(self, dct_mag: np.ndarray) -> List[float]:
        """
        Detect periodic harmonic spikes at sub-sampling strides (2, 4, 8, 16)
        characteristic of GAN transposed-convolution upsampling layers.
        """
        h, w = dct_mag.shape
        spikes = []
        for stride in [2, 4, 8, 16]:
            sub_grid = dct_mag[stride // 2 :: stride, stride // 2 :: stride]
            if sub_grid.size > 0:
                grid_energy = np.mean(sub_grid)
                local_bg = np.median(dct_mag) + 1e-6
                spike_ratio = float(grid_energy / local_bg)
            else:
                spike_ratio = 1.0
            spikes.append(spike_ratio)
        return spikes

    def _compute_ela_divergence(self, img_bgr: np.ndarray, gray: np.ndarray) -> List[float]:
        """
        Compute cross-divergence between direct DCT and Error Level Analysis (ELA) resynthesis.
        """
        # Save to memory at quality 90 and calculate residual
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, encimg = cv2.imencode(".jpg", img_bgr, encode_param)
        resaved = cv2.imdecode(encimg, 1)
        resaved_gray = cv2.cvtColor(resaved, cv2.COLOR_BGR2GRAY).astype(np.float32)
        ela = np.abs(gray.astype(np.float32) - resaved_gray) * 10.0
        ela = np.clip(ela, 0, 255).astype(np.uint8)

        # 2D DCT of ELA residual
        dct_orig = np.abs(self._compute_2d_dct(gray.astype(np.float32)))
        dct_ela = np.abs(self._compute_2d_dct(ela.astype(np.float32)))

        # Quadrant energy divergence
        h, w = gray.shape
        h_half, w_half = h // 2, w // 2

        quads_orig = [
            dct_orig[:h_half, :w_half],
            dct_orig[:h_half, w_half:],
            dct_orig[h_half:, :w_half],
            dct_orig[h_half:, w_half:],
        ]
        quads_ela = [
            dct_ela[:h_half, :w_half],
            dct_ela[:h_half, w_half:],
            dct_ela[h_half:, :w_half],
            dct_ela[h_half:, w_half:],
        ]

        divs = []
        for qo, qe in zip(quads_orig, quads_ela):
            eo = np.mean(qo) + 1e-6
            ee = np.mean(qe) + 1e-6
            divs.append(float(np.abs(np.log(eo / ee))))
        return divs

    def _compute_patch_spectral_variance(self, gray: np.ndarray) -> List[float]:
        """
        Compute radial spectral decay slopes across 16 disjoint sub-patches (4x4 grid).
        High variance flags localized inpainting (e.g., clothing or background edits).
        """
        h, w = gray.shape
        ph, pw = h // 4, w // 4
        slopes = []

        for row in range(4):
            for col in range(4):
                patch = gray[row * ph : (row + 1) * ph, col * pw : (col + 1) * pw]
                dct_p = np.abs(self._compute_2d_dct(patch.astype(np.float32)))
                log_p = np.log1p(dct_p)
                rad, _ = self._compute_azimuthal_radial_profile(log_p, num_bins=6)
                x_bins = np.arange(len(rad))
                if len(rad) >= 2:
                    slope, _ = np.polyfit(x_bins, rad, 1)
                    slopes.append(float(slope))
                else:
                    slopes.append(0.0)

        slopes_arr = np.array(slopes, dtype=np.float32)
        q75, q25 = np.percentile(slopes_arr, [75, 25])
        return [
            float(np.mean(slopes_arr)),
            float(np.std(slopes_arr)),
            float(np.min(slopes_arr)),
            float(np.max(slopes_arr)),
            float(q75 - q25),  # IQR
        ], slopes_arr

    def extract_from_image(self, img_bgr: np.ndarray) -> FrequencyFeatures:
        """
        Extract 28-dimensional frequency feature vector from BGR image array.
        """
        h, w = img_bgr.shape[:2]
        crop_dim = min(h, w, self.target_size)
        # Power of 2 nearest crop
        pot = 2 ** int(np.log2(crop_dim))
        pot = max(pot, 128)

        cy, cx = h // 2, w // 2
        half = pot // 2
        cropped = img_bgr[cy - half : cy + half, cx - half : cx + half]
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        # 1. 2D DCT and Log Magnitude
        dct_coeffs = self._compute_2d_dct(gray.astype(np.float32))
        dct_mag = np.abs(dct_coeffs)
        log_dct = np.log1p(dct_mag)

        # 2. Features 0-7: Radial Power Spectral Density (8 bands)
        radial_profile, _ = self._compute_azimuthal_radial_profile(log_dct, num_bins=8)

        # 3. Features 8-11: Quadrant Energy Ratios
        h_sub, w_sub = gray.shape
        h_mid, w_mid = h_sub // 2, w_sub // 2
        total_e = float(np.sum(dct_mag**2)) + 1e-6
        e_ll = float(np.sum(dct_mag[:h_mid, :w_mid] ** 2)) / total_e
        e_lh = float(np.sum(dct_mag[:h_mid, w_mid:] ** 2)) / total_e
        e_hl = float(np.sum(dct_mag[h_mid:, :w_mid] ** 2)) / total_e
        e_hh = float(np.sum(dct_mag[h_mid:, w_mid:] ** 2)) / total_e
        quadrant_energies = [e_ll, e_lh, e_hl, e_hh]

        # 4. Features 12-15: Harmonic Grid Spikes
        harmonic_spikes = self._compute_harmonic_spikes(dct_mag)

        # 5. Features 16-17: Moments (Kurtosis, Skewness)
        flat_mag = dct_mag.flatten()
        mean_m = np.mean(flat_mag)
        std_m = np.std(flat_mag) + 1e-6
        norm_mag = (flat_mag - mean_m) / std_m
        skewness = float(np.mean(norm_mag**3))
        kurtosis = float(np.mean(norm_mag**4) - 3.0)  # Excess kurtosis

        # 6. Feature 18: Spectral Rolloff (85% energy)
        sorted_energies = np.sort(flat_mag**2)
        cum_e = np.cumsum(sorted_energies)
        cutoff_idx = np.searchsorted(cum_e, 0.85 * cum_e[-1])
        rolloff_85 = float(cutoff_idx / len(sorted_energies))

        # 7. Features 19-22: ELA Divergence
        ela_divs = self._compute_ela_divergence(cropped, gray)

        # 8. Features 23-27: Patch Spectral Variance
        patch_stats, slopes_arr = self._compute_patch_spectral_variance(gray)

        # Combine all into 28-D vector
        vector_elements = (
            list(radial_profile)
            + quadrant_energies
            + harmonic_spikes
            + [kurtosis, skewness, rolloff_85]
            + ela_divs
            + patch_stats
        )

        assert len(vector_elements) == 28, f"Expected 28 features, got {len(vector_elements)}"
        vector = np.array(vector_elements, dtype=np.float32)

        return FrequencyFeatures(
            vector=vector,
            feature_names=FEATURE_NAMES,
            dct_spectrum=log_dct,
            radial_profile=radial_profile,
            patch_slopes=slopes_arr,
        )

    def extract_from_file(self, image_path: Union[str, Path]) -> FrequencyFeatures:
        """Extract frequency features from file path."""
        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"Image not found at {p}")
        img = cv2.imread(str(p))
        if img is None:
            raise ValueError(f"Failed to read image at {p}")
        return self.extract_from_image(img)
