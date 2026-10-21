"""
Visual Explainability Utilities for Frequency Agent.

Generates:
1. Azimuthal Radial Spectral Falloff Curve (comparing against natural optical 1/f^2 slope)
2. 2D-DCT Log-Power Spectrum with marked Nyquist quadrant boundaries
"""

from pathlib import Path
from typing import Optional, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_radial_spectral_decay(
    radial_profile: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
    image_label: str = "Target Image",
) -> np.ndarray:
    """
    Plot radial power decay curve vs. natural optical 1/f^2 reference line.
    """
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    bins = np.arange(len(radial_profile))

    # Actual spectral profile
    ax.plot(bins, radial_profile, marker="o", color="crimson", lw=2, label=f"{image_label} (Measured)")

    # Natural camera optical reference slope (-2.0 log slope)
    ref_slope = -0.55 * bins + radial_profile[0]
    ax.plot(bins, ref_slope, linestyle="--", color="navy", lw=1.8, label="Natural Optical Reference (1/f² falloff)")

    ax.set_title("Azimuthal Radial Spectral Power Decay", fontsize=11, fontweight="bold")
    ax.set_xlabel("Radial Frequency Band (Low -> High / Nyquist)", fontsize=9)
    ax.set_ylabel("Log Spectral Power", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    bgr = rgba[:, :, [2, 1, 0]]  # RGBA to BGR

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(p), dpi=120)

    plt.close(fig)
    return bgr


def compute_centered_fft_spectrum(gray_img: np.ndarray) -> dict:
    """
    Compute 2D Discrete Fourier Transform (DFT/FFT) with centered zero-frequency,
    log magnitude scaling, radial frequency band statistics (Low, Mid, High),
    and normalized spectral entropy.
    """
    if gray_img is None or gray_img.size == 0:
        raise ValueError("Cannot compute FFT spectrum on empty image.")

    h, w = gray_img.shape[:2]
    img_f = gray_img.astype(np.float32)

    # 1. 2D FFT and zero-frequency center shift
    fft2 = np.fft.fft2(img_f)
    fft_shifted = np.fft.fftshift(fft2)
    magnitude = np.abs(fft_shifted)

    # 2. Log magnitude scaling
    log_magnitude = np.log1p(magnitude)

    # 3. Radial distance grid normalized to [0.0, 1.0]
    cy, cx = h / 2.0, w / 2.0
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    r_max = np.sqrt(cy ** 2 + cx ** 2)
    rho = r / (r_max + 1e-12)

    # 4. Energy statistics across mathematically defined bands:
    # Low: rho < 0.15 (coarse structures, smooth shading)
    # Mid: 0.15 <= rho < 0.50 (edges, standard textures)
    # High: rho >= 0.50 (fine noise, sensor grain, synthetic artifacts)
    power = magnitude ** 2
    total_power = float(np.sum(power)) + 1e-12
    low_power = float(np.sum(power[rho < 0.15]))
    mid_power = float(np.sum(power[(rho >= 0.15) & (rho < 0.50)]))
    high_power = float(np.sum(power[rho >= 0.50]))

    low_pct = float((low_power / total_power) * 100.0)
    mid_pct = float((mid_power / total_power) * 100.0)
    high_pct = float((high_power / total_power) * 100.0)
    energy_ratio_hf = float(high_power / total_power)
    energy_ratio_hl = float(high_power / (low_power + 1e-12))

    # 5. Normalized spectral entropy
    mag_sum = float(np.sum(magnitude)) + 1e-12
    p = (magnitude / mag_sum).flatten()
    p_nz = p[p > 0]
    entropy = float(-np.sum(p_nz * np.log2(p_nz)) / np.log2(float(magnitude.size)))

    # 6. Safe percentile normalization for visualization
    p1 = float(np.percentile(log_magnitude, 1.0))
    p99 = float(np.percentile(log_magnitude, 99.8))
    if p99 > p1:
        norm_spectrum = np.clip((log_magnitude - p1) / (p99 - p1), 0.0, 1.0)
    else:
        norm_spectrum = np.zeros_like(log_magnitude)

    return {
        "magnitude": magnitude,
        "log_magnitude": log_magnitude,
        "norm_spectrum": norm_spectrum,
        "low_pct": low_pct,
        "mid_pct": mid_pct,
        "high_pct": high_pct,
        "energy_ratio_hf": energy_ratio_hf,
        "energy_ratio_hl": energy_ratio_hl,
        "spectral_entropy": entropy,
        "height": h,
        "width": w,
    }


def plot_centered_fft_spectrum(
    fft_data: dict,
    output_path: Optional[Union[str, Path]] = None,
    image_label: str = "Target Spectrum",
) -> np.ndarray:
    """
    Render 2D-FFT centered log-magnitude spectrum with concentric frequency band rings
    (Low, Mid, High) and computed energy percentage annotations.
    """
    norm_spec = fft_data["norm_spectrum"]
    h, w = norm_spec.shape
    cy, cx = h / 2.0, w / 2.0
    r_max = np.sqrt(cy ** 2 + cx ** 2)

    fig, ax = plt.subplots(figsize=(5.5, 5.0), dpi=130)
    im = ax.imshow(norm_spec, cmap="magma", origin="upper")

    # Draw concentric dashed circles for Low / Mid / High boundaries
    circle_low = plt.Circle((cx, cy), 0.15 * r_max, color="cyan", fill=False, linestyle="--", lw=1.2, alpha=0.85, label="Low/Mid Boundary (ρ=0.15)")
    circle_mid = plt.Circle((cx, cy), 0.50 * r_max, color="lime", fill=False, linestyle=":", lw=1.2, alpha=0.85, label="Mid/High Boundary (ρ=0.50)")
    ax.add_patch(circle_low)
    ax.add_patch(circle_mid)

    # Inset label with real computed statistics
    low_pct = fft_data.get("low_pct", 0.0)
    mid_pct = fft_data.get("mid_pct", 0.0)
    high_pct = fft_data.get("high_pct", 0.0)
    entropy = fft_data.get("spectral_entropy", 0.0)

    stats_str = f"Low: {low_pct:.1f}% | Mid: {mid_pct:.2f}% | High: {high_pct:.3f}%\nSpectral Entropy: {entropy:.3f}"
    ax.text(
        0.5,
        0.03,
        stats_str,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="semibold",
        color="white",
        ha="center",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.75, edgecolor="none"),
    )

    ax.set_title(f"2D FFT Log-Magnitude Spectrum ({w}×{h})", fontsize=10, fontweight="bold")
    ax.set_xlabel("Horizontal Frequency (Centered DC)", fontsize=8)
    ax.set_ylabel("Vertical Frequency (Centered DC)", fontsize=8)
    ax.set_xticks([0, int(cx), w - 1])
    ax.set_yticks([0, int(cy), h - 1])
    ax.set_xticklabels(["-Nyq", "DC", "+Nyq"], fontsize=7)
    ax.set_yticklabels(["+Nyq", "DC", "-Nyq"], fontsize=7)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized Log Power", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    plt.tight_layout()

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    bgr = rgba[:, :, [2, 1, 0]]

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(p), dpi=130)

    plt.close(fig)
    return bgr


def plot_annotated_dct_spectrum(
    log_dct: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """
    Render 2D-DCT log spectrum with robust percentile normalization and quadrant dividers.
    """
    p1 = float(np.percentile(log_dct, 1.0))
    p99 = float(np.percentile(log_dct, 99.8))
    if p99 > p1:
        norm_dct = np.clip((log_dct - p1) / (p99 - p1), 0.0, 1.0)
    else:
        norm_dct = np.zeros_like(log_dct)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
    im = ax.imshow(norm_dct, cmap="magma", origin="upper")
    h, w = log_dct.shape

    # Draw quadrant dividers
    ax.axvline(w // 2, color="cyan", linestyle="--", alpha=0.7, lw=1.2)
    ax.axhline(h // 2, color="cyan", linestyle="--", alpha=0.7, lw=1.2)

    # Annotate quadrants
    ax.text(w * 0.25, h * 0.25, "LL\n(Low Freq)", color="white", fontsize=8, ha="center", va="center")
    ax.text(w * 0.75, h * 0.25, "LH\n(Horiz)", color="white", fontsize=8, ha="center", va="center")
    ax.text(w * 0.25, h * 0.75, "HL\n(Vert)", color="white", fontsize=8, ha="center", va="center")
    ax.text(w * 0.75, h * 0.75, "HH\n(High/Diag)", color="white", fontsize=8, ha="center", va="center")

    ax.set_title("2D-DCT Log-Power Spectrum", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    bgr = rgba[:, :, [2, 1, 0]]

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(p), dpi=120)

    plt.close(fig)
    return bgr
