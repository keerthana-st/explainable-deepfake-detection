"""
Demonstration Script for Frequency Agent (Module 1).

Runs the Frequency Agent on sample authentic and synthetic images,
demonstrates explainable reasoning outputs with false-positive disclosures,
and generates blended high-frequency heatmaps.
"""

from pathlib import Path
import pprint
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from modules.frequency.agent import FrequencyAgent


def create_demo_images(output_dir: Path):
    """Create three test cases for demonstration."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Authentic natural photograph simulation (smooth lighting & gradient)
    x = np.linspace(0, 3 * np.pi, 256)
    y = np.linspace(0, 3 * np.pi, 256)
    xx, yy = np.meshgrid(x, y)
    base = np.sin(xx) * np.cos(yy) * 100.0 + 128.0
    natural_noise = cv2.GaussianBlur(np.random.normal(0, 10, (256, 256)), (9, 9), 0)
    natural_img = np.clip(base + natural_noise, 0, 255).astype(np.uint8)
    p_real = output_dir / "demo_authentic.png"
    cv2.imwrite(str(p_real), cv2.cvtColor(natural_img, cv2.COLOR_GRAY2BGR))

    # 2. GAN-upsampling synthetic fake simulation (periodic transposed-conv checkerboard)
    fake_img = base.copy()
    # Add checkerboard grid pattern with stride 4
    fake_img[::4, ::4] += 35.0
    fake_img[1::4, 1::4] -= 35.0
    fake_img = np.clip(fake_img, 0, 255).astype(np.uint8)
    p_fake = output_dir / "demo_gan_upsampled.png"
    cv2.imwrite(str(p_fake), cv2.cvtColor(fake_img, cv2.COLOR_GRAY2BGR))

    # 3. High-frequency texture / compression case (illustrating false-positive handling)
    textured_img = base.copy()
    fine_texture = np.random.normal(0, 15, (256, 256))
    textured_img = np.clip(textured_img + fine_texture, 0, 255).astype(np.uint8)
    p_texture = output_dir / "demo_textured_photo.jpg"
    # Save as JPEG with high compression
    cv2.imwrite(str(p_texture), cv2.cvtColor(textured_img, cv2.COLOR_GRAY2BGR), [cv2.IMWRITE_JPEG_QUALITY, 50])

    return [
        ("Authentic Natural Image", p_real),
        ("GAN Upsampling Synthetic Fake", p_fake),
        ("Compressed Natural Texture Photo", p_texture),
    ]


def main():
    print("=================================================================")
    print("      FREQUENCY AGENT (MODULE 1) - EXPLAINABILITY DEMO          ")
    print("=================================================================")

    demo_dir = PROJECT_ROOT / "data" / "demo_samples"
    heatmap_dir = PROJECT_ROOT / "data" / "demo_heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    samples = create_demo_images(demo_dir)
    agent = FrequencyAgent()

    for label, path in samples:
        print(f"\n>>> Analyzing Test Image: [{label}] ({path.name})")
        result = agent.analyze(path)

        print(f"  • Frequency Score:     {result.frequency_score:.4f} (Range [0.0, 1.0])")
        print(f"  • Raw Energy Score:    {result.raw_score:.4f}")
        print(f"  • Classification:      {'SUSPICIOUS / MANIPULATED' if result.is_suspicious else 'AUTHENTIC / NATURAL'}")
        print(f"  • Cropped Dimensions:  {result.cropped_shape}")
        print(f"  • Calibration Used:    Divisor={result.calibration_metadata['divisor']}, Threshold={result.calibration_metadata['threshold']}")
        print(f"\n  [EXPLAINABLE REASONING]:\n  {result.reasoning}")
        print(f"\n  [PROACTIVE FALSE-POSITIVE RISKS]:")
        for risk in result.false_positive_risks:
            print(f"    - {risk}")
        print(f"\n  [BLIND SPOT WARNING]:\n  {result.blind_spot_warning}")

        # Generate and save heatmap
        heatmap_out = heatmap_dir / f"heatmap_{path.stem}.png"
        result.get_heatmap(alpha=0.5, output_path=heatmap_out)
        print(f"\n  -> Visual heatmap saved to: {heatmap_out}")
        print("-" * 65)


if __name__ == "__main__":
    main()
