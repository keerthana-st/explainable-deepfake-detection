"""
Demonstration Script for Classifier Agent (Module 2).

Runs the Classifier Agent on diverse authentic and synthetic test cases,
demonstrating semantic classification, modern diffusion artifact detection,
and explainable reasoning.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from modules.classifier.agent import ClassifierAgent


def get_demo_cases():
    cases = []
    # User's actual test images if present
    user_chatgpt = Path(r"D:\ChatGPT Image Sep 3, 2026, 03_02_44 PM.png")
    user_profile = Path(r"D:\profile pic.png")

    if user_chatgpt.exists():
        cases.append(("User ChatGPT (DALL-E 3) Image", user_chatgpt))
    if user_profile.exists():
        cases.append(("User Profile Picture (Real Camera Portrait)", user_profile))

    # Also demo benchmark samples
    demo_dir = PROJECT_ROOT / "data" / "demo_samples"
    demo_dir.mkdir(parents=True, exist_ok=True)

    # 1. Real photo simulation
    p_real = demo_dir / "demo_authentic.png"
    if p_real.exists():
        cases.append(("Benchmark Authentic Natural Photo", p_real))

    # 2. GAN upsampling simulation
    p_gan = demo_dir / "demo_gan_upsampled.png"
    if p_gan.exists():
        cases.append(("Benchmark GAN Upsampling Fake", p_gan))

    # 3. Compressed photo simulation
    p_comp = demo_dir / "demo_textured_photo.jpg"
    if p_comp.exists():
        cases.append(("Benchmark Compressed Natural Photo", p_comp))

    return cases


def main():
    print("==============================================================================")
    print("        CLASSIFIER AGENT (MODULE 2) - SEMANTIC INFERENCE DEMO")
    print("==============================================================================")

    agent = ClassifierAgent()
    cases = get_demo_cases()

    for label, path in cases:
        print(f"\n>>> Analyzing Image: [{label}]")
        print(f"    File: {path.name}")
        res = agent.analyze(path)

        print(f"  • Predicted Label:       {res.predicted_label}")
        print(f"  • Classifier Score:      {res.classifier_score:.4f} (Probability image is FAKE)")
        print(f"  • All Probabilities:     {res.all_probabilities}")
        print(f"  • Model Architecture:    {res.model_architecture}")
        print(f"  • Physical Noise Var:    {res.semantic_indicators['noise_residual_variance']:.2f}")
        print(f"  • Laplacian Dispersion:  {res.semantic_indicators['laplacian_variance']:.2f}")
        print("\n  [SEMANTIC EXPLAINABLE REASONING]:")
        print(res.reasoning)
        print("\n  [DISAGREEMENT LAYER GUIDANCE]:")
        print(res.disagreement_context)
        print("-" * 78)


if __name__ == "__main__":
    main()
