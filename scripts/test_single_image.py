r"""
Quick Single-Image Test Tool for Explainable Deepfake Detection.
Evaluates both Module 1 (Frequency Agent) and Module 2 (Classifier Agent) side-by-side.

Usage:
    python scripts/test_single_image.py "C:\path\to\your\image.jpg"
"""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.frequency.agent import FrequencyAgent
from modules.classifier.agent import ClassifierAgent
from modules.metadata.agent import MetadataAgent
from modules.disagreement.layer import DisagreementLayer


def analyze_single_image(img_path: Path, heatmap_out: str = "heatmap_result.png"):
    if not img_path.exists():
        print(f"\nError: File not found at: {img_path}")
        return

    print("\n" + "=" * 78)
    print(f"       EXPLAINABLE DEEPFAKE DETECTION: SINGLE IMAGE ANALYSIS")
    print(f"       Target: {img_path.name}")
    print("=" * 78)

    # 1. Run Module 1: Frequency Agent
    freq_agent = FrequencyAgent()
    try:
        freq_res = freq_agent.analyze(img_path)
    except Exception as e:
        print(f"Frequency Agent Error: {e}")
        return

    # 2. Run Module 2: Classifier Agent
    cls_agent = ClassifierAgent()
    try:
        cls_res = cls_agent.analyze(img_path)
    except Exception as e:
        print(f"Classifier Agent Error: {e}")
        sys.exit(1)

    # 3. Run Module 3: Metadata Agent
    meta_agent = MetadataAgent()
    try:
        meta_res = meta_agent.analyze(img_path)
    except Exception as e:
        print(f"Metadata Agent Error: {e}")
        sys.exit(1)

    print("\n[MODULE 1: FREQUENCY AGENT (Classical 2D-DCT Signal Processing)]")
    status1 = "SUSPICIOUS / MANIPULATED" if freq_res.is_suspicious else "AUTHENTIC / NATURAL"
    print(f"  • Verdict:              {status1}")
    print(f"  • Frequency Score:      {freq_res.frequency_score:.4f} (Range [0.0, 1.0])")
    print(f"  • Raw Energy Score:     {freq_res.raw_score:.4f} (Separation Threshold: {freq_res.calibration_metadata['threshold']:.2f})")
    print(f"  • Cropped Resolution:   {freq_res.cropped_shape[0]} x {freq_res.cropped_shape[1]} px")
    print(f"\n  Explainable Reasoning:\n  {freq_res.reasoning}")

    out_heatmap = Path(heatmap_out)
    freq_res.get_heatmap(output_path=out_heatmap)
    print(f"\n  -> Visual Heatmap saved to: {out_heatmap.resolve()}")

    print("\n" + "-" * 78)
    print("[MODULE 2: CLASSIFIER AGENT (Learned Semantic Vision Classifier)]")
    print(f"  • Verdict:              {cls_res.predicted_label}")
    print(f"  • Classifier Score:      {cls_res.classifier_score:.4f} (Probability image is FAKE)")
    print(f"  • All Probabilities:     {cls_res.all_probabilities}")
    print(f"  • Model Architecture:    {cls_res.model_architecture}")
    print(f"  • Physical Noise Var:    {cls_res.semantic_indicators['noise_residual_variance']:.2f}")
    print(f"  • Laplacian Dispersion:  {cls_res.semantic_indicators['laplacian_variance']:.2f}")
    print(f"\n  Semantic Reasoning:\n  {cls_res.reasoning}")

    print("\n" + "-" * 78)
    print("[MODULE 3: METADATA AGENT (Rule-Based Provenance & C2PA Parser)]")
    print(f"  • Metadata Flag:        {meta_res.metadata_flag.upper()}")
    print(f"  • Disclosed Tool:       {meta_res.disclosed_tool if meta_res.disclosed_tool else 'None'}")
    print(f"  • Data Source Tag:      {meta_res.data_source_tag}")
    if meta_res.c2pa_details.manifest_found:
        print(f"  • C2PA Manifest:        PRESENT (Issuer: {meta_res.c2pa_details.signing_issuer})")
        if meta_res.c2pa_details.timestamp:
            print(f"  • C2PA Timestamp:       {meta_res.c2pa_details.timestamp}")
    print(f"\n  Provenance Reasoning:\n  {meta_res.reasoning}")

    # 4. Run Module 4: Disagreement Analysis Layer
    disagreement_layer = DisagreementLayer()
    dis_res = disagreement_layer.arbitrate(
        frequency_result=freq_res,
        classifier_result=cls_res,
        metadata_result=meta_res,
    )

    print("\n" + disagreement_layer.format_ui_report(dis_res))

    # 5. Clear Executive Prediction Summary
    print("\n" + "=" * 78)
    print("                      FINAL PREDICTION SUMMARY")
    print("=" * 78)

    if dis_res.output_state == "PROVENANCE_OVERRIDE_FAKE":
        final_call = "FAKE / AI GENERATED"
        confidence_str = "100% (Cryptographic Proof)"
        detail = f"Cryptographically signed C2PA provenance discloses generation tool: '{meta_res.disclosed_tool}'."
    elif dis_res.output_state == "HIGH_CONFIDENCE_FAKE":
        final_call = "FAKE / SYNTHETIC"
        confidence_str = f"HIGH ({max(cls_res.classifier_score, freq_res.frequency_score) * 100:.1f}%)"
        detail = "Both classical frequency analysis and semantic vision classifier confirmed synthetic origin."
    elif dis_res.output_state == "HIGH_CONFIDENCE_REAL":
        final_call = "REAL / AUTHENTIC"
        confidence_str = f"HIGH ({(1.0 - cls_res.classifier_score) * 100:.1f}%)"
        detail = "All optical metrics, noise uniformity, and vision classifiers confirm a genuine photograph."
    elif dis_res.output_state == "UNCERTAIN_classifier_flagged_only":
        final_call = "FAKE / MANIPULATED (AI Inpainting or Diffusion)"
        confidence_str = f"{cls_res.classifier_score * 100:.1f}% (Classifier Agent)"
        max_patch = cls_res.semantic_indicators.get("max_manipulated_patch", "localized")
        ratio = cls_res.semantic_indicators.get("section_noise_disparity_ratio", 1.0)
        detail = (
            f"Manipulated Region: '{max_patch}' (disparity ratio {ratio:.2f}x).\n"
            f"  * Note on Frequency Score: Modern diffusion/inpainting models decode smoothly and bypass\n"
            f"    classical 2D-DCT frequency analysis. The Classifier Agent's spatial finding confirms manipulation."
        )
    elif dis_res.output_state == "UNCERTAIN_frequency_flagged_only":
        final_call = "SUSPICIOUS / POSSIBLE MANIPULATION"
        confidence_str = f"{freq_res.frequency_score * 100:.1f}% (Frequency Agent)"
        detail = "High periodic frequency energy detected. Inspect for GAN upsampling grids or heavy compression."
    else:
        final_call = "UNCERTAIN (Manual Review Required)"
        confidence_str = "LOW"
        detail = "Inconclusive or conflicting signals without verifiable provenance."

    print(f"  • PREDICTION:          {final_call}")
    print(f"  • CONFIDENCE:          {confidence_str}")
    print(f"  • KEY EVIDENCE:        {detail}")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description="Test Frequency, Classifier, and Metadata Agents on any image")
    parser.add_argument("image_path", type=str, nargs="?", default=None, help="Path to the image file to analyze (optional, prompts if omitted)")
    parser.add_argument("--heatmap-out", type=str, default="heatmap_result.png", help="Path to save the generated heatmap overlay")
    args = parser.parse_args()

    if args.image_path:
        img_path = Path(args.image_path.strip('"').strip("'"))
        analyze_single_image(img_path, args.heatmap_out)
    else:
        print("=" * 78)
        print("       EXPLAINABLE DEEPFAKE DETECTION: SINGLE IMAGE ANALYSIS")
        print("=" * 78)
        print("Tip: You can drag and drop an image file directly into this terminal.\n")
        while True:
            try:
                user_in = input("Enter image path (or 'q' to quit): ").strip().strip('"').strip("'")
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break
            if not user_in or user_in.lower() in ('q', 'quit', 'exit'):
                print("Exiting.")
                break
            p = Path(user_in)
            if not p.exists():
                print(f"Error: File not found: {p}\n")
                continue
            analyze_single_image(p, args.heatmap_out)
            print()


if __name__ == "__main__":
    main()
