r"""
Quick Single-Image Test Tool for Explainable Deepfake Detection.
Evaluates the full decoupled multi-agent pipeline:
  Module 0: Common Image Preprocessing
  Module 1: Frequency Agent (Receives Grayscale image)
  Module 2: Classifier Agent (Receives RGB image)
  Module 3: Metadata Agent (Receives Original file / raw bytes)
  Module 4: Disagreement Analysis Layer (Zero-fusion categorical arbitration)

Usage:
    python scripts/test_single_image.py "C:\path\to\your\image.jpg"
    python scripts/test_single_image.py "walpa2.jpeg"
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.preprocessing import (
    ImagePreprocessor,
    PreprocessingConfig,
    PreprocessingResult,
    preprocess_image,
)
from modules.frequency.agent import FrequencyAgent
from modules.classifier.agent import ClassifierAgent
from modules.metadata.agent import MetadataAgent
from modules.disagreement.layer import DisagreementLayer


def resolve_or_run_preprocessing(
    img_path: Path,
    output_dir: Path = Path("outputs/preprocessing"),
    force_reprocess: bool = False,
) -> PreprocessingResult:
    """
    Resolves preprocessed representations for the target image:
    1. If the target path points to or matches existing preprocessed outputs in output_dir,
       detects and loads them directly.
    2. Otherwise, automatically runs the Image Preprocessing Module and saves the 5 decoupled artifacts.
    """
    resolved_path = img_path

    # Check if the file is in current directory or if it exists in output_dir/original
    if not resolved_path.exists():
        direct_orig = output_dir / "original" / img_path.name
        if direct_orig.exists():
            resolved_path = direct_orig
        else:
            matches = list((output_dir / "original").glob(f"{img_path.stem}*.*"))
            if matches:
                resolved_path = matches[0]

    if not resolved_path.exists():
        return PreprocessingResult(
            status="failed",
            error_message=f"File not found: {img_path}",
            metadata={"filename": img_path.name, "format": "UNKNOWN / MISSING"},
        )

    # Derive base identifier stem (strip artifact suffixes if someone passed _gray, _rgb, etc.)
    stem = resolved_path.stem
    for suffix in ("_gray", "_rgb", "_normalized", "_metadata"):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]

    # Search for existing decoupled preprocessed outputs
    gray_matches = list((output_dir / "grayscale").glob(f"{stem}*_gray.png"))
    rgb_matches = list((output_dir / "rgb").glob(f"{stem}*_rgb.png"))
    orig_matches = list((output_dir / "original").glob(f"{stem}*.*"))
    meta_matches = list((output_dir / "metadata").glob(f"{stem}*_metadata.json"))
    norm_matches = list((output_dir / "normalized").glob(f"{stem}*_normalized.npy"))

    if not force_reprocess and gray_matches and rgb_matches and orig_matches:
        orig_p = orig_matches[0]
        rgb_p = rgb_matches[0]
        gray_p = gray_matches[0]
        norm_p = norm_matches[0] if norm_matches else None
        meta_p = meta_matches[0] if meta_matches else None

        meta_dict = {}
        if meta_p and meta_p.exists():
            try:
                with open(meta_p, "r", encoding="utf-8") as f:
                    meta_dict = json.load(f)
            except Exception:
                pass

        saved = {
            "original": str(orig_p.resolve()),
            "rgb": str(rgb_p.resolve()),
            "grayscale": str(gray_p.resolve()),
        }
        if norm_p and norm_p.exists():
            saved["normalized"] = str(norm_p.resolve())
        if meta_p and meta_p.exists():
            saved["metadata"] = str(meta_p.resolve())

        try:
            with open(orig_p, "rb") as f:
                raw_b = f.read()
        except Exception:
            raw_b = None

        return PreprocessingResult(
            status="success",
            original_path=str(orig_p.resolve()),
            raw_bytes=raw_b,
            saved_paths=saved,
            metadata=meta_dict,
        )

    # Run preprocessing module automatically
    return preprocess_image(
        image_input=resolved_path,
        output_dir=output_dir,
        save_artifacts=True,
    )


def analyze_single_image(
    img_path: Path,
    heatmap_out: str = "heatmap_result.png",
    output_dir: str = "outputs/preprocessing",
    force_preprocess: bool = False,
):
    print("\n" + "=" * 78)
    print(f"       EXPLAINABLE DEEPFAKE DETECTION: MULTI-AGENT ANALYSIS")
    print(f"       Target: {img_path.name}")
    print("=" * 78)

    # =========================================================================
    # MODULE 0: COMMON IMAGE PREPROCESSING
    # =========================================================================
    prep_dir = Path(output_dir)
    prep_res = resolve_or_run_preprocessing(
        img_path,
        output_dir=prep_dir,
        force_reprocess=force_preprocess,
    )

    if not prep_res.is_success:
        print("\n" + prep_res.format_report())
        print("\n[ERROR] Preprocessing failed. Multi-agent analysis aborted.")
        return

    m = prep_res.metadata
    orig_path_str = prep_res.saved_paths.get("original", prep_res.original_path or str(img_path))
    rgb_path_str = prep_res.saved_paths.get("rgb")
    gray_path_str = prep_res.saved_paths.get("grayscale")
    norm_path_str = prep_res.saved_paths.get("normalized", "In-memory tensor")
    meta_path_str = prep_res.saved_paths.get("metadata", "N/A")

    orig_w = m.get("original_width", "N/A")
    orig_h = m.get("original_height", "N/A")
    proc_w = m.get("processed_width", "N/A")
    proc_h = m.get("processed_height", "N/A")
    resize_m = m.get("resize_method", "aspect_ratio_preserve")
    norm_s = m.get("normalization_status", "SUCCESS")
    fmt = m.get("format", img_path.suffix.lstrip(".").upper())
    size_str = m.get("file_size_formatted", f"{len(prep_res.raw_bytes):,} bytes" if prep_res.raw_bytes else "N/A")

    print("\n" + "=" * 78)
    print("MODULE 0: PREPROCESSING (Common Image Ingestion & Forensic Normalization)")
    print("=" * 78)
    print(f"  • Input Source:         {m.get('filename', img_path.name)} ({fmt}, {size_str})")
    print(f"  • Original Resolution:  {orig_w} x {orig_h} px")
    print(f"  • Processed Resolution: {proc_w} x {proc_h} px ({resize_m})")
    print(f"  • Normalization:        {norm_s}")
    print(f"  • Metadata Integrity:   INTACT (Raw file preserved for forensic inspection)")
    print(f"  • Preprocessing Status: SUCCESSFUL")
    print("\n  Decoupled Forensic Artifacts Generated:")
    print(f"    • Original (Raw):     {orig_path_str}")
    print(f"    • RGB Image:          {rgb_path_str}")
    print(f"    • Grayscale Image:    {gray_path_str}")
    print(f"    • Normalized Tensor:  {norm_path_str}")
    print(f"    • Metadata Audit:     {meta_path_str}")

    # =========================================================================
    # MODULE 1: FREQUENCY AGENT (Receives Grayscale representation)
    # =========================================================================
    freq_input = gray_path_str or prep_res.grayscale_image
    freq_agent = FrequencyAgent()
    try:
        freq_res = freq_agent.analyze(freq_input)
    except Exception as e:
        print(f"\nFrequency Agent Error: {e}")
        return

    verdict_str = "SUSPICIOUS / MANIPULATED" if freq_res.is_suspicious else "AUTHENTIC / NATURAL"
    sep_thresh = freq_res.calibration_metadata.get("threshold", 0.50)

    print("\n" + "=" * 78)
    print("MODULE 1: FREQUENCY AGENT (2D-DCT Spectral Analysis)")
    print("=" * 78)
    print(f"  • Input Representation: Grayscale Image (for 8x8 block DCT & energy analysis)")
    print(f"  • Input Path:           {gray_path_str}")
    print(f"  • Classical Verdict:    {verdict_str}")
    print(f"  • Frequency Score:      {freq_res.frequency_score:.4f} (Range [0.0, 1.0])")
    print(f"  • Raw Energy Score:     {freq_res.raw_score:.4f}")
    print(f"  • Separation Threshold: {sep_thresh:.2f}")
    print(f"  • Cropped Resolution:   {freq_res.cropped_shape[0]} x {freq_res.cropped_shape[1]} px")
    print(f"\n  Explainable Frequency Reasoning:\n  {freq_res.reasoning}")

    # Save visual explainability outputs
    out_heatmap = Path(heatmap_out)
    freq_res.get_heatmap(output_path=out_heatmap)
    rad_path = out_heatmap.parent / "radial_decay.png"
    freq_res.get_radial_plot(output_path=rad_path)
    spec_path = out_heatmap.parent / "dct_spectrum.png"
    freq_res.get_spectrum_plot(output_path=spec_path)

    print(f"\n  Visual Explainability Artifacts:")
    print(f"    -> Frequency Heatmap saved to: {out_heatmap.resolve()}")
    print(f"    -> Radial Spectral Decay plot: {rad_path.resolve()}")
    print(f"    -> Annotated 2D-DCT Spectrum:  {spec_path.resolve()}")

    # =========================================================================
    # MODULE 2: CLASSIFIER AGENT (Receives RGB representation)
    # =========================================================================
    cls_input = rgb_path_str or prep_res.rgb_image
    cls_agent = ClassifierAgent()
    try:
        cls_res = cls_agent.analyze(cls_input, metadata=prep_res.metadata)
    except Exception as e:
        print(f"\nClassifier Agent Error: {e}")
        sys.exit(1)

    print("\n" + "=" * 78)
    print("MODULE 2: CLASSIFIER AGENT (EfficientNet-B0 + Multi-Crop Attention)")
    print("=" * 78)
    print(f"  • Input Representation: RGB Image (for deep vision feature extraction)")
    print(f"  • Input Path:           {rgb_path_str}")
    print(f"  • Verdict:              {cls_res.predicted_label}")
    print(f"  • Classifier Score:     {cls_res.classifier_score:.4f} (Probability image is FAKE)")
    print(f"  • All Probabilities:    {cls_res.all_probabilities}")
    print(f"  • Model Architecture:   {cls_res.model_architecture}")
    if cls_res.peak_bbox:
        print(f"  • Peak Salient BBox:    ymin={cls_res.peak_bbox['ymin']}, ymax={cls_res.peak_bbox['ymax']}, xmin={cls_res.peak_bbox['xmin']}, xmax={cls_res.peak_bbox['xmax']}")
    print(f"  • Physical Noise Var:   {cls_res.semantic_indicators['noise_residual_variance']:.2f}")
    print(f"  • Laplacian Dispersion: {cls_res.semantic_indicators['laplacian_variance']:.2f}")

    gradcam_path = out_heatmap.parent / "gradcam_overlay.png"
    cls_res.get_gradcam_overlay(output_path=gradcam_path)
    print(f"\n  Visual Explainability Artifacts:")
    print(f"    -> Grad-CAM Saliency Overlay:  {gradcam_path.resolve()}")
    print(f"\n  Semantic Reasoning:\n  {cls_res.reasoning}")

    # =========================================================================
    # MODULE 3: METADATA AGENT (Receives Original file / raw bytes)
    # =========================================================================
    meta_input = orig_path_str
    meta_agent = MetadataAgent()
    try:
        meta_res = meta_agent.analyze(meta_input)
    except Exception as e:
        print(f"\nMetadata Agent Error: {e}")
        sys.exit(1)

    print("\n" + "=" * 78)
    print("MODULE 3: METADATA AGENT (4-Tier Provenance & DQT Consistency Engine)")
    print("=" * 78)
    print(f"  • Input Representation: Original Raw Image (preserves EXIF/XMP/C2PA/DQT bytes)")
    print(f"  • Input Path:           {orig_path_str}")
    print(f"  • Integrity Notice:     Preprocessed PNG bypassed (prevents metadata destruction)")
    print(f"  • Metadata Flag:        {meta_res.metadata_flag.upper()}")
    print(f"  • Disclosed Tool:       {meta_res.disclosed_tool if meta_res.disclosed_tool else 'None'}")
    print(f"  • Data Source Tag:      {meta_res.data_source_tag}")
    if meta_res.optical_consistency:
        is_opt = meta_res.optical_consistency.get("is_consistent", True)
        ev_val = meta_res.optical_consistency.get("calculated_ev")
        ev_str = f" (EV={ev_val})" if ev_val is not None else ""
        print(f"  • Optical Consistency:  {'CONSISTENT' if is_opt else 'CONTRADICTIONS DETECTED'}{ev_str}")
    if meta_res.dqt_profile and meta_res.dqt_profile.get("dqt_found"):
        print(f"  • DQT Profile:          {meta_res.dqt_profile.get('matched_profile')}")
    if meta_res.c2pa_details.manifest_found:
        print(f"  • C2PA Manifest:        PRESENT (Issuer: {meta_res.c2pa_details.signing_issuer})")
        if meta_res.c2pa_details.timestamp:
            print(f"  • C2PA Timestamp:       {meta_res.c2pa_details.timestamp}")
    print(f"\n  Provenance Reasoning:\n  {meta_res.reasoning}")

    # =========================================================================
    # MODULE 4: DISAGREEMENT ANALYSIS LAYER
    # =========================================================================
    disagreement_layer = DisagreementLayer()
    dis_res = disagreement_layer.arbitrate(
        frequency_result=freq_res,
        classifier_result=cls_res,
        metadata_result=meta_res,
    )

    print("\n" + disagreement_layer.format_ui_report(dis_res))

    # =========================================================================
    # FINAL PREDICTION SUMMARY
    # =========================================================================
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
        final_call = "UNCERTAIN (Classifier Flagged Only - Manual Review Recommended)"
        confidence_str = f"LOW ({cls_res.classifier_score * 100:.1f}% Classifier vs {freq_res.frequency_score * 100:.1f}% Frequency)"
        max_patch = cls_res.semantic_indicators.get("max_manipulated_patch", "localized")
        ratio = cls_res.semantic_indicators.get("section_noise_disparity_ratio", 1.0)
        detail = (
            f"Classifier Agent flagged anomaly in '{max_patch}' (disparity ratio {ratio:.2f}x),\n"
            f"  but Frequency Agent confirmed natural spectral decay. Inconclusive without C2PA provenance."
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
    parser = argparse.ArgumentParser(
        description="Explainable Deepfake Detection - Decoupled Multi-Agent Evaluation Tool"
    )
    parser.add_argument(
        "image_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to the image file to analyze (optional, prompts if omitted)",
    )
    parser.add_argument(
        "--heatmap-out",
        type=str,
        default="heatmap_result.png",
        help="Path to save the generated heatmap overlay (default: heatmap_result.png)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/preprocessing",
        help="Directory where preprocessed decoupled artifacts are saved/detected (default: outputs/preprocessing)",
    )
    parser.add_argument(
        "--force-preprocess",
        action="store_true",
        help="Force re-running preprocessing even if matching cached artifacts exist",
    )
    args = parser.parse_args()

    if args.image_path:
        img_path = Path(args.image_path.strip('"').strip("'"))
        analyze_single_image(
            img_path=img_path,
            heatmap_out=args.heatmap_out,
            output_dir=args.output_dir,
            force_preprocess=args.force_preprocess,
        )
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
            if not user_in or user_in.lower() in ("q", "quit", "exit"):
                print("Exiting.")
                break
            p = Path(user_in)
            if not p.exists() and not (Path(args.output_dir) / "original" / p.name).exists():
                print(f"Error: File not found: {p}\n")
                continue
            analyze_single_image(
                img_path=p,
                heatmap_out=args.heatmap_out,
                output_dir=args.output_dir,
                force_preprocess=args.force_preprocess,
            )
            print()


if __name__ == "__main__":
    main()

