r"""
Universal Prediction Script for Explainable Deepfake Detection.

Allows predicting on a single image, multiple images, or entire folders.
Supports interactive mode if no arguments are provided.

Usage:
    python scripts/predict.py "path/to/image.jpg"
    python scripts/predict.py "path/to/folder"
    python scripts/predict.py "img1.jpg" "img2.png" "img3.jpeg"
    python scripts/predict.py  (interactive prompt)
"""

import argparse
from pathlib import Path
import sys
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.preprocessing import preprocess_image
from modules.frequency.agent import FrequencyAgent
from modules.classifier.agent import ClassifierAgent
from modules.metadata.agent import MetadataAgent
from modules.disagreement.layer import DisagreementLayer

VALID_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}


def analyze_image(img_path: Path, freq_agent, cls_agent, meta_agent, dis_layer):
    try:
        # Preprocessing: generate decoupled representations
        prep_res = preprocess_image(img_path, save_artifacts=False)
        if not prep_res.is_success:
            return {
                'file': img_path.name,
                'prediction': 'ERROR',
                'confidence': 'N/A',
                'evidence': f"Preprocessing: {prep_res.error_message}"[:45],
                'freq_score': 'N/A',
                'cls_score': 'N/A',
                'meta': 'N/A',
            }

        # Module 1 receives Grayscale representation
        freq_res = freq_agent.analyze(prep_res.grayscale_image, keep_original=False)
        # Module 2 receives RGB representation
        cls_res = cls_agent.analyze(prep_res.rgb_image, metadata=prep_res.metadata)
        # Module 3 receives original raw image/bytes
        meta_res = meta_agent.analyze(img_path)
        # Module 4 performs zero-fusion arbitration
        dis_res = dis_layer.arbitrate(freq_res, cls_res, meta_res)

        # Map to user-friendly prediction
        if dis_res.output_state == 'PROVENANCE_OVERRIDE_FAKE':
            pred = 'FAKE'
            conf = '100%'
            evidence = f"C2PA Provenance: {meta_res.disclosed_tool or 'AI Disclosed'}"
        elif dis_res.output_state == 'HIGH_CONFIDENCE_FAKE':
            score = max(cls_res.classifier_score, freq_res.frequency_score)
            pred = 'FAKE'
            conf = f"{score * 100:.1f}%"
            evidence = "Both Frequency and Vision Classifier Agree Fake"
        elif dis_res.output_state == 'HIGH_CONFIDENCE_REAL':
            score = 1.0 - cls_res.classifier_score
            pred = 'REAL'
            conf = f"{score * 100:.1f}%"
            evidence = "Natural Optical Grain & Uniform Noise Floor"
        elif dis_res.output_state == 'UNCERTAIN_classifier_flagged_only':
            pred = 'UNCERTAIN'
            conf = 'LOW'
            max_patch = cls_res.semantic_indicators.get('max_manipulated_patch', 'localized')
            ratio = cls_res.semantic_indicators.get('section_noise_disparity_ratio', 1.0)
            evidence = f"Classifier Flagged '{max_patch}' ({ratio:.1f}x noise disparity); Frequency Real"
        elif dis_res.output_state == 'UNCERTAIN_frequency_flagged_only':
            pred = 'SUSPICIOUS'
            conf = f"{freq_res.frequency_score * 100:.1f}%"
            evidence = "Periodic GAN Upsampling Lattice Detected"
        else:
            pred = 'UNCERTAIN'
            conf = '50%'
            evidence = "Inconclusive multi-agent signals"

        return {
            'file': img_path.name,
            'prediction': pred,
            'confidence': conf,
            'evidence': evidence,
            'freq_score': f"{freq_res.frequency_score:.3f}",
            'cls_score': f"{cls_res.classifier_score:.3f}",
            'meta': meta_res.metadata_flag.upper(),
        }
    except Exception as e:
        return {
            'file': img_path.name,
            'prediction': 'ERROR',
            'confidence': 'N/A',
            'evidence': str(e)[:45],
            'freq_score': 'N/A',
            'cls_score': 'N/A',
            'meta': 'N/A',
        }


def main():
    parser = argparse.ArgumentParser(description="Predict REAL vs FAKE on any image or folder")
    parser.add_argument('inputs', nargs='*', help="Image file(s) or folder(s) to analyze")
    parser.add_argument('--detailed', action='store_true', help="Show per-module score breakdown")
    args = parser.parse_args()

    paths_to_check = []
    if args.inputs:
        for item in args.inputs:
            p = Path(item.strip('"').strip("'"))
            if p.is_dir():
                for f in sorted(p.iterdir()):
                    if f.suffix.lower() in VALID_EXTS:
                        paths_to_check.append(f)
            elif p.is_file():
                paths_to_check.append(p)
            else:
                print(f"Warning: Not found: {p}")
    else:
        # Interactive mode
        print("=" * 85)
        print("         EXPLAINABLE DEEPFAKE DETECTION -- INTERACTIVE PREDICTION")
        print("=" * 85)
        try:
            user_in = input("Enter image file path or directory (or drag & drop): ").strip('"').strip("'")
        except EOFError:
            return
        if not user_in:
            return
        p = Path(user_in)
        if p.is_dir():
            for f in sorted(p.iterdir()):
                if f.suffix.lower() in VALID_EXTS:
                    paths_to_check.append(f)
        elif p.is_file():
            paths_to_check.append(p)
        else:
            print(f"Error: Invalid path: {p}")
            return

    if not paths_to_check:
        print("No valid images found to analyze.")
        return

    print(f"\nAnalyzing {len(paths_to_check)} image(s)... Please wait.")
    freq_agent = FrequencyAgent()
    cls_agent = ClassifierAgent()
    meta_agent = MetadataAgent()
    dis_layer = DisagreementLayer()

    results = []
    for img_p in paths_to_check:
        res = analyze_image(img_p, freq_agent, cls_agent, meta_agent, dis_layer)
        results.append(res)

    print("\n" + "=" * 110)
    header = f"{'IMAGE NAME':<24} | {'PREDICTION':<14} | {'CONFIDENCE':<12} | {'PRIMARY EVIDENCE / DETECTED ARTIFACT':<50}"
    print(header)
    print("=" * 110)
    for r in results:
        row = f"{r['file'][:23]:<24} | {r['prediction']:<14} | {r['confidence']:<12} | {r['evidence'][:50]:<50}"
        print(row)
    print("=" * 110)

    if args.detailed:
        print("\n" + "-" * 75)
        print("DETAILED MODULE SCORES BREAKDOWN:")
        print(f"{'IMAGE':<24} | {'FREQ SCORE':<12} | {'CLASSIFIER':<12} | {'METADATA':<12}")
        print("-" * 75)
        for r in results:
            print(f"{r['file'][:23]:<24} | {r['freq_score']:<12} | {r['cls_score']:<12} | {r['meta']:<12}")
        print("-" * 75)


if __name__ == '__main__':
    main()
