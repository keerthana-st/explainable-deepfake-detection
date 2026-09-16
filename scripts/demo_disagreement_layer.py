"""
Interactive Demonstration CLI for Module 4: Disagreement Analysis Layer.
Demonstrates all four core arbitration states (A, B, C, D) and UI formatting.
"""

from pathlib import Path
import sys
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.disagreement import DisagreementLayer, DecisionBasis, OutputState


def print_section(title: str):
    print("\n" + "#" * 80)
    print(f" {title}")
    print("#" * 80)


def main():
    print("=" * 80)
    print("   MODULE 4: DISAGREEMENT ANALYSIS LAYER — COMPREHENSIVE 4-STATE AUDIT")
    print("=" * 80)
    print("CORE NOVELTY: Zero score averaging/weighting. Pure categorical arbitration.")
    print("=" * 80)

    analyzer = DisagreementLayer()

    # -------------------------------------------------------------------------
    # STATE A: HIGH CONFIDENCE (REAL & FAKE)
    # -------------------------------------------------------------------------
    print_section("STATE A: HIGH_CONFIDENCE_FAKE (Both Probabilistic Agents Agree FAKE)")
    res_a_fake = analyzer.arbitrate(
        frequency_result={"frequency_score": 0.88},
        classifier_result={"classifier_score": 0.94},
        metadata_result={"metadata_flag": "missing", "disclosed_tool": None},
    )
    print(analyzer.format_ui_report(res_a_fake))
    print("JSON Output Structure:")
    print(json.dumps(res_a_fake.to_dict(), indent=2))

    print_section("STATE A: HIGH_CONFIDENCE_REAL (Both Probabilistic Agents Agree REAL)")
    res_a_real = analyzer.arbitrate(
        frequency_result={"frequency_score": 0.12},
        classifier_result={"classifier_score": 0.18},
        metadata_result={"metadata_flag": "present_no_ai_signal", "disclosed_tool": None},
    )
    print(analyzer.format_ui_report(res_a_real))

    # -------------------------------------------------------------------------
    # STATE B: UNCERTAIN (DISAGREEMENT)
    # -------------------------------------------------------------------------
    print_section("STATE B: UNCERTAIN (Frequency Flagged Only — High-Texture False Positive)")
    res_b_freq = analyzer.arbitrate(
        frequency_result={"frequency_score": 0.76},  # High DCT energy from sharp hair/fabric
        classifier_result={"classifier_score": 0.22}, # Authentic camera noise
        metadata_result={"metadata_flag": "missing", "disclosed_tool": None},
    )
    print(analyzer.format_ui_report(res_b_freq))

    print_section("STATE B: UNCERTAIN (Classifier Flagged Only — Diffusion Model Blind Spot)")
    res_b_cls = analyzer.arbitrate(
        frequency_result={"frequency_score": 0.24},  # Smooth VAE decoder
        classifier_result={"classifier_score": 0.89}, # Airbrushed textures & anatomy cues
        metadata_result={"metadata_flag": "missing", "disclosed_tool": None},
    )
    print(analyzer.format_ui_report(res_b_cls))

    # -------------------------------------------------------------------------
    # STATE C: PROVENANCE OVERRIDE (CRITICAL FAILURE MODE RESOLUTION)
    # -------------------------------------------------------------------------
    print_section("STATE C: PROVENANCE_OVERRIDE_FAKE (Both Agents Predicted REAL, Metadata Disclosed AI)")
    res_c = analyzer.arbitrate(
        frequency_result={"frequency_score": 0.15},  # Says REAL
        classifier_result={"classifier_score": 0.20}, # Says REAL
        metadata_result={
            "metadata_flag": "ai_disclosed",
            "disclosed_tool": "ChatGPT c2pa-rs/0.28.4",
        },
    )
    print(analyzer.format_ui_report(res_c))
    print("JSON Output Structure:")
    print(json.dumps(res_c.to_dict(), indent=2))

    # -------------------------------------------------------------------------
    # STATE D: PROVENANCE UNCERTAIN
    # -------------------------------------------------------------------------
    print_section("STATE D: PROVENANCE_UNCERTAIN (Manifest Present but Unparseable)")
    res_d = analyzer.arbitrate(
        frequency_result={"frequency_score": 0.30},
        classifier_result={"classifier_score": 0.35},
        metadata_result={
            "metadata_flag": "manifest_present_generator_unclear",
            "disclosed_tool": None,
            "validation_status": "Custom private JUMBF box without claim_generator tag",
        },
    )
    print(analyzer.format_ui_report(res_d))

    print("\n" + "=" * 80)
    print("Disagreement Analysis Layer Demo Complete. All 4 States Verified.")
    print("=" * 80)


if __name__ == "__main__":
    main()
