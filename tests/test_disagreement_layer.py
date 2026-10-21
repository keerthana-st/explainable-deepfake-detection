"""
Unit and Regression Tests for Module 4: Disagreement Analysis Layer.

Validates:
1. Strict adherence to the four output states (A, B, C, D)
2. Zero mathematical score averaging / weighting (pure categorical decision logic)
3. Provenance Override mechanics (State C) and explicit override phrasing
4. Fallback behavior when manifest is unclear (State D)
5. Contradiction-proof UI report generation
6. Configurable thresholds per agent
"""

import pytest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.disagreement import (
    DisagreementLayer,
    DisagreementResult,
    DecisionBasis,
    ConflictType,
    OutputState,
)


@pytest.fixture
def analyzer():
    return DisagreementLayer(default_frequency_threshold=0.50, default_classifier_threshold=0.50)


# =============================================================================
# 1. STATE A TESTS: CONVERGENT PROBABILISTIC AGREEMENT
# =============================================================================

def test_state_a_both_agree_fake(analyzer):
    """STATE A: Both probabilistic agents agree fake -> HIGH_CONFIDENCE_FAKE."""
    res = analyzer.arbitrate(
        frequency_result={"frequency_score": 0.85},
        classifier_result={"classifier_score": 0.90},
        metadata_result={"metadata_flag": "missing", "disclosed_tool": None},
    )

    assert res.verdict == "fake"
    assert res.confidence == "high"
    assert res.decision_basis == DecisionBasis.AGREE.value
    assert res.output_state == OutputState.HIGH_CONFIDENCE_FAKE.value
    assert res.conflict_type is None
    assert "HIGH CONFIDENCE SYNTHETIC" in res.reasoning
    assert "0.8500" in res.reasoning
    assert "0.9000" in res.reasoning

    # Verify UI consistency
    report = analyzer.format_ui_report(res)
    assert "VERDICT: FAKE" in report
    assert "CONVERGENT_PROBABILISTIC_AGREEMENT" in report


def test_state_a_both_agree_real(analyzer):
    """STATE A: Both probabilistic agents agree real -> HIGH_CONFIDENCE_REAL."""
    res = analyzer.arbitrate(
        frequency_result={"frequency_score": 0.15},
        classifier_result={"classifier_score": 0.20},
        metadata_result={"metadata_flag": "present_no_ai_signal", "disclosed_tool": None},
    )

    assert res.verdict == "real"
    assert res.confidence == "high"
    assert res.decision_basis == DecisionBasis.AGREE.value
    assert res.output_state == OutputState.HIGH_CONFIDENCE_REAL.value
    assert res.conflict_type is None
    assert "HIGH CONFIDENCE AUTHENTIC" in res.reasoning

    # Verify UI consistency
    report = analyzer.format_ui_report(res)
    assert "VERDICT: REAL" in report
    assert "CONVERGENT_PROBABILISTIC_AGREEMENT" in report


# =============================================================================
# 2. STATE B TESTS: PROBABILISTIC DISAGREEMENT (BOTH DIRECTIONS)
# =============================================================================

def test_state_b_frequency_flagged_only(analyzer):
    """STATE B: Frequency fake, Classifier real -> UNCERTAIN_frequency_flagged_only."""
    res = analyzer.arbitrate(
        frequency_result={"frequency_score": 0.72},
        classifier_result={"classifier_score": 0.25},
        metadata_result={"metadata_flag": "missing", "disclosed_tool": None},
    )

    assert res.verdict == "uncertain"
    assert res.confidence == "low"
    assert res.decision_basis == DecisionBasis.DISAGREE.value
    assert res.conflict_type == ConflictType.FREQUENCY_FLAGGED_ONLY.value
    assert res.output_state == OutputState.UNCERTAIN_FREQUENCY_FLAGGED_ONLY.value
    assert "FREQUENCY FLAGGED ONLY" in res.reasoning
    assert "high-contrast natural textures" in res.reasoning.lower() or "sharpening" in res.reasoning.lower()

    # Verify UI consistency
    report = analyzer.format_ui_report(res)
    assert "VERDICT: UNCERTAIN" in report
    assert "AGENT_DISAGREEMENT" in report
    assert "frequency_flagged_only" in report


def test_state_b_classifier_flagged_only(analyzer):
    """STATE B: Frequency real, Classifier fake -> UNCERTAIN_classifier_flagged_only."""
    res = analyzer.arbitrate(
        frequency_result={"frequency_score": 0.18},
        classifier_result={"classifier_score": 0.88},
        metadata_result={"metadata_flag": "missing", "disclosed_tool": None},
    )

    assert res.verdict == "uncertain"
    assert res.confidence == "low"
    assert res.decision_basis == DecisionBasis.DISAGREE.value
    assert res.conflict_type == ConflictType.CLASSIFIER_FLAGGED_ONLY.value
    assert res.output_state == OutputState.UNCERTAIN_CLASSIFIER_FLAGGED_ONLY.value
    assert "CLASSIFIER FLAGGED ONLY" in res.reasoning
    assert "diffusion" in res.reasoning.lower()

    # Verify UI consistency
    report = analyzer.format_ui_report(res)
    assert "VERDICT: UNCERTAIN" in report
    assert "AGENT_DISAGREEMENT" in report
    assert "classifier_flagged_only" in report


# =============================================================================
# 3. STATE C TESTS: PROVENANCE OVERRIDE (MANDATORY RESOLUTION)
# =============================================================================

def test_state_c_provenance_override_when_both_agents_said_real(analyzer):
    """
    CRITICAL REQUIREMENT 4:
    Both probabilistic agents agree REAL, but cryptographic metadata discloses an AI tool.
    Must return FAKE with PROVENANCE_OVERRIDE basis.
    Must NEVER be labeled 'CONCORDANT', 'AGREEMENT', or 'NO CONFLICT'.
    """
    res = analyzer.arbitrate(
        frequency_result={"frequency_score": 0.10},  # REAL
        classifier_result={"classifier_score": 0.12}, # REAL
        metadata_result={
            "metadata_flag": "ai_disclosed",
            "disclosed_tool": "ChatGPT c2pa-rs/0.28.4",
        },
    )

    # 1. Verdict & Confidence
    assert res.verdict == "fake"
    assert res.confidence == "high"
    assert res.decision_basis == DecisionBasis.PROVENANCE_OVERRIDE.value
    assert res.output_state == OutputState.PROVENANCE_OVERRIDE_FAKE.value

    # 2. Phrasing mandates: Must state probabilistic agents were overridden
    assert "PROVENANCE OVERRIDE DETECTED" in res.reasoning
    assert "evaluated this image as authentic photographic capture" in res.reasoning
    assert "ChatGPT c2pa-rs/0.28.4" in res.reasoning
    assert "overrides probabilistic signals" in res.reasoning

    # 3. Anti-Contradiction Bug Check: Must NEVER contain agreement labels in UI
    report = analyzer.format_ui_report(res)
    assert "VERDICT: FAKE / SYNTHETIC (PROVENANCE OVERRIDE)" in report
    assert "CONFIDENCE: HIGH | BASIS: PROVENANCE_OVERRIDE" in report
    assert "CONCORDANT" not in report.upper()
    assert "NO CONFLICT" not in report.upper()
    assert "BASIS: CONVERGENT_PROBABILISTIC_AGREEMENT" not in report
    assert res.decision_basis != "AGREE"
    assert "AGREEMENT" not in res.output_state


def test_state_c_provenance_override_on_disagreeing_agents(analyzer):
    """STATE C: Probabilistic agents disagree, but metadata proves AI generation."""
    res = analyzer.arbitrate(
        frequency_result={"frequency_score": 0.80},  # FAKE
        classifier_result={"classifier_score": 0.20}, # REAL
        metadata_result={
            "metadata_flag": "ai_disclosed",
            "disclosed_tool": "Midjourney v6.0",
        },
    )

    assert res.verdict == "fake"
    assert res.confidence == "high"
    assert res.decision_basis == DecisionBasis.PROVENANCE_OVERRIDE.value
    assert res.output_state == OutputState.PROVENANCE_OVERRIDE_FAKE.value
    assert "Midjourney v6.0" in res.reasoning
    assert "PROVENANCE OVERRIDE DETECTED" in res.reasoning


def test_state_c_provenance_override_when_both_agents_said_fake(analyzer):
    """STATE C: Both agents said fake and metadata confirms AI."""
    res = analyzer.arbitrate(
        frequency_result={"frequency_score": 0.85},
        classifier_result={"classifier_score": 0.95},
        metadata_result={
            "metadata_flag": "ai_disclosed",
            "disclosed_tool": "OpenAI Media Service API v2.2.0",
        },
    )

    assert res.verdict == "fake"
    assert res.confidence == "high"
    assert res.decision_basis == DecisionBasis.PROVENANCE_OVERRIDE.value
    assert res.output_state == OutputState.PROVENANCE_OVERRIDE_FAKE.value


def test_user_specified_provenance_override_contract(analyzer):
    """
    EXACT USER VERIFICATION TEST CASE:
    Input:
      frequency_score: 0.096 (real)
      classifier_score: 0.150 (real)
      metadata_flag: "ai_disclosed"
      disclosed_tool: "ChatGPT"

    Expected output:
      final_verdict: "fake"
      decision_basis: "PROVENANCE_OVERRIDE"
      conflict_breakdown: must explain that both vision models said real,
        but provenance overrides them
      UI:
        - Header must say FAKE
        - Badge must say PROVENANCE OVERRIDE (never CONCORDANT)
        - Reasoning must state both models said real but provenance confirmed fake
    """
    res = analyzer.arbitrate(
        frequency_result={"frequency_score": 0.096},
        classifier_result={"classifier_score": 0.150},
        metadata_result={"metadata_flag": "ai_disclosed", "disclosed_tool": "ChatGPT"},
    )

    # 1. Output properties
    assert res.verdict == "fake"
    assert res.decision_basis == "PROVENANCE_OVERRIDE"
    assert res.output_state == OutputState.PROVENANCE_OVERRIDE_FAKE.value
    assert res.confidence == "high"
    assert res.badge == "PROVENANCE OVERRIDE"

    # 2. Reasoning Phrasing Mandate
    expected_override_phrase = (
        "Frequency and Classifier agents both indicated REAL, but cryptographic "
        "provenance metadata (disclosed generator: ChatGPT) confirms synthetic origin. "
        "This provenance evidence overrides the probabilistic signals, as it constitutes "
        "direct, verifiable proof rather than statistical inference."
    )
    assert expected_override_phrase in res.reasoning
    assert "Frequency and Classifier agents both indicated REAL" in res.reasoning
    assert "disclosed generator: ChatGPT" in res.reasoning
    assert "overrides the probabilistic signals" in res.reasoning

    # 3. UI report consistency
    report = analyzer.format_ui_report(res)
    assert "FAKE" in report
    assert "PROVENANCE OVERRIDE" in report
    assert "CONCORDANT" not in report.upper()
    assert "NO CONFLICT" not in report.upper()
    assert "BASIS: CONVERGENT_PROBABILISTIC_AGREEMENT" not in report
    assert "CONCORDANT" not in res.badge.upper()
    assert "AGREEMENT" not in res.badge.upper()
    assert "NO CONFLICT" not in res.badge.upper()


# =============================================================================
# 4. STATE D TESTS: PROVENANCE UNCERTAIN (EDGE CASE)
# =============================================================================

def test_state_d_manifest_unclear_falls_back_to_probabilistic(analyzer):
    """
    STATE D: C2PA manifest present but unparseable.
    Must NOT trigger provenance override; falls back to probabilistic verdict.
    """
    # Case D1: Probabilistic agents agree REAL
    res_d1 = analyzer.arbitrate(
        frequency_result={"frequency_score": 0.20},
        classifier_result={"classifier_score": 0.30},
        metadata_result={
            "metadata_flag": "manifest_present_generator_unclear",
            "disclosed_tool": None,
            "validation_status": "Unknown JUMBF box format",
        },
    )

    assert res_d1.verdict == "real"
    assert res_d1.confidence == "high"
    assert res_d1.decision_basis == DecisionBasis.PROVENANCE_UNCERTAIN.value
    assert res_d1.output_state == OutputState.PROVENANCE_UNCERTAIN.value
    assert "DOES NOT qualify for an override" in res_d1.reasoning

    # Case D2: Probabilistic agents disagree
    res_d2 = analyzer.arbitrate(
        frequency_result={"frequency_score": 0.70},
        classifier_result={"classifier_score": 0.20},
        metadata_result={
            "metadata_flag": "manifest_present_generator_unclear",
            "disclosed_tool": None,
        },
    )

    assert res_d2.verdict == "uncertain"
    assert res_d2.confidence == "low"
    assert res_d2.decision_basis == DecisionBasis.PROVENANCE_UNCERTAIN.value


# =============================================================================
# 5. MATHEMATICAL NON-FUSION VALIDATION TEST
# =============================================================================

def test_strict_non_fusion_guarantee(analyzer):
    """
    CORE NOVELTY VERIFICATION:
    Verifies that no mathematical fusion or averaging occurs.
    If fusion existed:
      Scenario 1: freq=0.99, cls=0.49 -> average=0.74 (> 0.5) -> would predict FAKE
      Scenario 2: freq=0.51, cls=0.49 -> average=0.50 -> would predict border
    Under categorical arbitration:
      BOTH scenarios MUST yield identical UNCERTAIN / DISAGREE outcomes because
      0.99 and 0.51 are both categorically 'fake', and 0.49 is categorically 'real'.
    """
    res1 = analyzer.arbitrate(
        frequency_result={"frequency_score": 0.99},
        classifier_result={"classifier_score": 0.49},
        metadata_result={"metadata_flag": "missing"},
    )
    res2 = analyzer.arbitrate(
        frequency_result={"frequency_score": 0.51},
        classifier_result={"classifier_score": 0.49},
        metadata_result={"metadata_flag": "missing"},
    )

    assert res1.verdict == "uncertain"
    assert res2.verdict == "uncertain"
    assert res1.decision_basis == res2.decision_basis == DecisionBasis.DISAGREE.value
    assert res1.output_state == res2.output_state == OutputState.UNCERTAIN_FREQUENCY_FLAGGED_ONLY.value


# =============================================================================
# 6. CONFIGURABLE THRESHOLDS & SCHEMA SERIALIZATION
# =============================================================================

def test_configurable_thresholds(analyzer):
    """Tests that per-agent thresholds can be customized dynamically."""
    # With threshold 0.80, a score of 0.60 is deemed REAL
    res = analyzer.arbitrate(
        frequency_result={"frequency_score": 0.60},
        classifier_result={"classifier_score": 0.60},
        metadata_result={"metadata_flag": "missing"},
        custom_freq_threshold=0.80,
        custom_classifier_threshold=0.80,
    )

    assert res.verdict == "real"
    assert res.output_state == OutputState.HIGH_CONFIDENCE_REAL.value


def test_schema_json_serializability(analyzer):
    """Validates that to_dict() returns all required schema fields."""
    res = analyzer.arbitrate(
        frequency_result={"frequency_score": 0.35},
        classifier_result={"classifier_score": 0.45},
        metadata_result={"metadata_flag": "missing"},
    )

    d = res.to_dict()
    assert "verdict" in d
    assert "confidence" in d
    assert "decision_basis" in d
    assert "conflict_type" in d
    assert "output_state" in d
    assert "reasoning" in d
    assert "scores" in d
    assert d["scores"]["frequency"] == 0.35
    assert d["scores"]["classifier"] == 0.45
    assert d["scores"]["metadata_flag"] == "missing"
    assert d["scores"]["disclosed_tool"] is None
