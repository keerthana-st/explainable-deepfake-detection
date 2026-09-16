"""
Disagreement Analysis Layer (Module 4) for Explainable Deepfake Detection.

This module directly compares agent outputs without mathematical score fusion,
implementing a strict two-stage decision logic:
1. Category 1: Independent thresholding and comparison of probabilistic vision agents.
2. Category 2: Subsequent cryptographic provenance override verification.
"""

from typing import Optional, Dict, Any, Union
import logging

from modules.disagreement.schema import (
    DecisionBasis,
    ConflictType,
    OutputState,
    AgentScoresSnapshot,
    DisagreementResult,
)
from modules.disagreement.reasoning import build_disagreement_reasoning

logger = logging.getLogger(__name__)


class DisagreementLayer:
    """
    Arbitration layer that evaluates multi-agent evidence categorically.
    
    GUARANTEES:
    - Never averages, weights, or mathematically fuses scores into a single blended number.
    - Treats Category 1 (probabilistic models) and Category 2 (deterministic provenance) as distinct evidence tiers.
    - Eliminates UI contradictions by driving all display labels from a single decision basis.
    """

    def __init__(
        self,
        default_frequency_threshold: float = 0.50,
        default_classifier_threshold: float = 0.50,
    ):
        """
        Initializes the DisagreementLayer with calibratable, independent decision thresholds.

        Args:
            default_frequency_threshold: Threshold above which frequency energy is deemed fake/suspicious.
            default_classifier_threshold: Threshold above which classifier probability is deemed fake.
        """
        self.freq_threshold = float(default_frequency_threshold)
        self.classifier_threshold = float(default_classifier_threshold)

    def arbitrate(
        self,
        frequency_result: Any,
        classifier_result: Any,
        metadata_result: Any,
        custom_freq_threshold: Optional[float] = None,
        custom_classifier_threshold: Optional[float] = None,
    ) -> DisagreementResult:
        """
        Executes the full 4-step decision logic on upstream agent results.

        Accepts either module result objects (FrequencyAnalysisResult, ClassifierResult, MetadataResult)
        or raw dictionaries/scalars.

        Returns:
            DisagreementResult containing verdict, confidence, decision_basis, conflict_type,
            output_state, non-templated reasoning, and scores snapshot.
        """
        # ---------------------------------------------------------------------
        # Extract raw scores and metadata strings cleanly from input objects
        # ---------------------------------------------------------------------
        freq_score = self._extract_float(frequency_result, ["frequency_score", "score", "raw_score"])
        cls_score = self._extract_float(classifier_result, ["classifier_score", "probability_fake", "score"])
        
        meta_flag = self._extract_str(metadata_result, ["metadata_flag", "flag"], default="missing")
        disclosed_tool = self._extract_optional_str(metadata_result, ["disclosed_tool", "tool_name", "tool"])
        manifest_status = self._extract_optional_str(metadata_result, ["validation_status", "status"])

        # Determine thresholds to apply
        f_thresh = float(custom_freq_threshold) if custom_freq_threshold is not None else self.freq_threshold
        c_thresh = float(custom_classifier_threshold) if custom_classifier_threshold is not None else self.classifier_threshold

        # ---------------------------------------------------------------------
        # STEP 1: Compare Frequency and Classifier scores independently
        # (STRICTLY NO MATHEMATICAL AVERAGE OR WEIGHTED BLEND)
        # ---------------------------------------------------------------------
        freq_verdict = "fake" if freq_score > f_thresh else "real"
        classifier_verdict = "fake" if cls_score > c_thresh else "real"

        # ---------------------------------------------------------------------
        # STEP 2: Determine Category 1 (probabilistic) state
        # ---------------------------------------------------------------------
        if freq_verdict == classifier_verdict:
            probabilistic_state = DecisionBasis.AGREE.value
            probabilistic_verdict = freq_verdict
            conflict_type = None
        else:
            probabilistic_state = DecisionBasis.DISAGREE.value
            probabilistic_verdict = "uncertain"
            if freq_verdict == "fake":
                conflict_type = ConflictType.FREQUENCY_FLAGGED_ONLY.value
            else:
                conflict_type = ConflictType.CLASSIFIER_FLAGGED_ONLY.value

        # ---------------------------------------------------------------------
        # STEP 3: Apply Category 2 (metadata) check as a SEPARATE, SUBSEQUENT step
        # ---------------------------------------------------------------------
        normalized_flag = meta_flag.lower().strip()
        has_ai_disclosed = (normalized_flag == "ai_disclosed") and bool(disclosed_tool)
        is_manifest_unclear = (normalized_flag == "manifest_present_generator_unclear")

        if has_ai_disclosed:
            final_verdict = "fake"
            final_confidence = "high"
            decision_basis = DecisionBasis.PROVENANCE_OVERRIDE.value
        elif is_manifest_unclear:
            final_verdict = probabilistic_verdict
            final_confidence = "high" if probabilistic_state == DecisionBasis.AGREE.value else "low"
            decision_basis = DecisionBasis.PROVENANCE_UNCERTAIN.value
        else:
            final_verdict = probabilistic_verdict
            final_confidence = "high" if probabilistic_state == DecisionBasis.AGREE.value else "low"
            decision_basis = probabilistic_state

        # ---------------------------------------------------------------------
        # STEP 4: Construct the exact four possible OUTPUT STATES
        # ---------------------------------------------------------------------
        if decision_basis == DecisionBasis.PROVENANCE_OVERRIDE.value:
            output_state = OutputState.PROVENANCE_OVERRIDE_FAKE.value

        elif decision_basis == DecisionBasis.PROVENANCE_UNCERTAIN.value:
            output_state = OutputState.PROVENANCE_UNCERTAIN.value

        elif decision_basis == DecisionBasis.AGREE.value:
            if final_verdict == "fake":
                output_state = OutputState.HIGH_CONFIDENCE_FAKE.value
            else:
                output_state = OutputState.HIGH_CONFIDENCE_REAL.value

        elif decision_basis == DecisionBasis.DISAGREE.value:
            if conflict_type == ConflictType.FREQUENCY_FLAGGED_ONLY.value:
                output_state = OutputState.UNCERTAIN_FREQUENCY_FLAGGED_ONLY.value
            else:
                output_state = OutputState.UNCERTAIN_CLASSIFIER_FLAGGED_ONLY.value
        else:
            output_state = "UNCERTAIN"

        # Construct non-templated explainable reasoning
        reasoning = build_disagreement_reasoning(
            output_state=output_state,
            freq_verdict=freq_verdict,
            classifier_verdict=classifier_verdict,
            freq_score=freq_score,
            classifier_score=cls_score,
            metadata_flag=meta_flag,
            disclosed_tool=disclosed_tool,
            raw_manifest_status=manifest_status,
        )

        scores_snapshot = AgentScoresSnapshot(
            frequency=freq_score,
            classifier=cls_score,
            metadata_flag=meta_flag,
            disclosed_tool=disclosed_tool,
        )

        return DisagreementResult(
            verdict=final_verdict,
            confidence=final_confidence,
            decision_basis=decision_basis,
            conflict_type=conflict_type,
            output_state=output_state,
            reasoning=reasoning,
            scores=scores_snapshot,
            freq_verdict=freq_verdict,
            classifier_verdict=classifier_verdict,
            probabilistic_state=probabilistic_state,
        )

    def format_ui_report(self, result: DisagreementResult) -> str:
        """
        Renders an internally consistent, contradiction-proof UI report.
        
        CRITICAL: Verdict header, confidence badge, and conflict breakdown text
        are strictly derived from the same decision_basis and output_state properties.
        """
        basis = result.decision_basis
        state = result.output_state
        verdict_upper = result.verdict.upper()
        conf_upper = result.confidence.upper()

        lines = [
            "=" * 78,
            "[DISAGREEMENT ANALYSIS LAYER: MULTI-AGENT CROSS-EVALUATION]",
            "=" * 78,
        ]

        # 1. Unified Status & Badge
        if basis == DecisionBasis.PROVENANCE_OVERRIDE.value:
            header_text = "VERDICT: FAKE / SYNTHETIC (PROVENANCE OVERRIDE)"
            badge_text = f"CONFIDENCE: {conf_upper} | BASIS: PROVENANCE_OVERRIDE"
            conflict_summary = (
                f"• OVERRIDE STATUS: CRYPTOGRAPHIC PROVENANCE OVERRIDE ACTIVE\n"
                f"  Factual provenance credentials disclose synthesis by '{result.scores.disclosed_tool}'.\n"
                f"  Probabilistic signals (Frequency: {result.freq_verdict.upper()}, "
                f"Classifier: {result.classifier_verdict.upper()}) were overridden by cryptographic fact."
            )

        elif basis == DecisionBasis.PROVENANCE_UNCERTAIN.value:
            header_text = f"VERDICT: {verdict_upper} (PROVENANCE UNCERTAIN)"
            badge_text = f"CONFIDENCE: {conf_upper} | BASIS: PROVENANCE_UNCERTAIN"
            conflict_summary = (
                f"• PROVENANCE STATUS: Unresolved C2PA manifest present in file.\n"
                f"  Probabilistic Consensus: Frequency ({result.freq_verdict.upper()}) vs. "
                f"Classifier ({result.classifier_verdict.upper()})."
            )

        elif basis == DecisionBasis.AGREE.value:
            header_text = f"VERDICT: {verdict_upper} (HIGH CONFIDENCE)"
            badge_text = f"CONFIDENCE: {conf_upper} | BASIS: CONVERGENT_PROBABILISTIC_AGREEMENT"
            conflict_summary = (
                f"• CONSENSUS STATUS: Both Category 1 probabilistic vision agents agree on {verdict_upper}.\n"
                f"  - Frequency Agent: {result.freq_verdict.upper()} (Score: {result.scores.frequency:.4f})\n"
                f"  - Classifier Agent: {result.classifier_verdict.upper()} (Score: {result.scores.classifier:.4f})"
            )

        else:  # DISAGREE
            header_text = "VERDICT: UNCERTAIN (MANUAL REVIEW REQUIRED)"
            badge_text = f"CONFIDENCE: {conf_upper} | BASIS: AGENT_DISAGREEMENT"
            conflict_summary = (
                f"• CONFLICT BREAKDOWN: Category 1 probabilistic agents are in active disagreement:\n"
                f"  - Frequency Agent: {result.freq_verdict.upper()} (Score: {result.scores.frequency:.4f})\n"
                f"  - Classifier Agent: {result.classifier_verdict.upper()} (Score: {result.scores.classifier:.4f})\n"
                f"  - Conflict Type: {result.conflict_type}"
            )

        lines.append(f"  • {header_text}")
        lines.append(f"  • [{badge_text}]")
        lines.append(f"  • Output State:        {state}")
        lines.append("")
        lines.append(f"{conflict_summary}")
        lines.append("")
        lines.append("Explainable Arbitration Reasoning:")
        for r_line in result.reasoning.splitlines():
            lines.append(f"  {r_line}")
        lines.append("=" * 78)

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Safe Attribute Extractors
    # -------------------------------------------------------------------------
    def _extract_float(self, obj: Any, candidate_attrs: list[str]) -> float:
        if isinstance(obj, (int, float)):
            return float(obj)
        if isinstance(obj, dict):
            for attr in candidate_attrs:
                if attr in obj and obj[attr] is not None:
                    return float(obj[attr])
        for attr in candidate_attrs:
            if hasattr(obj, attr):
                val = getattr(obj, attr)
                if val is not None:
                    return float(val)
        return 0.0

    def _extract_str(self, obj: Any, candidate_attrs: list[str], default: str = "") -> str:
        if isinstance(obj, str):
            return obj
        if isinstance(obj, dict):
            for attr in candidate_attrs:
                if attr in obj and obj[attr] is not None:
                    return str(obj[attr])
        for attr in candidate_attrs:
            if hasattr(obj, attr):
                val = getattr(obj, attr)
                if val is not None:
                    return str(val)
        return default

    def _extract_optional_str(self, obj: Any, candidate_attrs: list[str]) -> Optional[str]:
        if isinstance(obj, dict):
            for attr in candidate_attrs:
                if attr in obj and obj[attr]:
                    return str(obj[attr])
        for attr in candidate_attrs:
            if hasattr(obj, attr):
                val = getattr(obj, attr)
                if val:
                    return str(val)
        return None
