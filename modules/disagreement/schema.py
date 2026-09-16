"""
Schema definitions, enumerations, and data models for Module 4: Disagreement Analysis Layer.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Dict, Any


class DecisionBasis(str, Enum):
    """
    Four core decision basis states governing multi-agent arbitration.
    """
    AGREE = "AGREE"
    DISAGREE = "DISAGREE"
    PROVENANCE_OVERRIDE = "PROVENANCE_OVERRIDE"
    PROVENANCE_UNCERTAIN = "PROVENANCE_UNCERTAIN"


class ConflictType(str, Enum):
    """
    Specific conflict type when Category 1 probabilistic agents disagree.
    """
    FREQUENCY_FLAGGED_ONLY = "frequency_flagged_only"
    CLASSIFIER_FLAGGED_ONLY = "classifier_flagged_only"


class OutputState(str, Enum):
    """
    Exact four output states per Module 4 specification.
    """
    # STATE A: Both independent probabilistic agents agree; no provenance override
    HIGH_CONFIDENCE_REAL = "HIGH_CONFIDENCE_REAL"
    HIGH_CONFIDENCE_FAKE = "HIGH_CONFIDENCE_FAKE"

    # STATE B: Probabilistic agents disagree; no provenance override
    UNCERTAIN_FREQUENCY_FLAGGED_ONLY = "UNCERTAIN_frequency_flagged_only"
    UNCERTAIN_CLASSIFIER_FLAGGED_ONLY = "UNCERTAIN_classifier_flagged_only"

    # STATE C: Cryptographic provenance metadata proves AI generation (overrides visual models)
    PROVENANCE_OVERRIDE_FAKE = "PROVENANCE_OVERRIDE_FAKE"

    # STATE D: C2PA manifest container present but could not be parsed/matched
    PROVENANCE_UNCERTAIN = "PROVENANCE_UNCERTAIN"


@dataclass
class AgentScoresSnapshot:
    """
    Snapshot of upstream agent outputs for transparency and auditability.
    """
    frequency: float
    classifier: float
    metadata_flag: str
    disclosed_tool: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frequency": float(self.frequency),
            "classifier": float(self.classifier),
            "metadata_flag": str(self.metadata_flag),
            "disclosed_tool": self.disclosed_tool,
        }


@dataclass
class DisagreementResult:
    """
    Structured outcome of the Disagreement Analysis Layer.
    """
    # 1. Primary verdict: 'real', 'fake', or 'uncertain'
    verdict: str

    # 2. Confidence level: 'high' or 'low'
    confidence: str

    # 3. Decision basis: 'AGREE', 'DISAGREE', 'PROVENANCE_OVERRIDE', or 'PROVENANCE_UNCERTAIN'
    decision_basis: str

    # 4. Conflict type: 'frequency_flagged_only', 'classifier_flagged_only', or None
    conflict_type: Optional[str]

    # 5. Full explainable reasoning (state-specific, non-templated)
    reasoning: str

    # 6. Exact output state label (State A, B, C, or D)
    output_state: str

    # 7. Raw upstream scores snapshot
    scores: AgentScoresSnapshot

    # 8. Upstream individual verdicts (for audit)
    freq_verdict: str = ""
    classifier_verdict: str = ""
    probabilistic_state: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes result into the exact required JSON-compatible output structure.
        """
        return {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "decision_basis": self.decision_basis,
            "conflict_type": self.conflict_type,
            "output_state": self.output_state,
            "reasoning": self.reasoning,
            "scores": self.scores.to_dict(),
        }
