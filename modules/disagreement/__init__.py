"""
Module 4: Disagreement Analysis Layer for Explainable Deepfake Detection.
"""

from modules.disagreement.layer import DisagreementLayer
from modules.disagreement.schema import (
    DisagreementResult,
    DecisionBasis,
    ConflictType,
    OutputState,
    AgentScoresSnapshot,
)

__all__ = [
    "DisagreementLayer",
    "DisagreementResult",
    "DecisionBasis",
    "ConflictType",
    "OutputState",
    "AgentScoresSnapshot",
]
