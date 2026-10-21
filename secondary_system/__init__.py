"""
Secondary Region-Level Analysis System.
Provides localized forensic segmentation, regional evidence combination,
global-to-regional consistency analysis, and explainability visualizations.
"""

from .models import (
    AgreementType,
    ConsistencyStatus,
    Region,
    RegionEvidence,
    RegionalConsistencyResult,
    RegionType,
    RegionVerdict,
    SecondaryAnalysisResult,
    SecondaryTriggerMode,
)

__all__ = [
    "AgreementType",
    "ConsistencyStatus",
    "Region",
    "RegionEvidence",
    "RegionalConsistencyResult",
    "RegionType",
    "RegionVerdict",
    "SecondaryAnalysisResult",
    "SecondaryTriggerMode",
]
