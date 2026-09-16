"""
Module 1: Frequency Agent (Explainable Deepfake Detection).
"""

from .agent import (
    DEFAULT_CALIBRATION_DATE,
    DEFAULT_CALIBRATION_DATASET,
    DEFAULT_CALIBRATION_DIVISOR,
    DEFAULT_DECISION_THRESHOLD,
    DIFFUSION_BLIND_SPOT_DISCLOSURE,
    FrequencyAgent,
    FrequencyAnalysisResult,
    KNOWN_FALSE_POSITIVE_SOURCES,
)
from .calibration import (
    CalibrationMetadata,
    calibrate_from_scores,
    compute_roc_and_optimal_threshold,
    generate_benchmark_validation_set,
)
from .visualization import generate_frequency_heatmap

__all__ = [
    "FrequencyAgent",
    "FrequencyAnalysisResult",
    "CalibrationMetadata",
    "generate_frequency_heatmap",
    "compute_roc_and_optimal_threshold",
    "calibrate_from_scores",
    "generate_benchmark_validation_set",
    "DEFAULT_CALIBRATION_DIVISOR",
    "DEFAULT_DECISION_THRESHOLD",
    "DEFAULT_CALIBRATION_DATASET",
    "DEFAULT_CALIBRATION_DATE",
    "KNOWN_FALSE_POSITIVE_SOURCES",
    "DIFFUSION_BLIND_SPOT_DISCLOSURE",
]
