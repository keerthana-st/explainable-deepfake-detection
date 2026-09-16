"""
Module 2: Classifier Agent (Explainable Deepfake Detection).
"""

from .agent import (
    ClassifierAgent,
    ClassifierAnalysisResult,
    VLM_HALLUCINATION_DISCLAIMER,
)
from .dataset import (
    generate_four_category_benchmark,
    verify_dataset_label_order,
)
from .evaluation import (
    CategoryConfusionMatrix,
    CategoryMetrics,
    evaluate_classifier_agent,
)
from .model import (
    DeepfakeClassifierModel,
    get_default_transform,
    resolve_label_mapping,
)

__all__ = [
    "ClassifierAgent",
    "ClassifierAnalysisResult",
    "DeepfakeClassifierModel",
    "resolve_label_mapping",
    "get_default_transform",
    "generate_four_category_benchmark",
    "verify_dataset_label_order",
    "evaluate_classifier_agent",
    "CategoryConfusionMatrix",
    "CategoryMetrics",
    "VLM_HALLUCINATION_DISCLAIMER",
]
