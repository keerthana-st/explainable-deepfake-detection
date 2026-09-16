"""
Evaluation Engine for Classifier Agent (Module 2).

Generates category-separated confusion matrices across the 4 mandatory test
categories:
1. Known-real ordinary photos
2. Known GAN-based fakes
3. Known diffusion-based fakes (ChatGPT/DALL-E 3, Midjourney)
4. Compressed/re-saved real photos

Per project specifications, reports separate category metrics rather than a single
blended aggregate that could hide category-specific failures.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

from .agent import ClassifierAgent


@dataclass
class CategoryMetrics:
    category_name: str
    ground_truth: str  # 'REAL' or 'FAKE'
    total_samples: int
    correct_predictions: int
    incorrect_predictions: int
    accuracy: float
    mean_confidence: float
    predicted_real_count: int
    predicted_fake_count: int


@dataclass
class CategoryConfusionMatrix:
    categories: Dict[str, CategoryMetrics]

    def print_report(self):
        print("\n" + "=" * 78)
        print("         CLASSIFIER AGENT: 4-CATEGORY CONFUSION MATRIX REPORT")
        print("=" * 78)
        header = f"{'Category Name':<22} | {'Truth':<5} | {'N':<4} | {'Pred REAL':<9} | {'Pred FAKE':<9} | {'Accuracy':<8} | {'Mean P(Fake)':<11}"
        print(header)
        print("-" * 78)
        for cat_name, m in self.categories.items():
            row = (
                f"{m.category_name:<22} | {m.ground_truth:<5} | {m.total_samples:<4} | "
                f"{m.predicted_real_count:<9} | {m.predicted_fake_count:<9} | "
                f"{m.accuracy * 100.0:>7.1f}% | {m.mean_confidence:>11.4f}"
            )
            print(row)
        print("=" * 78)
        print("KEY INSIGHTS & FAILURE MODE AUDIT:")
        for cat_name, m in self.categories.items():
            if m.ground_truth == "FAKE":
                fnr = (m.incorrect_predictions / m.total_samples) * 100.0
                print(f"  • {cat_name}: Recall = {m.accuracy * 100.0:.1f}%, Miss Rate (FNR) = {fnr:.1f}%")
            else:
                fpr = (m.incorrect_predictions / m.total_samples) * 100.0
                print(f"  • {cat_name}: Specificity = {m.accuracy * 100.0:.1f}%, False Alarm (FPR) = {fpr:.1f}%")
        print("=" * 78)


def evaluate_classifier_agent(
    agent: ClassifierAgent,
    benchmark_paths: Dict[str, List[Union[str, Path]]],
) -> CategoryConfusionMatrix:
    """
    Evaluate agent across all 4 mandatory categories.

    Parameters
    ----------
    agent : ClassifierAgent
        Initialized classifier agent.
    benchmark_paths : Dict[str, List[Path]]
        Dictionary with keys 'real_ordinary', 'fake_gan', 'fake_diffusion', 'real_compressed'.

    Returns
    -------
    CategoryConfusionMatrix
    """
    truth_mapping = {
        "real_ordinary": "REAL",
        "fake_gan": "FAKE",
        "fake_diffusion": "FAKE",
        "real_compressed": "REAL",
    }

    results: Dict[str, CategoryMetrics] = {}

    for cat_key, file_list in benchmark_paths.items():
        truth = truth_mapping.get(cat_key, "REAL")
        pred_real = 0
        pred_fake = 0
        conf_list = []

        for p in file_list:
            res = agent.analyze(p)
            conf_list.append(res.classifier_score)
            if res.predicted_label == "REAL":
                pred_real += 1
            else:
                pred_fake += 1

        total = len(file_list)
        correct = pred_real if truth == "REAL" else pred_fake
        incorrect = total - correct
        acc = correct / total if total > 0 else 0.0
        mean_conf = float(np.mean(conf_list)) if conf_list else 0.0

        results[cat_key] = CategoryMetrics(
            category_name=cat_key,
            ground_truth=truth,
            total_samples=total,
            correct_predictions=correct,
            incorrect_predictions=incorrect,
            accuracy=acc,
            mean_confidence=mean_conf,
            predicted_real_count=pred_real,
            predicted_fake_count=pred_fake,
        )

    return CategoryConfusionMatrix(categories=results)
