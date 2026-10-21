"""
Region Analyzer for the Secondary System.
Reuses existing instances of ClassifierAgent and FrequencyAgent to evaluate
each extracted region without reloading models or creating redundant detectors.
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np

from modules.classifier.agent import ClassifierAgent
from modules.frequency.agent import FrequencyAgent

from .evidence_combiner import EvidenceCombiner
from .models import Region, RegionEvidence, RegionVerdict
from .region_preprocessor import RegionPreprocessor


class RegionAnalyzer:
    """
    Forensic region evaluation engine querying existing Primary System agents.
    """

    def __init__(
        self,
        classifier_agent: Optional[ClassifierAgent] = None,
        frequency_agent: Optional[FrequencyAgent] = None,
        evidence_combiner: Optional[EvidenceCombiner] = None,
        preprocessor: Optional[RegionPreprocessor] = None,
    ):
        # Reuse existing agents or instantiate once
        self.classifier_agent = classifier_agent or ClassifierAgent()
        self.frequency_agent = frequency_agent or FrequencyAgent()
        self.combiner = evidence_combiner or EvidenceCombiner()
        self.preprocessor = preprocessor or RegionPreprocessor()

    def analyze_regions(
        self,
        regions: List[Region],
        full_np_img: np.ndarray,
    ) -> List[RegionEvidence]:
        """
        Analyzes a list of regions against Classifier and Frequency agents.

        Args:
            regions: List of detected Region objects
            full_np_img: Full RGB image numpy array (H, W, 3)

        Returns:
            List of RegionEvidence objects with predictions and explanations
        """
        results: List[RegionEvidence] = []

        for region in regions:
            # 1. Preprocess region into matched RGB and Grayscale representations
            rgb_tensor, gray_array, err = self.preprocessor.preprocess_region(region, full_np_img)

            if err == RegionVerdict.INSUFFICIENT_SIZE.value:
                # Handle tiny region per Part 26 error handling
                results.append(RegionEvidence(
                    region_id=region.region_id,
                    region_type=region.region_type.value,
                    bbox=list(region.bbox),
                    classifier_prediction="UNCERTAIN",
                    real_probability=0.50,
                    fake_probability=0.50,
                    frequency_score=0.50,
                    frequency_anomaly=False,
                    agent_agreement=False,
                    agreement_type="INCONCLUSIVE",
                    combined_score=0.50,
                    region_verdict=RegionVerdict.INSUFFICIENT_SIZE.value,
                    confidence="LOW",
                    explanation="Insufficient region dimensions (< 16x16 px) for reliable forensic analysis.",
                ))
                continue

            ymin, xmin, ymax, xmax = region.bbox
            crop_rgb = full_np_img[ymin:ymax, xmin:xmax]

            # 2. Invoke Existing Classifier Agent on Region RGB Crop
            try:
                cls_res = self.classifier_agent.analyze(crop_rgb)
                fake_prob = float(cls_res.classifier_score)
                real_prob = float(cls_res.all_probabilities.get("REAL", 1.0 - fake_prob))
                cls_pred = str(cls_res.predicted_label)
            except Exception as e:
                fake_prob = 0.50
                real_prob = 0.50
                cls_pred = "UNCERTAIN"

            # 3. Invoke Existing Frequency Agent on Region Grayscale Array
            try:
                freq_res = self.frequency_agent.analyze(gray_array)
                freq_score = float(freq_res.frequency_score)
                freq_anomaly = bool(freq_res.is_suspicious)
            except Exception as e:
                freq_score = 0.50
                freq_anomaly = False

            # 4. Combine Evidence
            combined_score, agree, agree_type, verdict, conf_level, reasoning = self.combiner.combine(
                fake_prob=fake_prob,
                freq_score=freq_score,
                region_type=region.region_type.value,
            )

            results.append(RegionEvidence(
                region_id=region.region_id,
                region_type=region.region_type.value,
                bbox=list(region.bbox),
                classifier_prediction=cls_pred,
                real_probability=real_prob,
                fake_probability=fake_prob,
                frequency_score=freq_score,
                frequency_anomaly=freq_anomaly,
                agent_agreement=agree,
                agreement_type=agree_type.value,
                combined_score=combined_score,
                region_verdict=verdict.value,
                confidence=conf_level,
                explanation=reasoning,
            ))

        return results
