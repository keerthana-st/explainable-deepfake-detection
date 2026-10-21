"""
Evidence Combination and Regional Verdict Logic for the Secondary System.
Combines Classifier fake probability and Frequency anomaly score,
evaluates agent agreement, and assigns explainable regional verdicts.
"""

from typing import Optional, Tuple
from .models import AgreementType, RegionVerdict


class EvidenceCombiner:
    """
    Combines multi-agent evidence for an individual spatial region.
    """

    def __init__(
        self,
        classifier_weight: float = 0.60,
        frequency_weight: float = 0.40,
        fake_threshold: float = 0.70,
        real_threshold: float = 0.30,
        agent_decision_threshold: float = 0.50,
    ):
        # Normalize weights so they sum to 1.0
        total_w = classifier_weight + frequency_weight
        self.w_cls = classifier_weight / total_w if total_w > 0 else 0.60
        self.w_freq = frequency_weight / total_w if total_w > 0 else 0.40

        self.fake_threshold = fake_threshold
        self.real_threshold = real_threshold
        self.agent_thresh = agent_decision_threshold

    def combine(
        self,
        fake_prob: Optional[float] = None,
        freq_score: Optional[float] = None,
        region_type: str = "region",
        classifier_score: Optional[float] = None,
        frequency_score: Optional[float] = None,
        is_frequency_suspicious: Optional[bool] = None,
        area_pixels: Optional[int] = None,
        **kwargs,
    ) -> Tuple[float, bool, AgreementType, RegionVerdict, str, str]:
        """
        Calculates combined score, agreement level, verdict, confidence, and reasoning.

        Args:
            fake_prob: Classifier P(Fake) in [0.0, 1.0] (or classifier_score)
            freq_score: Frequency Agent anomaly score in [0.0, 1.0] (or frequency_score)
            region_type: Semantic name of the region (e.g., 'face', 'clothing')

        Returns:
            combined_score: Weighted combination in [0.0, 1.0]
            agent_agreement: True if both agents concordantly agree
            agreement_type: AgreementType enum
            verdict: RegionVerdict enum
            confidence_level: 'HIGH', 'MEDIUM', 'LOW'
            explanation: Forensic explanation text
        """
        if fake_prob is None and classifier_score is not None:
            fake_prob = classifier_score
        if freq_score is None and frequency_score is not None:
            freq_score = frequency_score
        if fake_prob is None:
            fake_prob = 0.5
        if freq_score is None:
            freq_score = 0.5
        # 1. Weighted linear combination
        combined_score = (self.w_cls * fake_prob) + (self.w_freq * freq_score)
        combined_score = float(max(0.0, min(1.0, combined_score)))

        cls_calls_fake = fake_prob >= self.agent_thresh
        freq_calls_fake = freq_score >= self.agent_thresh

        # 2. Check Agreement
        if cls_calls_fake == freq_calls_fake:
            agent_agreement = True
            agreement_type = AgreementType.STRONG_AGREEMENT
            confidence_level = "HIGH"
        else:
            agent_agreement = False
            agreement_type = AgreementType.DISAGREEMENT
            confidence_level = "LOW"

        # 3. Determine Region Verdict (Strictly Fake or Real per user requirement)
        if combined_score >= 0.50:
            verdict = RegionVerdict.AI_EDITED
            explanation = (
                f"Synthetic manipulation detected in {region_type}: "
                f"Classifier fake probability {fake_prob*100:.1f}%, "
                f"Frequency anomaly score {freq_score*100:.1f}% (Combined: {combined_score*100:.1f}%)."
            )
        else:
            verdict = RegionVerdict.REAL
            explanation = (
                f"Authentic characteristics in {region_type}: "
                f"Classifier fake probability {fake_prob*100:.1f}%, "
                f"Frequency anomaly score {freq_score*100:.1f}% (Combined: {combined_score*100:.1f}%)."
            )

        return combined_score, agent_agreement, agreement_type, verdict, confidence_level, explanation
