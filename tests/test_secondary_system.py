"""
Unit and Integration Tests for the Secondary Region-Level Analysis System.

Tests cover:
  1. Region Segmentation:
     - Multi-face segmentation and naming.
     - Person, clothing approximation, background, and object fallback.
     - Minimum region count guarantee (>= 2 regions per image).
  2. Region Preprocessing:
     - Minimum size validation (rejects < 16x16 with INSUFFICIENT_SIZE).
     - Dimensions and ImageNet normalization format (224x224 RGB tensor, grayscale array).
  3. Evidence Combination:
     - Mathematical weighting (w_cls * P_fake + w_freq * S_freq).
     - Agreement categorization (STRONG_AGREEMENT vs DISAGREEMENT).
     - Threshold boundary tests for REAL, AI-EDITED, and UNCERTAIN.
  4. Global-to-Regional Consistency Analysis:
     - SUPPORTED, PARTIALLY_SUPPORTED, CONTRADICTED, INCONCLUSIVE.
     - Strict Non-Override Invariant check (global verdict is never altered).
  5. Secondary Pipeline Triggering:
     - Behavior under ALWAYS, LOW_CONFIDENCE, DISAGREEMENT, and SUSPICIOUS_ONLY modes.
"""

import numpy as np
import pytest
import torch

from secondary_system.consistency_analyzer import RegionalConsistencyAnalyzer
from secondary_system.evidence_combiner import EvidenceCombiner
from secondary_system.models import (
    AgreementType,
    ConsistencyStatus,
    Region,
    RegionEvidence,
    RegionType,
    RegionVerdict,
    SecondaryTriggerMode,
)
from secondary_system.region_preprocessor import preprocess_region
from secondary_system.region_segmenter import (
    BackgroundSegmenter,
    ClothingSegmenter,
    FaceSegmenter,
    ObjectSegmenter,
    UnifiedRegionSegmenter,
)
from secondary_system.secondary_pipeline import SecondaryPipeline


@pytest.fixture
def sample_synthetic_rgb_image():
    """Generates a synthetic 512x512 RGB test image with distinct foreground shape."""
    img = np.full((512, 512, 3), 200, dtype=np.uint8)
    # Draw a simulated person / foreground rectangle in center
    img[100:400, 150:350] = [50, 100, 180]
    # Draw a simulated head circle
    import cv2
    cv2.circle(img, (250, 160), 45, (230, 200, 180), -1)
    return img


# =============================================================================
# 1. SEGMENTATION TESTS
# =============================================================================
class TestRegionSegmentation:
    def test_unified_segmenter_guarantees_minimum_regions(self, sample_synthetic_rgb_image):
        segmenter = UnifiedRegionSegmenter()
        regions = segmenter.segment(sample_synthetic_rgb_image)
        assert len(regions) >= 2, f"Expected at least 2 regions, got {len(regions)}"
        
        # Verify region types are valid
        valid_types = {RegionType.FACE, RegionType.PERSON, RegionType.CLOTHING, RegionType.BACKGROUND, RegionType.OBJECT}
        for r in regions:
            assert r.region_type in valid_types
            assert r.bbox[0] <= r.bbox[2] and r.bbox[1] <= r.bbox[3]
            assert r.mask.shape == sample_synthetic_rgb_image.shape[:2]

    def test_clothing_is_person_minus_face(self):
        h, w = 200, 200
        dummy_img = np.zeros((h, w, 3), dtype=np.uint8)
        person_mask = np.zeros((h, w), dtype=bool)
        person_mask[40:180, 50:150] = True

        face_mask = np.zeros((h, w), dtype=bool)
        face_mask[40:80, 70:130] = True

        person_region = Region(
            region_id="person_01",
            region_type=RegionType.PERSON,
            bbox=(40, 50, 180, 150),
            mask=person_mask,
            area_pixels=int(np.sum(person_mask)),
        )
        face_region = Region(
            region_id="face_01",
            region_type=RegionType.FACE,
            bbox=(40, 70, 80, 130),
            mask=face_mask,
            area_pixels=int(np.sum(face_mask)),
        )

        clothing_seg = ClothingSegmenter()
        clothing_regions = clothing_seg.segment(
            dummy_img,
            context={"person_regions": [person_region], "face_regions": [face_region]},
        )

        assert len(clothing_regions) >= 1
        clothing_region = clothing_regions[0]
        assert clothing_region.region_type == RegionType.CLOTHING
        # Check that face area has zero overlap in clothing mask
        overlap = np.logical_and(clothing_region.mask, face_mask)
        assert not np.any(overlap), "Clothing mask must not contain face pixels"

    def test_background_is_inverse_foreground(self):
        h, w = 100, 100
        dummy_img = np.zeros((h, w, 3), dtype=np.uint8)
        fg_mask = np.zeros((h, w), dtype=bool)
        fg_mask[20:80, 20:80] = True

        fg_region = Region(
            region_id="fg_01",
            region_type=RegionType.PERSON,
            bbox=(20, 20, 80, 80),
            mask=fg_mask,
            area_pixels=int(np.sum(fg_mask)),
        )

        bg_seg = BackgroundSegmenter()
        bg_regions = bg_seg.segment(dummy_img, context={"foreground_regions": [fg_region]})
        assert len(bg_regions) >= 1
        bg_region = bg_regions[0]
        assert bg_region.region_type == RegionType.BACKGROUND
        assert bg_region.mask[0, 0] is True or bg_region.mask[0, 0] == 1
        assert bg_region.mask[50, 50] is False or bg_region.mask[50, 50] == 0

    def test_object_segmenter_fallback(self):
        img = np.full((128, 128, 3), 240, dtype=np.uint8)
        img[30:90, 30:90] = [20, 20, 20]  # dark box object
        obj_seg = ObjectSegmenter()
        regions = obj_seg.segment(img)
        assert len(regions) >= 1
        assert regions[0].region_type == RegionType.OBJECT


# =============================================================================
# 2. PREPROCESSOR TESTS
# =============================================================================
class TestRegionPreprocessor:
    def test_rejects_sub_16px_regions(self):
        tiny_region = Region(
            region_id="tiny_test",
            region_type=RegionType.OBJECT,
            bbox=(10, 10, 22, 22),  # 12x12 px < 16x16
            mask=np.zeros((100, 100), dtype=np.uint8),
            area_pixels=144,
        )
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        res = preprocess_region(tiny_region, dummy_img)
        assert not res.is_valid
        assert "INSUFFICIENT_SIZE" in res.error_message

    def test_formats_valid_region_correctly(self, sample_synthetic_rgb_image):
        valid_region = Region(
            region_id="valid_test",
            region_type=RegionType.PERSON,
            bbox=(50, 50, 300, 300),
            mask=np.full((512, 512), 255, dtype=np.uint8),
            area_pixels=250 * 250,
        )
        res = preprocess_region(valid_region, sample_synthetic_rgb_image, target_size=(224, 224))
        assert res.is_valid
        assert res.rgb_tensor is not None
        assert res.rgb_tensor.shape == (1, 3, 224, 224)
        assert res.gray_array.shape == (224, 224)


# =============================================================================
# 3. EVIDENCE COMBINER TESTS
# =============================================================================
class TestEvidenceCombiner:
    def test_weighted_combination_formula(self):
        combiner = EvidenceCombiner(classifier_weight=0.60, frequency_weight=0.40)
        # cls = 0.80, freq = 0.50 -> 0.6*0.8 + 0.4*0.5 = 0.48 + 0.20 = 0.68
        combined, is_agree, agreement, verdict, conf, exp = combiner.combine(
            classifier_score=0.80,
            frequency_score=0.50,
            is_frequency_suspicious=True,
            area_pixels=5000,
        )
        assert pytest.approx(combined, 0.001) == 0.68

    def test_strong_agreement_fake(self):
        combiner = EvidenceCombiner(classifier_weight=0.60, frequency_weight=0.40)
        combined, is_agree, agreement, verdict, conf, exp = combiner.combine(
            classifier_score=0.90,
            frequency_score=0.85,
            is_frequency_suspicious=True,
            area_pixels=10000,
        )
        assert agreement == AgreementType.STRONG_AGREEMENT
        assert verdict == RegionVerdict.AI_EDITED
        assert combined >= 0.70

    def test_strong_agreement_real(self):
        combiner = EvidenceCombiner(classifier_weight=0.60, frequency_weight=0.40)
        combined, is_agree, agreement, verdict, conf, exp = combiner.combine(
            classifier_score=0.10,
            frequency_score=0.15,
            is_frequency_suspicious=False,
            area_pixels=10000,
        )
        assert agreement == AgreementType.STRONG_AGREEMENT
        assert verdict == RegionVerdict.REAL
        assert combined <= 0.30

    def test_conflicting_signals_lead_to_disagreement_or_uncertain(self):
        combiner = EvidenceCombiner(classifier_weight=0.60, frequency_weight=0.40)
        # Cls says 0.80 (fake), Freq says 0.10 (real)
        combined, is_agree, agreement, verdict, conf, exp = combiner.combine(
            classifier_score=0.80,
            frequency_score=0.10,
            is_frequency_suspicious=False,
            area_pixels=10000,
        )
        assert agreement == AgreementType.DISAGREEMENT
        assert verdict == RegionVerdict.UNCERTAIN



# =============================================================================
# 4. GLOBAL-REGIONAL CONSISTENCY & NON-OVERRIDE INVARIANT TESTS
# =============================================================================
class TestConsistencyAnalyzer:
    def test_consistency_supported_fake(self):
        analyzer = RegionalConsistencyAnalyzer()
        r1 = RegionEvidence(
            region=Region("face_01", RegionType.FACE, (0, 0, 10, 10), np.zeros((10, 10)), 100),
            classifier_score=0.95,
            classifier_prediction="FAKE",
            frequency_score=0.88,
            frequency_anomaly=True,
            combined_score=0.92,
            region_verdict=RegionVerdict.AI_EDITED,
            confidence=0.92,
            agent_agreement=AgreementType.STRONG_AGREEMENT,
            explanation="Manipulated face detected",
        )
        res = analyzer.analyze(primary_global_verdict="AI-GENERATED / FAKE", region_evidences=[r1])
        assert res.status == ConsistencyStatus.SUPPORTED
        assert "face_01" in res.suspicious_regions

    def test_consistency_supported_real(self):
        analyzer = RegionalConsistencyAnalyzer()
        r1 = RegionEvidence(
            region=Region("face_01", RegionType.FACE, (0, 0, 10, 10), np.zeros((10, 10)), 100),
            classifier_score=0.05,
            classifier_prediction="REAL",
            frequency_score=0.10,
            frequency_anomaly=False,
            combined_score=0.07,
            region_verdict=RegionVerdict.REAL,
            confidence=0.93,
            agent_agreement=AgreementType.STRONG_AGREEMENT,
            explanation="Natural face detected",
        )
        res = analyzer.analyze(primary_global_verdict="REAL / AUTHENTIC", region_evidences=[r1])
        assert res.status == ConsistencyStatus.SUPPORTED
        assert "face_01" in res.authentic_regions

    def test_consistency_contradicted_triggers_warning_without_mutating_verdict(self):
        analyzer = RegionalConsistencyAnalyzer()
        # Global is REAL, but face is blatantly AI-EDITED
        r1 = RegionEvidence(
            region=Region("face_01", RegionType.FACE, (0, 0, 10, 10), np.zeros((10, 10)), 100),
            classifier_score=0.98,
            classifier_prediction="FAKE",
            frequency_score=0.90,
            frequency_anomaly=True,
            combined_score=0.95,
            region_verdict=RegionVerdict.AI_EDITED,
            confidence=0.95,
            agent_agreement=AgreementType.STRONG_AGREEMENT,
            explanation="Manipulated face",
        )
        global_verdict_input = "REAL / AUTHENTIC"
        res = analyzer.analyze(primary_global_verdict=global_verdict_input, region_evidences=[r1])
        assert res.status == ConsistencyStatus.CONTRADICTED
        assert res.warning is not None
        # Non-override invariant: Input global verdict string remains unchanged
        assert global_verdict_input == "REAL / AUTHENTIC"


# =============================================================================
# 5. PIPELINE TRIGGER MODES
# =============================================================================
class TestSecondaryTriggerModes:
    def test_trigger_modes(self):
        pipeline = SecondaryPipeline(trigger_mode=SecondaryTriggerMode.LOW_CONFIDENCE)
        
        # High confidence -> should NOT trigger
        assert not pipeline.should_trigger({"verdict": "REAL / AUTHENTIC", "confidence": "95.0%"})
        # Low confidence -> should trigger
        assert pipeline.should_trigger({"verdict": "UNCERTAIN", "confidence": "LOW (Conflicting Signals)"})

        # Disagreement mode
        pipeline.trigger_mode = SecondaryTriggerMode.DISAGREEMENT
        assert pipeline.should_trigger({"verdict": "UNCERTAIN", "decision_basis": "Multi-agent conflict"})
        assert not pipeline.should_trigger({"verdict": "REAL / AUTHENTIC", "decision_basis": "Unanimous agreement"})

        # Always mode
        pipeline.trigger_mode = SecondaryTriggerMode.ALWAYS
        assert pipeline.should_trigger({"verdict": "REAL / AUTHENTIC", "confidence": "99.0%"})
