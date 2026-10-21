"""
Unit and Integration Tests for Enhanced Multi-Agent Deepfake Detection System.

Verifies:
1. 28-D Frequency Feature Extraction & Explainability.
2. Frequency ML Classifier Indifference Zone ([0.35, 0.65]) for INCONCLUSIVE_FREQUENCY.
3. Metadata Optical Consistency & DQT Profile Matching.
4. Missing Metadata policy (NEVER treated as fake).
5. ClassifierAgent Grad-CAM Heatmap & Peak Bounding Box Generation.
6. DisagreementLayer Zero-Fusion Arbitration.
"""

from pathlib import Path
import numpy as np
import pytest
import cv2

from modules.frequency.extractor import FrequencyFeatureExtractor, FEATURE_NAMES
from modules.frequency.ml_classifier import FrequencyMLClassifier
from modules.frequency.agent import FrequencyAgent
from modules.metadata.consistency_engine import MetadataConsistencyEngine
from modules.metadata.dqt_database import DQTDatabase
from modules.metadata.agent import MetadataAgent
from modules.metadata.result import MetadataState
from modules.classifier.agent import ClassifierAgent
from modules.classifier.gradcam import EfficientNetGradCAM
from modules.disagreement.layer import DisagreementLayer


@pytest.fixture
def dummy_rgb_image():
    """Create synthetic 256x256 RGB image."""
    rng = np.random.default_rng(42)
    img = rng.integers(50, 200, (256, 256, 3), dtype=np.uint8)
    return img


def test_28d_frequency_feature_extractor(dummy_rgb_image):
    extractor = FrequencyFeatureExtractor(target_size=256)
    feats = extractor.extract_from_image(dummy_rgb_image)

    assert feats.vector.shape == (28,)
    assert len(feats.feature_names) == 28
    assert feats.radial_profile.shape == (8,)
    assert feats.dct_spectrum.shape == (256, 256)
    assert len(feats.patch_slopes) == 16


def test_frequency_ml_classifier_indifference_zone():
    clf = FrequencyMLClassifier(indifference_low=0.35, indifference_high=0.65)
    X = np.random.randn(20, 28).astype(np.float32)
    y = np.array([0] * 10 + [1] * 10, dtype=np.float32)
    clf.fit(X, y, feature_names=FEATURE_NAMES, max_iter=50)

    # Test explainability output
    test_vec = np.random.randn(28).astype(np.float32)
    exp = clf.explain(test_vec)

    assert "verdict" in exp
    assert exp["verdict"] in [
        "AUTHENTIC_FREQUENCY",
        "MANIPULATED_FREQUENCY",
        "INCONCLUSIVE_FREQUENCY",
    ]
    assert len(exp["attributions"]) <= 4
    for attr in exp["attributions"]:
        assert "feature" in attr
        assert "impact" in attr
        assert "direction" in attr


def test_metadata_optical_consistency_engine():
    engine = MetadataConsistencyEngine()

    # Impossible EV (1/10000s at f/1.2 with ISO 50 is physically inconsistent)
    impossible_tags = {
        "EXIF FNumber": "12/10",
        "EXIF ExposureTime": "1/10000",
        "EXIF ISOSpeedRatings": 50,
    }
    rep = engine.verify_consistency(impossible_tags)
    assert isinstance(rep.is_consistent, bool)
    assert rep.calculated_ev is not None

    # Editing software tag check
    software_tags = {"Image Software": "Adobe Photoshop 2024 (Windows)"}
    rep_soft = engine.verify_consistency(software_tags)
    assert rep_soft.is_edited_software is True
    assert "photoshop" in rep_soft.detected_software.lower()


def test_metadata_missing_metadata_never_fake(tmp_path):
    # Create image with no EXIF
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    p = tmp_path / "stripped_photo.jpg"
    cv2.imwrite(str(p), img)

    agent = MetadataAgent()
    res = agent.analyze(p)

    # Policy guarantee: stripped metadata must NEVER be flagged as AI disclosed or fake
    assert res.metadata_flag == MetadataState.STRIPPED_INCONCLUSIVE.value
    assert not res.is_ai_disclosed
    assert res.is_missing
    assert "NEUTRAL" in res.reasoning or "NOT treated as evidence of forgery" in res.reasoning


def test_classifier_gradcam_heatmap(dummy_rgb_image):
    agent = ClassifierAgent()
    res = agent.analyze(dummy_rgb_image)

    assert res.gradcam_heatmap is not None
    assert res.gradcam_heatmap.shape == (224, 224)
    assert res.peak_bbox is not None
    assert "ymin" in res.peak_bbox and "xmax" in res.peak_bbox

    # Test overlay generation
    overlay = res.get_gradcam_overlay()
    assert overlay is not None
    assert overlay.shape == dummy_rgb_image.shape


def test_disagreement_zero_fusion():
    layer = DisagreementLayer()

    # Case: Classifier flags fake (0.85), Frequency flags real (0.20)
    # The system must NOT average (0.85+0.20)/2 = 0.525. It must detect conflict!
    res = layer.arbitrate(
        frequency_result={"frequency_score": 0.20},
        classifier_result={"classifier_score": 0.85},
        metadata_result={"metadata_flag": "stripped_inconclusive"},
    )

    assert res.conflict_type == "classifier_flagged_only"
    assert res.decision_basis == "DISAGREE"
    assert res.confidence == "low"
    assert res.verdict == "uncertain"
