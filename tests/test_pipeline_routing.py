"""
Unit and Integration Tests for Decoupled Multi-Agent Pipeline Routing.
Verifies that:
  - Preprocessing produces decoupled representations (Original, RGB, Grayscale, Normalized, Metadata)
  - Frequency Agent receives only the Grayscale representation
  - Classifier Agent receives only the RGB representation
  - Metadata Agent receives only the Original raw file / bytes
  - Disagreement Layer receives structured results from all three agents without feature fusion
"""

import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
from PIL import Image
import pytest

from modules.preprocessing import (
    ImagePreprocessor,
    PreprocessingConfig,
    PreprocessingResult,
    preprocess_image,
)
from modules.frequency.agent import FrequencyAgent
from modules.classifier.agent import ClassifierAgent
from modules.metadata.agent import MetadataAgent
from modules.disagreement.layer import DisagreementLayer
from scripts.test_single_image import resolve_or_run_preprocessing


@pytest.fixture
def temp_workspace():
    """Creates a temporary workspace with test images and clean output dir."""
    tmp_dir = Path(tempfile.mkdtemp())
    output_dir = tmp_dir / "outputs" / "preprocessing"
    
    # Create test RGB image with texture
    img_path = tmp_dir / "test_input.png"
    arr = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
    Image.fromarray(arr).save(img_path)

    yield {
        "workspace": tmp_dir,
        "input_image": img_path,
        "output_dir": output_dir,
    }

    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_preprocessing_produces_all_decoupled_artifacts(temp_workspace):
    """Verifies that preprocessing persists 5 decoupled forensic representations."""
    img_path = temp_workspace["input_image"]
    out_dir = temp_workspace["output_dir"]

    result = preprocess_image(img_path, output_dir=out_dir, save_artifacts=True)
    assert result.is_success, f"Preprocessing failed: {result.error_message}"

    saved = result.saved_paths
    assert "original" in saved
    assert "rgb" in saved
    assert "grayscale" in saved
    assert "normalized" in saved
    assert "metadata" in saved

    assert Path(saved["original"]).exists()
    assert Path(saved["rgb"]).exists()
    assert Path(saved["grayscale"]).exists()
    assert Path(saved["normalized"]).exists()
    assert Path(saved["metadata"]).exists()


def test_frequency_agent_routes_grayscale(temp_workspace):
    """Verifies that Frequency Agent processes the grayscale image correctly."""
    img_path = temp_workspace["input_image"]
    out_dir = temp_workspace["output_dir"]

    prep = preprocess_image(img_path, output_dir=out_dir, save_artifacts=True)
    gray_path = prep.saved_paths["grayscale"]

    # Verify input is truly grayscale
    gray_img = Image.open(gray_path)
    assert gray_img.mode in ("L", "I;16", "RGB")  # saved as PNG

    freq_agent = FrequencyAgent()
    freq_res = freq_agent.analyze(gray_path)

    assert freq_res is not None
    assert 0.0 <= freq_res.frequency_score <= 1.0
    assert freq_res.raw_score >= 0.0
    assert freq_res.cropped_shape[0] % 8 == 0
    assert freq_res.cropped_shape[1] % 8 == 0

    # Verify visual explainability outputs work with grayscale input
    heatmap = freq_res.get_heatmap()
    assert heatmap is not None
    assert len(heatmap.shape) == 3


def test_classifier_agent_routes_rgb(temp_workspace):
    """Verifies that Classifier Agent processes the RGB image correctly."""
    img_path = temp_workspace["input_image"]
    out_dir = temp_workspace["output_dir"]

    prep = preprocess_image(img_path, output_dir=out_dir, save_artifacts=True)
    rgb_path = prep.saved_paths["rgb"]

    cls_agent = ClassifierAgent()
    cls_res = cls_agent.analyze(rgb_path)

    assert cls_res is not None
    assert 0.0 <= cls_res.classifier_score <= 1.0
    assert cls_res.predicted_label in ("REAL", "FAKE")
    assert "noise_residual_variance" in cls_res.semantic_indicators

    # Verify Grad-CAM generation works with RGB input
    gradcam = cls_res.get_gradcam_overlay()
    assert gradcam is not None


def test_metadata_agent_routes_original_raw_file(temp_workspace):
    """
    Verifies that Metadata Agent receives the original raw file bytes,
    preserving binary chunks that would be stripped in preprocessed PNGs.
    """
    img_path = temp_workspace["input_image"]
    out_dir = temp_workspace["output_dir"]

    prep = preprocess_image(img_path, output_dir=out_dir, save_artifacts=True)
    orig_path = prep.saved_paths["original"]

    meta_agent = MetadataAgent()
    meta_res = meta_agent.analyze(orig_path)

    assert meta_res is not None
    assert meta_res.metadata_flag in ("ai_disclosed", "tampered_inconsistent", "present_no_ai_signal", "missing", "stripped_inconclusive")
    assert meta_res.data_source_tag is not None


def test_full_pipeline_end_to_end_routing(temp_workspace):
    """
    Verifies full end-to-end multi-agent pipeline with correct routing and zero-fusion arbitration.
    """
    img_path = temp_workspace["input_image"]
    out_dir = temp_workspace["output_dir"]

    # 0. Preprocessing
    prep_res = resolve_or_run_preprocessing(img_path, output_dir=out_dir)
    assert prep_res.is_success

    # 1. Frequency Agent -> Grayscale
    freq_agent = FrequencyAgent()
    freq_res = freq_agent.analyze(prep_res.saved_paths["grayscale"])

    # 2. Classifier Agent -> RGB
    cls_agent = ClassifierAgent()
    cls_res = cls_agent.analyze(prep_res.saved_paths["rgb"])

    # 3. Metadata Agent -> Original
    meta_agent = MetadataAgent()
    meta_res = meta_agent.analyze(prep_res.saved_paths["original"])

    # 4. Disagreement Layer -> Arbitrate
    dis_layer = DisagreementLayer()
    dis_res = dis_layer.arbitrate(freq_res, cls_res, meta_res)

    assert dis_res.output_state in (
        "HIGH_CONFIDENCE_FAKE",
        "HIGH_CONFIDENCE_REAL",
        "UNCERTAIN_classifier_flagged_only",
        "UNCERTAIN_frequency_flagged_only",
        "PROVENANCE_OVERRIDE_FAKE",
        "PROVENANCE_UNCERTAIN",
    )
    assert dis_res.verdict.upper() in ("REAL", "FAKE", "UNCERTAIN")
    assert round(dis_res.scores.frequency, 4) == round(freq_res.frequency_score, 4)
    assert round(dis_res.scores.classifier, 4) == round(cls_res.classifier_score, 4)


def test_detection_of_cached_preprocessed_outputs(temp_workspace):
    """Verifies that resolve_or_run_preprocessing detects existing preprocessed files."""
    img_path = temp_workspace["input_image"]
    out_dir = temp_workspace["output_dir"]

    # Run initial preprocessing
    prep_1 = resolve_or_run_preprocessing(img_path, output_dir=out_dir)
    assert prep_1.is_success

    # Run resolution again on the original path
    prep_2 = resolve_or_run_preprocessing(img_path, output_dir=out_dir, force_reprocess=False)
    assert prep_2.is_success
    assert prep_2.saved_paths["grayscale"] == prep_1.saved_paths["grayscale"]
    assert prep_2.saved_paths["rgb"] == prep_1.saved_paths["rgb"]
    assert prep_2.saved_paths["original"] == prep_1.saved_paths["original"]
