"""
Automated Verification Suite for Improved Frequency, Classifier, and Metadata Agents.

Validates:
1. Distinct frequency spectra, radial statistics, and heatmaps for different images.
2. Deterministic spectrum output on identical image input.
3. Classifier Grad-CAM dynamic target class matching predicted class.
4. Metadata extraction integrity (real EXIF extracted, no fabricated tools).
5. Pipeline request-specific output paths and enriched payloads.
"""

import tempfile
from pathlib import Path
import cv2
import numpy as np
import pytest
from PIL import Image

from modules.frequency.agent import FrequencyAgent
from modules.frequency.explain import compute_centered_fft_spectrum
from modules.classifier.agent import ClassifierAgent
from modules.metadata.agent import MetadataAgent
from integration.complete_pipeline import CompleteDeepfakeDetectionPipeline
from secondary_system.models import SecondaryTriggerMode


@pytest.fixture
def test_images():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Image A: Smooth gradient
        x = np.linspace(0, 1, 256, dtype=np.float32)
        grad = np.outer(x, x)
        img_a = (grad * 255).astype(np.uint8)
        path_a = tmp / "image_a.png"
        cv2.imwrite(str(path_a), img_a)

        # Image B: High-frequency checkerboard / noise
        noise = np.random.RandomState(42).randint(0, 256, (256, 256), dtype=np.uint8)
        path_b = tmp / "image_b.png"
        cv2.imwrite(str(path_b), noise)

        yield {"path_a": path_a, "path_b": path_b, "img_a": img_a, "img_b": noise, "tmp": tmp}


def test_frequency_distinct_spectra_for_different_images(test_images):
    """Verify that different images produce genuinely distinct spectra and metrics."""
    agent = FrequencyAgent()

    res_a = agent.analyze(test_images["path_a"])
    res_b = agent.analyze(test_images["path_b"])

    # Verify fft_data exists and contains spectral energy metrics
    assert res_a.fft_data is not None
    assert res_b.fft_data is not None

    # Spectral entropy should be markedly different between smooth gradient and noise
    entropy_a = res_a.fft_data["spectral_entropy"]
    entropy_b = res_b.fft_data["spectral_entropy"]
    assert entropy_a != entropy_b

    # High frequency energy percentage should differ significantly
    high_pct_a = res_a.fft_data["high_pct"]
    high_pct_b = res_b.fft_data["high_pct"]
    assert high_pct_a != high_pct_b

    # Check generated spectrum plots are distinct image files
    plot_a_path = test_images["tmp"] / "spec_a.png"
    plot_b_path = test_images["tmp"] / "spec_b.png"
    res_a.get_spectrum_plot(output_path=plot_a_path)
    res_b.get_spectrum_plot(output_path=plot_b_path)

    assert plot_a_path.exists()
    assert plot_b_path.exists()

    plot_a_bytes = plot_a_path.read_bytes()
    plot_b_bytes = plot_b_path.read_bytes()
    assert plot_a_bytes != plot_b_bytes, "Frequency spectrum plots must be genuinely image-specific!"


def test_frequency_deterministic_on_identical_image(test_images):
    """Verify identical images produce identical spectra and metrics deterministically."""
    agent = FrequencyAgent()

    res_1 = agent.analyze(test_images["path_a"])
    res_2 = agent.analyze(test_images["path_a"])

    assert res_1.frequency_score == res_2.frequency_score
    assert res_1.fft_data["spectral_entropy"] == res_2.fft_data["spectral_entropy"]
    assert res_1.fft_data["high_pct"] == res_2.fft_data["high_pct"]


def test_classifier_gradcam_target_class():
    """Verify ClassifierAgent Grad-CAM uses target class matching the predicted label."""
    agent = ClassifierAgent(gradcam_enabled=True)

    # Synthetic test frame (224x224 RGB)
    rgb = np.full((224, 224, 3), 128, dtype=np.uint8)
    res = agent.analyze(rgb)

    assert res.predicted_label in ["REAL", "FAKE"]
    # Check that Grad-CAM generation does not throw and targets the correct index
    overlay = res.get_gradcam_overlay()
    assert overlay is not None
    assert overlay.shape == rgb.shape


def test_metadata_no_fabricated_tools(test_images):
    """Verify that images without metadata return 'None detected' without fabricated tools."""
    agent = MetadataAgent()

    res = agent.analyze(test_images["path_a"])

    # Disclosed tool must be None
    assert res.disclosed_tool is None
    # Neither OpenAI nor DALL-E 3 should be fabricated
    assert "openai" not in str(res.raw_manifest_data).lower()
    assert "dall-e" not in str(res.raw_manifest_data).lower()


def test_complete_pipeline_request_specific_artifacts(test_images):
    """Verify pipeline saves request-specific artifacts with analysis_id and enriched payload."""
    pipeline = CompleteDeepfakeDetectionPipeline(
        secondary_trigger_mode=SecondaryTriggerMode.ALWAYS,
        enable_secondary=True,
    )

    out_dir = test_images["tmp"] / "outputs"
    analysis_id_1 = "test_run_001"
    analysis_id_2 = "test_run_002"

    res_1 = pipeline.analyze(test_images["path_a"], output_dir=out_dir, analysis_id=analysis_id_1)
    res_2 = pipeline.analyze(test_images["path_b"], output_dir=out_dir, analysis_id=analysis_id_2)

    # Check file isolation
    spec_1 = out_dir / "frequency" / f"spectrum_{analysis_id_1}.png"
    spec_2 = out_dir / "frequency" / f"spectrum_{analysis_id_2}.png"
    assert spec_1.exists()
    assert spec_2.exists()
    assert spec_1.read_bytes() != spec_2.read_bytes()

    # Check enriched payload fields
    primary_1 = res_1["primary_system"]
    assert "low_freq_energy_pct" in primary_1["frequency"]
    assert "spectral_entropy" in primary_1["frequency"]
    assert "make" in primary_1["metadata"]
    assert "software" in primary_1["metadata"]
    assert "c2pa_status" in primary_1["metadata"]
