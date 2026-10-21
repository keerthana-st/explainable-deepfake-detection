"""
Integration Tests for the Complete Deepfake Detection Pipeline.

Verifies:
  1. Full end-to-end pipeline execution from Preprocessing through Primary and Secondary systems.
  2. JSON serialization invariance (schema validity).
  3. Graceful error handling for missing/corrupted files.
  4. Toggleable execution (Secondary System enabled vs disabled).
"""

import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from integration.complete_pipeline import CompleteDeepfakeDetectionPipeline
from secondary_system.models import SecondaryTriggerMode


@pytest.fixture(scope="module")
def complete_pipeline():
    """Instantiates pipeline once for integration test suite."""
    return CompleteDeepfakeDetectionPipeline(
        secondary_trigger_mode=SecondaryTriggerMode.ALWAYS,
        classifier_weight=0.60,
        frequency_weight=0.40,
        enable_secondary=True,
        enable_gradcam=True,
    )


@pytest.fixture
def synthetic_test_image(tmp_path):
    """Creates a temporary valid PNG image for pipeline testing."""
    img_path = tmp_path / "synthetic_portrait.png"
    arr = np.full((300, 300, 3), 180, dtype=np.uint8)
    # Add a mock person silhouette
    arr[60:260, 80:220] = [60, 110, 190]
    # Add a simulated head
    import cv2
    cv2.circle(arr, (150, 100), 30, (220, 190, 170), -1)
    Image.fromarray(arr).save(img_path)
    return str(img_path)


class TestCompletePipelineIntegration:
    def test_end_to_end_execution(self, complete_pipeline, synthetic_test_image, tmp_path):
        out_dir = tmp_path / "pipeline_outputs"
        result = complete_pipeline.analyze(synthetic_test_image, output_dir=out_dir)

        # 1. Status Check
        assert result["status"] == "success"
        assert "synthetic_portrait" in result["file"]


        # 2. Primary System Payload
        primary = result["primary_system"]
        assert "global_verdict" in primary
        assert "confidence" in primary
        assert "decision_basis" in primary
        assert "classifier" in primary
        assert "frequency" in primary
        assert "metadata" in primary
        assert "disagreement" in primary

        # Classifier & Frequency Agent payloads
        assert 0.0 <= primary["classifier"]["score"] <= 1.0
        assert primary["classifier"]["prediction"] in ["REAL", "FAKE"]
        assert 0.0 <= primary["frequency"]["score"] <= 1.0

        # 3. Secondary System Payload
        secondary = result["secondary_system"]
        assert secondary is not None
        assert secondary["enabled"] is True
        assert secondary["executed"] is True
        assert secondary["region_count"] >= 2
        assert len(secondary["regions"]) >= 2

        # Check region structure
        for r in secondary["regions"]:
            assert "region_id" in r
            assert "region_type" in r
            assert "combined_score" in r
            assert "verdict" in r
            assert "agreement" in r

        # Consistency analysis structure
        consistency = secondary["regional_consistency"]
        assert consistency is not None
        assert consistency["status"] in ["SUPPORTED", "PARTIALLY_SUPPORTED", "CONTRADICTED", "INCONCLUSIVE"]

        # 4. Visualizations Payload
        vis = result["visualizations"]
        assert "dct_spectrum" in vis
        assert "radial_decay" in vis
        assert "frequency_heatmap" in vis
        assert "region_overlay" in vis
        assert "spatial_heatmap" in vis

        # 5. Final Explainability Narrative
        explanation = result["final_explanation"]
        assert isinstance(explanation, str)
        assert "1. GLOBAL STATUS" in explanation
        assert "2. LOCALIZED ATTRIBUTION" in explanation
        assert "3. MULTI-AGENT EVIDENCE" in explanation

    def test_json_serializability(self, complete_pipeline, synthetic_test_image, tmp_path):
        """Ensures the entire result object can be cleanly serialized to JSON."""
        out_dir = tmp_path / "json_outputs"
        result = complete_pipeline.analyze(synthetic_test_image, output_dir=out_dir)

        # Must not raise TypeError
        json_output = json.dumps(result, indent=2, default=str)
        assert len(json_output) > 100
        parsed = json.loads(json_output)
        assert parsed["status"] == "success"

    def test_graceful_missing_file_handling(self, complete_pipeline):
        """Ensures non-existent files are handled gracefully without uncaught exceptions."""
        result = complete_pipeline.analyze("non_existent_file_path_12345.jpg")
        assert result["status"] == "error"
        assert "Preprocessing failed" in result["final_explanation"]

    def test_standalone_primary_when_secondary_disabled(self, synthetic_test_image, tmp_path):
        """Ensures primary system runs standalone when secondary is disabled."""
        pipeline_no_sec = CompleteDeepfakeDetectionPipeline(
            enable_secondary=False,
            enable_gradcam=False,
        )
        res = pipeline_no_sec.analyze(synthetic_test_image, output_dir=tmp_path / "no_sec")
        assert res["status"] == "success"
        assert res["secondary_system"] is None
        assert "Evaluated globally" in res["final_explanation"]
