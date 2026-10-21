"""
Unit and Regression Tests for Frequency Agent (Module 1).

Tests adhere strictly to specification requirements:
1. Score is always in [0, 1] regardless of input.
2. Injected periodic high-frequency pattern scores measurably higher than a smooth image.
3. Empty, corrupted, or too-small images raise clear errors (no silent failures).
4. Coefficient map dimensions match cropped input dimensions exactly.
5. Calibration metadata, explainability fields, and heatmaps meet project requirements.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest

from modules.frequency.agent import (
    DEFAULT_CALIBRATION_DATE,
    DEFAULT_CALIBRATION_DIVISOR,
    DEFAULT_DECISION_THRESHOLD,
    FrequencyAgent,
    FrequencyAnalysisResult,
)
from modules.frequency.visualization import generate_frequency_heatmap


@pytest.fixture
def agent():
    """Fixture to provide a default initialized FrequencyAgent."""
    return FrequencyAgent()


class TestFrequencyScoreBounds:
    """Test Requirement: Score is always in [0, 1] regardless of input."""

    def test_score_bounded_on_pure_zeros(self, agent):
        img = np.zeros((128, 128), dtype=np.uint8)
        res = agent.analyze(img)
        assert 0.0 <= res.frequency_score <= 1.0
        assert res.frequency_score == 0.0
        assert res.raw_score == 0.0

    def test_score_bounded_on_pure_ones(self, agent):
        img = np.full((128, 128), 255, dtype=np.uint8)
        res = agent.analyze(img)
        assert 0.0 <= res.frequency_score <= 1.0
        assert res.frequency_score == 0.0  # Constant DC only, zero AC/high-frequency energy

    def test_score_bounded_on_extreme_noise(self, agent):
        # Extreme white noise with maximum high-frequency energy
        img = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        res = agent.analyze(img)
        assert 0.0 <= res.frequency_score <= 1.0
        assert res.raw_score > 0.0

    def test_score_bounded_on_super_high_amplitude_signal(self, agent):
        # Checkerboard with maximum possible contrast
        cb = np.zeros((64, 64), dtype=np.uint8)
        cb[::2, ::2] = 255
        cb[1::2, 1::2] = 255
        res = agent.analyze(cb)
        assert 0.0 <= res.frequency_score <= 1.0
        assert res.frequency_score <= 1.0  # Clamped to 1.0


class TestCoreMechanismRegression:
    """
    Test Requirement: A synthetic image with injected periodic high-frequency pattern
    scores measurably higher than a smooth synthetic image.
    """

    def test_periodic_pattern_scores_higher_than_smooth(self, agent):
        size = 128
        x = np.linspace(0, 4 * np.pi, size)
        y = np.linspace(0, 4 * np.pi, size)
        xx, yy = np.meshgrid(x, y)

        # 1. Smooth synthetic natural-like image (low frequencies only)
        smooth_img = (np.sin(xx) * np.cos(yy) * 100.0 + 128.0).astype(np.uint8)

        # 2. Synthetic image with injected periodic high-frequency pattern (simulating GAN upsampling)
        periodic_img = smooth_img.copy().astype(np.float32)
        # High-frequency alternating checkerboard grid (falls directly in the high-frequency DCT quadrant)
        grid_r, grid_c = np.indices((size, size))
        checkerboard = ((-1.0) ** (grid_r + grid_c)) * 30.0
        periodic_img = np.clip(periodic_img + checkerboard, 0, 255).astype(np.uint8)

        res_smooth = agent.analyze(smooth_img)
        res_periodic = agent.analyze(periodic_img)

        # Verification: Periodic pattern must have significantly higher raw and frequency scores
        assert res_periodic.raw_score > res_smooth.raw_score * 3.0, (
            f"Periodic score ({res_periodic.raw_score}) should be significantly higher "
            f"than smooth score ({res_smooth.raw_score})"
        )
        assert res_periodic.frequency_score > res_smooth.frequency_score
        assert res_periodic.is_suspicious is True
        assert res_smooth.is_suspicious is False


class TestInputValidationAndErrors:
    """Test Requirement: Empty/corrupted/too-small images raise a clear error, not silent wrong result."""

    def test_none_input_raises_value_error(self, agent):
        with pytest.raises(ValueError, match="cannot be None"):
            agent.analyze(None)

    def test_empty_array_raises_value_error(self, agent):
        with pytest.raises(ValueError, match="empty"):
            agent.analyze(np.array([]))

    def test_too_small_image_raises_value_error(self, agent):
        # 7x7 is smaller than minimum 8x8 block size
        small_img = np.zeros((7, 7), dtype=np.uint8)
        with pytest.raises(ValueError, match="too small"):
            agent.analyze(small_img)

    def test_dimension_with_one_axis_too_small(self, agent):
        img = np.zeros((128, 6), dtype=np.uint8)
        with pytest.raises(ValueError, match="too small"):
            agent.analyze(img)

    def test_nonexistent_file_raises_value_error(self, agent):
        with pytest.raises(ValueError, match="does not exist"):
            agent.analyze("nonexistent_path_12345.png")


class TestDimensionIntegrityAndCropping:
    """Test Requirement: Coefficient map dimensions match cropped input dimensions exactly."""

    @pytest.mark.parametrize(
        "orig_h,orig_w,expected_h,expected_w",
        [
            (64, 64, 64, 64),
            (65, 65, 64, 64),
            (100, 107, 96, 104),
            (250, 311, 248, 304),
            (8, 8, 8, 8),
            (15, 15, 8, 8),
        ],
    )
    def test_coefficient_map_shape_matches_cropped_dimensions(
        self, agent, orig_h, orig_w, expected_h, expected_w
    ):
        img = np.random.randint(0, 256, (orig_h, orig_w, 3), dtype=np.uint8)
        res = agent.analyze(img)

        assert res.cropped_shape == (expected_h, expected_w)
        assert res.coefficient_map.shape == (expected_h, expected_w)

    def test_coefficient_map_block_structure(self, agent):
        # Create an image where exactly ONE 8x8 block has high-frequency noise
        img = np.zeros((32, 32), dtype=np.uint8)
        # Block (1, 1): rows 8..15, cols 8..15
        img[8:16:2, 8:16:2] = 255

        res = agent.analyze(img)
        # The block (1, 1) region should have positive energy
        active_block_energy = res.coefficient_map[8:16, 8:16]
        # Neighboring quiet blocks should have 0 energy
        quiet_block_energy = res.coefficient_map[0:8, 0:8]

        assert np.all(active_block_energy > 0)
        assert np.all(quiet_block_energy == 0)
        # Spatial energy assignment in the map is constant within each 8x8 block
        assert np.all(active_block_energy == active_block_energy[0, 0])


class TestMetadataAndExplainability:
    """Verify structured output fields, calibration metadata, and required disclosures."""

    def test_result_structure_and_metadata(self, agent):
        img = np.zeros((32, 32), dtype=np.uint8)
        res = agent.analyze(img)

        assert isinstance(res, FrequencyAnalysisResult)
        assert hasattr(res, "frequency_score")
        assert hasattr(res, "raw_score")
        assert hasattr(res, "coefficient_map")
        assert hasattr(res, "calibration_metadata")
        assert hasattr(res, "reasoning")
        assert hasattr(res, "false_positive_risks")
        assert hasattr(res, "blind_spot_warning")

        meta = res.calibration_metadata
        assert "divisor" in meta
        assert "threshold" in meta
        assert "calibration_dataset" in meta
        assert "calibration_date" in meta
        assert meta["divisor"] == DEFAULT_CALIBRATION_DIVISOR
        assert meta["threshold"] == DEFAULT_DECISION_THRESHOLD

    def test_false_positive_and_blind_spot_disclosures(self, agent):
        img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        res = agent.analyze(img)

        # Proactive false-positive risk factors must be present
        assert len(res.false_positive_risks) >= 4
        risk_text = " ".join(res.false_positive_risks).lower()
        assert "jpeg compression" in risk_text
        assert "texture" in risk_text
        assert "sharpening" in risk_text

        # Blind spot warning must explicitly name diffusion models
        assert "diffusion" in res.blind_spot_warning.lower()

    def test_heatmap_generation(self, agent):
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        res = agent.analyze(img)

        heatmap = res.get_heatmap(alpha=0.5)
        assert heatmap is not None
        assert heatmap.shape == (64, 64, 3)
        assert heatmap.dtype == np.uint8

    def test_command_prompt_output_clean_of_tabular_ml(self, agent):
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        res = agent.analyze(img)

        # 1. format_console_output structure check
        out = res.format_console_output()
        assert "MODULE 1: FREQUENCY AGENT" in out
        assert "• Classical Verdict:" in out
        assert "• Frequency Score:" in out
        assert "• Raw Energy Score:" in out
        assert "• Separation Threshold:" in out
        assert "• Cropped Resolution:" in out
        assert "Explainable Reasoning:" in out
        assert "NATURAL FREQUENCY PROFILE" in out

        # 2. Strict omission of all 28-D Tabular ML details from displayed output
        forbidden_terms = [
            "28-D Tabular ML",
            "28-D ML Verdict",
            "radial_power_band_0",
            "z-score",
            "ML probability",
            "Top Frequency Driver",
            "28-D ML Classifier Analysis",
        ]
        for term in forbidden_terms:
            assert term.lower() not in out.lower()
            assert term.lower() not in res.reasoning.lower()

        # 3. Internal fields and DCT calculations remain fully intact
        assert hasattr(res, "coefficient_map")
        assert hasattr(res, "frequency_score")
        assert hasattr(res, "raw_score")
        assert res.features_28d is not None
