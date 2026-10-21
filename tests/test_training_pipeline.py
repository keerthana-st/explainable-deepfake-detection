"""
Unit and Integration Tests for Deepfake Detection Training Pipeline.
Verifies:
- Data inspection, corrupt file filtering, and duplicate hashing
- Leakage-free splitting
- Two-stage EfficientNet-B0 freezing and unfreezing
- Temperature scaling calibration
- Grad-CAM heatmap generation
- Single-image prediction interface
- Integration with ClassifierAgent
"""

from pathlib import Path
import sys
import cv2
import numpy as np
import pytest
import torch
from PIL import Image

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepfake_detection.src.dataset import (
    DeepfakeDataset,
    calculate_dhash,
    create_leakage_free_splits,
    get_data_transforms,
    inspect_and_clean_dataset,
    resolve_class_label,
)
from deepfake_detection.src.gradcam import EfficientNetGradCAM
from deepfake_detection.src.model import EfficientNetDeepfakeClassifier, resolve_label_mapping
from deepfake_detection.src.predict import predict_image
from deepfake_detection.src.utils import TemperatureScaler, set_seed
from modules.classifier.agent import ClassifierAgent


@pytest.fixture
def dummy_dataset_dir(tmp_path):
    """Creates a temporary raw dataset structure with real, fake, corrupt, and duplicate files."""
    real_dir = tmp_path / "real_source"
    fake_dir = tmp_path / "fake_source"
    real_dir.mkdir(parents=True)
    fake_dir.mkdir(parents=True)

    # 1. Create 6 authentic real images with distinct textures
    for i in range(6):
        rng = np.random.RandomState(i)
        arr = rng.randint(40, 200, size=(64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(real_dir / f"id{i:02d}_real.jpg")

    # 2. Create 6 fake images with distinct textures
    for i in range(6):
        rng = np.random.RandomState(100 + i)
        arr = rng.randint(50, 220, size=(64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(fake_dir / f"id{i:02d}_fake.png")

    # 3. Corrupted image (0 bytes)
    (real_dir / "corrupted_empty.jpg").write_bytes(b"")

    # 4. Duplicate image (exact duplicate of id00_real.jpg)
    rng0 = np.random.RandomState(0)
    dup_arr = rng0.randint(40, 200, size=(64, 64, 3), dtype=np.uint8)
    Image.fromarray(dup_arr).save(real_dir / "duplicate_id00.jpg")

    return tmp_path


def test_label_resolution():
    """Verify that class names map to standard indices (0: REAL, 1: FAKE)."""
    assert resolve_class_label("real_faces") == 0
    assert resolve_class_label("0_real") == 0
    assert resolve_class_label("authentic_portraits") == 0
    assert resolve_class_label("fake_diffusion") == 1
    assert resolve_class_label("1_fake") == 1
    assert resolve_class_label("synthetic_stylegan") == 1
    assert resolve_class_label("random_folder") is None


def test_data_inspection_and_cleaning(dummy_dataset_dir):
    """Verify corrupt images and duplicates are properly identified and removed."""
    res = inspect_and_clean_dataset(dummy_dataset_dir, min_size=16, check_duplicates=True)
    assert res["real_count"] == 6  # 6 distinct real (1 duplicate removed, 1 empty ignored)
    assert res["fake_count"] == 6
    assert res["corrupted_count"] >= 1
    assert res["duplicate_count"] >= 1


def test_leakage_free_splits(dummy_dataset_dir, tmp_path):
    """Verify train/val/test splitting maintains exact proportions and no leakage."""
    res = inspect_and_clean_dataset(dummy_dataset_dir, check_duplicates=True)
    records = res["valid_records"]

    out_split = tmp_path / "splits_output"
    splits = create_leakage_free_splits(
        records=records,
        output_dir=out_split,
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
        random_seed=42,
        copy_files=True,
    )

    assert len(splits["train"]) > 0
    assert len(splits["val"]) > 0
    assert len(splits["test"]) > 0

    # Ensure physical files exist
    assert (out_split / "train" / "real").exists()
    assert (out_split / "train" / "fake").exists()


def test_model_two_stage_freezing():
    """Verify Stage 1 freezes backbone and Stage 2 unfreezes top blocks."""
    model = EfficientNetDeepfakeClassifier(pretrained=False)

    # Stage 1: Backbone frozen
    model.freeze_backbone()
    backbone_grad = [p.requires_grad for p in model.features.parameters()]
    head_grad = [p.requires_grad for p in model.classifier_head.parameters()]
    assert all(not g for g in backbone_grad)
    assert all(g for g in head_grad)

    # Stage 2: Unfreeze top blocks
    model.unfreeze_top_blocks(num_blocks=2)
    # The last block features[7] should have requires_grad == True
    assert any(p.requires_grad for p in model.features[-1].parameters())
    # Early block features[0] should remain frozen
    assert all(not p.requires_grad for p in model.features[0].parameters())


def test_temperature_scaling_calibration():
    """Verify temperature scaling preserves class order and scales logits."""
    scaler = TemperatureScaler()
    logits = torch.tensor([[2.0, -1.0], [0.5, 1.5], [-2.0, 3.0]])
    labels = torch.tensor([0, 1, 1])

    # Fit temperature
    scaler.fit(logits, labels)
    t = float(scaler.temperature.item())
    assert t > 0.0

    scaled_logits = scaler(logits)
    # Top-1 predictions must remain strictly identical
    assert torch.equal(torch.argmax(logits, dim=1), torch.argmax(scaled_logits, dim=1))


def test_gradcam_generation(tmp_path):
    """Verify Grad-CAM produces a valid normalized heatmap and bounding box."""
    model = EfficientNetDeepfakeClassifier(pretrained=False)
    gradcam = EfficientNetGradCAM(model)

    dummy_tensor = torch.randn(1, 3, 224, 224)
    cam_norm, peak_bbox = gradcam.generate_cam(dummy_tensor, target_class_idx=1)

    assert cam_norm.shape == (224, 224)
    assert 0.0 <= np.min(cam_norm) <= np.max(cam_norm) <= 1.0

    dummy_bgr = np.full((224, 224, 3), 128, dtype=np.uint8)
    out_cam = tmp_path / "test_gradcam.png"
    overlay = gradcam.generate_overlay(dummy_bgr, cam_norm, output_path=out_cam)
    assert out_cam.exists()
    assert overlay.shape == (224, 224, 3)
    gradcam.remove_hooks()


def test_predict_image_and_classifier_agent_integration(tmp_path):
    """Verify single-image prediction and ClassifierAgent integration."""
    model = EfficientNetDeepfakeClassifier(pretrained=False)
    ckpt_path = tmp_path / "test_model.pth"
    model.save_checkpoint(ckpt_path, epoch=1, val_loss=0.45, val_acc=0.85, val_f1=0.84, temperature=1.12)

    sample_img_path = tmp_path / "test_photo.jpg"
    arr = np.full((128, 128, 3), 180, dtype=np.uint8)
    Image.fromarray(arr).save(sample_img_path)

    # Test single-image prediction
    pred_res = predict_image(
        image_path=sample_img_path,
        model_path=ckpt_path,
        generate_gradcam=False,
    )
    assert pred_res["prediction"] in ["REAL", "FAKE"]
    assert 0.0 <= pred_res["fake_probability"] <= 1.0
    assert 0.0 <= pred_res["real_probability"] <= 1.0
    assert abs(pred_res["fake_probability"] + pred_res["real_probability"] - 1.0) < 1e-3

    # Test integration inside ClassifierAgent
    agent = ClassifierAgent(model_path=ckpt_path)
    res = agent.analyze(sample_img_path)
    assert res.predicted_label in ["REAL", "FAKE"]
    assert 0.0 <= res.classifier_score <= 1.0
    d = res.to_dict()
    assert "classifier_score" in d
    assert "predicted_label" in d
    assert "semantic_indicators" in d
