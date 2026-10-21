"""
Single-Image Prediction & Forensic Inference Function.

Performs:
1. Safe RGB conversion and ImageNet normalization
2. Forward pass through fine-tuned EfficientNet-B0
3. Temperature-scaled calibrated probability calculation
4. Optional Grad-CAM explainability heatmap overlay generation
5. Returns structured JSON-compatible forensic dictionary
"""

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from torchvision.transforms import InterpolationMode

from .gradcam import EfficientNetGradCAM
from .model import EfficientNetDeepfakeClassifier
from .utils import get_device


def get_inference_transform():
    """Standard inference preprocessing matching evaluation."""
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def predict_image(
    image_path: Union[str, Path],
    model_path: Union[str, Path] = "models/best_model.pth",
    temperature: Optional[float] = None,
    generate_gradcam: bool = True,
    gradcam_output_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Predict whether a single image is REAL or FAKE with calibrated confidence.

    Args:
        image_path: Path to target image file
        model_path: Path to trained checkpoint (.pth)
        temperature: Optional temperature override (defaults to checkpoint temperature)
        generate_gradcam: Whether to produce visual saliency overlay
        gradcam_output_path: Destination path for Grad-CAM overlay image
        device: Compute device (auto-detected if None)

    Returns:
        Structured prediction dictionary with label, calibrated probabilities, and heatmap
    """
    img_p = Path(image_path)
    if not img_p.exists():
        raise FileNotFoundError(f"Input image not found at: {img_p}")

    if device is None:
        device = get_device()

    # 1. Load Model Checkpoint
    model = EfficientNetDeepfakeClassifier(pretrained=False)
    meta = model.load_checkpoint(model_path, map_location=str(device))
    model.to(device)
    model.eval()

    # Determine temperature
    t_val = temperature if temperature is not None else meta.get("temperature", 1.0)
    fake_idx = getattr(model, "fake_idx", 1)
    real_idx = getattr(model, "real_idx", 0)

    # 2. Image Loading & Preprocessing
    with Image.open(img_p) as pil_raw:
        pil_img = ImageOps.exif_transpose(pil_raw).convert("RGB")
        w_orig, h_orig = pil_img.size

    transform = get_inference_transform()
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    # 3. Model Inference & Calibrated Softmax
    with torch.no_grad():
        raw_logits = model(input_tensor)
        calibrated_logits = raw_logits / max(1e-4, t_val)
        probs = torch.softmax(calibrated_logits, dim=1).squeeze(0).cpu().numpy()

    p_fake = float(probs[fake_idx])
    p_real = float(probs[real_idx])
    predicted_label = "FAKE" if p_fake >= 0.50 else "REAL"
    confidence = p_fake if predicted_label == "FAKE" else p_real

    # 4. Optional Grad-CAM Saliency Overlay
    cam_path_str = None
    peak_bbox = None
    if generate_gradcam:
        try:
            cam_engine = EfficientNetGradCAM(model)
            cam_norm, peak_bbox = cam_engine.generate_cam(input_tensor, target_class_idx=fake_idx)

            out_cam = Path(gradcam_output_path) if gradcam_output_path else img_p.parent / f"{img_p.stem}_gradcam.png"
            orig_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            cam_engine.generate_overlay(orig_bgr, cam_norm, output_path=out_cam)
            cam_path_str = str(out_cam.resolve())
            cam_engine.remove_hooks()
        except Exception as e:
            print(f"[Grad-CAM Warning] Could not generate overlay: {e}")

    result = {
        "file": img_p.name,
        "path": str(img_p.resolve()),
        "original_resolution": f"{w_orig} x {h_orig} px",
        "prediction": predicted_label,
        "confidence": round(confidence, 4),
        "confidence_percentage": f"{confidence * 100:.2f}%",
        "fake_probability": round(p_fake, 4),
        "real_probability": round(p_real, 4),
        "is_calibrated": True,
        "temperature_used": round(t_val, 4),
        "peak_salient_bbox": peak_bbox,
        "gradcam_overlay_path": cam_path_str,
    }

    # Format human-readable output
    print("\n" + "=" * 50)
    print("        DEEPFAKE PREDICTION SUMMARY")
    print("=" * 50)
    print(f"File:             {result['file']}")
    print(f"Prediction:       {result['prediction']}")
    print(f"Confidence:       {result['confidence_percentage']}")
    print(f"Fake Probability: {result['fake_probability']:.4f}")
    print(f"Real Probability: {result['real_probability']:.4f}")
    if cam_path_str:
        print(f"Grad-CAM Overlay: {cam_path_str}")
    print("=" * 50 + "\n")

    return result


def main():
    parser = argparse.ArgumentParser(description="Predict REAL vs FAKE using fine-tuned EfficientNet-B0")
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("--model", type=str, default="models/best_model.pth", help="Path to checkpoint .pth file")
    parser.add_argument("--temperature", type=float, default=None, help="Temperature scaling override")
    parser.add_argument("--no-gradcam", action="store_true", help="Disable Grad-CAM heatmap generation")
    parser.add_argument("--out-cam", type=str, default=None, help="Output path for Grad-CAM overlay")
    args = parser.parse_args()

    predict_image(
        image_path=args.image_path,
        model_path=args.model,
        temperature=args.temperature,
        generate_gradcam=not args.no_gradcam,
        gradcam_output_path=args.out_cam,
    )


if __name__ == "__main__":
    main()
