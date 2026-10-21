"""
Selective Regional Grad-CAM for the Secondary System.
Generates localized feature attribution heatmaps exclusively for suspicious
or high-disagreement regions using the existing fine-tuned EfficientNet model.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import cv2
import numpy as np
import torch
from PIL import Image

from deepfake_detection.src.gradcam import EfficientNetGradCAM
from modules.classifier.agent import ClassifierAgent

from .models import RegionEvidence, RegionVerdict
from .region_preprocessor import RegionPreprocessor


class RegionalGradCAMRunner:
    """
    Selective Grad-CAM generator for high-suspicion regions.
    """

    def __init__(
        self,
        classifier_agent: ClassifierAgent,
        preprocessor: Optional[RegionPreprocessor] = None,
    ):
        self.classifier_agent = classifier_agent
        self.preprocessor = preprocessor or RegionPreprocessor()
        self.gradcam_engine = EfficientNetGradCAM(self.classifier_agent.model)

    def run_selective_gradcam(
        self,
        np_img: np.ndarray,
        region_evidences: List[RegionEvidence],
        output_dir: Union[str, Path] = "outputs/gradcam",
    ) -> Dict[str, str]:
        """
        Executes Grad-CAM attribution selectively on regions flagged as:
        - AI-EDITED
        - UNCERTAIN / High Disagreement

        Args:
            np_img: Full RGB image array
            region_evidences: List of evaluated RegionEvidence
            output_dir: Folder to save Grad-CAM overlay images

        Returns:
            Dictionary mapping region_id to saved gradcam overlay image path
        """
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        results: Dict[str, str] = {}

        h_full, w_full = np_img.shape[:2]

        for r in region_evidences:
            # Check selection criteria (Part 18: only suspicious or uncertain regions)
            is_suspicious = r.region_verdict == RegionVerdict.AI_EDITED.value
            is_uncertain = r.region_verdict == RegionVerdict.UNCERTAIN.value and not r.agent_agreement

            if not (is_suspicious or is_uncertain):
                continue

            ymin, xmin, ymax, xmax = r.bbox
            crop_rgb = np_img[ymin:ymax, xmin:xmax]
            if crop_rgb.shape[0] < 16 or crop_rgb.shape[1] < 16:
                continue

            try:
                # Preprocess crop to 224x224 tensor
                pil_crop = Image.fromarray(crop_rgb)
                tensor_crop = self.preprocessor.transform(pil_crop).unsqueeze(0).to(self.classifier_agent.device)

                # Generate CAM normalized to [0, 1]
                target_idx = getattr(self.classifier_agent, "fake_idx", 1)
                cam_norm, _ = self.gradcam_engine.generate_cam(tensor_crop, target_class_idx=target_idx)

                # Resize CAM to match crop dimensions
                cam_resized = cv2.resize(cam_norm, (crop_rgb.shape[1], crop_rgb.shape[0]))
                heatmap_color = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
                heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

                # Blend with crop
                crop_overlay = cv2.addWeighted(crop_rgb, 0.60, heatmap_rgb, 0.40, 0)

                # Embed blended crop back onto full image canvas
                full_canvas = np_img.copy()
                full_canvas[ymin:ymax, xmin:xmax] = crop_overlay

                # Draw bounding box outline around the highlighted region
                cv2.rectangle(full_canvas, (xmin, ymin), (xmax, ymax), (220, 50, 50), 2)
                cv2.putText(
                    full_canvas,
                    f"Grad-CAM: {r.region_id} ({r.fake_probability*100:.0f}%)",
                    (xmin + 4, max(ymin - 6, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (220, 50, 50),
                    1,
                    cv2.LINE_AA,
                )

                save_file = out_p / f"gradcam_{r.region_id}.png"
                Image.fromarray(full_canvas).save(save_file)

                saved_path_str = str(save_file.resolve())
                r.gradcam_path = saved_path_str
                results[r.region_id] = saved_path_str

            except Exception as e:
                print(f"[Grad-CAM Warning] Could not generate regional CAM for {r.region_id}: {e}")

        return results
