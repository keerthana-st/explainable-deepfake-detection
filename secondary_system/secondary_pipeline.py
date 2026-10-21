"""
Secondary System Pipeline Orchestrator.
Coordinates trigger evaluation, region segmentation, localized agent analysis,
evidence combination, global-regional consistency analysis, and explainability visualizations.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

from modules.classifier.agent import ClassifierAgent
from modules.frequency.agent import FrequencyAgent

from .consistency_analyzer import RegionalConsistencyAnalyzer
from .evidence_combiner import EvidenceCombiner
from .gradcam_regional import RegionalGradCAMRunner
from .models import (
    ConsistencyStatus,
    RegionEvidence,
    RegionalConsistencyResult,
    SecondaryAnalysisResult,
    SecondaryTriggerMode,
)
from .region_analyzer import RegionAnalyzer
from .region_preprocessor import RegionPreprocessor
from .region_segmenter import UnifiedRegionSegmenter
from .visualization import draw_region_overlay, generate_spatial_suspicion_heatmap


class SecondaryPipeline:
    """
    Forensic regional analysis extension to the Primary Deepfake Detection System.
    """

    def __init__(
        self,
        classifier_agent: Optional[ClassifierAgent] = None,
        frequency_agent: Optional[FrequencyAgent] = None,
        trigger_mode: SecondaryTriggerMode = SecondaryTriggerMode.ALWAYS,
        classifier_weight: float = 0.60,
        frequency_weight: float = 0.40,
        fake_threshold: float = 0.70,
        real_threshold: float = 0.30,
        enable_gradcam: bool = True,
    ):
        self.trigger_mode = trigger_mode
        self.enable_gradcam = enable_gradcam

        # Agents (reused instances)
        self.classifier_agent = classifier_agent or ClassifierAgent()
        self.frequency_agent = frequency_agent or FrequencyAgent()

        # Sub-modules
        self.segmenter = UnifiedRegionSegmenter()
        self.preprocessor = RegionPreprocessor()
        self.combiner = EvidenceCombiner(
            classifier_weight=classifier_weight,
            frequency_weight=frequency_weight,
            fake_threshold=fake_threshold,
            real_threshold=real_threshold,
        )
        self.analyzer = RegionAnalyzer(
            classifier_agent=self.classifier_agent,
            frequency_agent=self.frequency_agent,
            evidence_combiner=self.combiner,
            preprocessor=self.preprocessor,
        )
        self.consistency_analyzer = RegionalConsistencyAnalyzer()
        self.gradcam_runner = RegionalGradCAMRunner(
            classifier_agent=self.classifier_agent,
            preprocessor=self.preprocessor,
        )

    def should_trigger(self, primary_result: Dict[str, Any]) -> bool:
        """
        Evaluates whether the Secondary System should run based on configured trigger policy.
        """
        if self.trigger_mode == SecondaryTriggerMode.ALWAYS:
            return True

        global_verdict = str(primary_result.get("verdict", "")).upper()
        confidence = str(primary_result.get("confidence", "")).upper()
        decision_basis = str(primary_result.get("decision_basis", "")).upper()

        if self.trigger_mode == SecondaryTriggerMode.LOW_CONFIDENCE:
            return "LOW" in confidence or "UNCERTAIN" in global_verdict

        if self.trigger_mode == SecondaryTriggerMode.DISAGREEMENT:
            badge = str(primary_result.get("badge", "")).upper()
            return any(k in decision_basis for k in ["DISAGREE", "CONFLICT"]) or any(k in badge for k in ["DISCORDANT", "CONFLICT", "DISAGREE"])


        if self.trigger_mode == SecondaryTriggerMode.SUSPICIOUS_ONLY:
            return any(k in global_verdict for k in ["FAKE", "AI-EDITED", "SUSPICIOUS", "MANIPULATED"])

        if self.trigger_mode == SecondaryTriggerMode.MANUAL:
            return bool(primary_result.get("trigger_secondary_manual", False))

        return True

    def analyze(
        self,
        np_img: np.ndarray,
        primary_global_verdict: str,
        primary_result_dict: Optional[Dict[str, Any]] = None,
        output_dir: Union[str, Path] = "outputs/secondary",
    ) -> SecondaryAnalysisResult:
        """
        Execute regional forensic analysis for an image.

        Args:
            np_img: Full RGB image array of shape (H, W, 3)
            primary_global_verdict: Upstream primary verdict (e.g. 'REAL', 'FAKE', 'AI-EDITED')
            primary_result_dict: Upstream primary analysis dictionary
            output_dir: Directory where visualizations will be saved

        Returns:
            SecondaryAnalysisResult containing localized evidence and consistency metrics
        """
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)

        # 1. Evaluate trigger policy
        p_dict = primary_result_dict or {"verdict": primary_global_verdict}
        fired = self.should_trigger(p_dict)

        if not fired:
            return SecondaryAnalysisResult(
                enabled=False,
                trigger_mode=self.trigger_mode.value,
                trigger_fired=False,
                summary_explanation=f"Secondary regional analysis bypassed (Trigger policy: {self.trigger_mode.value}).",
            )

        # 2. Segment Semantic Regions
        regions = self.segmenter.segment_image(np_img)

        # 3. Analyze each Region using existing agents
        region_evidences = self.analyzer.analyze_regions(regions, np_img)

        # 4. Global vs Regional Consistency Analysis
        consistency_res = self.consistency_analyzer.analyze_consistency(
            primary_global_verdict=primary_global_verdict,
            region_evidences=region_evidences,
        )

        # 5. Identify Suspicious Regions
        suspicious_regions = [r.region_id for r in region_evidences if r.region_verdict == "AI-EDITED"]

        # 6. Generate Visualizations
        overlay_path = str((out_p / "region_overlay.png").resolve())
        heatmap_path = str((out_p / "spatial_heatmap.png").resolve())

        draw_region_overlay(np_img, region_evidences, save_path=overlay_path)
        generate_spatial_suspicion_heatmap(np_img, region_evidences, save_path=heatmap_path)

        visuals: Dict[str, str] = {
            "region_overlay": overlay_path,
            "spatial_heatmap": heatmap_path,
        }

        # Generate individual region crops and heatmaps
        import cv2
        for r in region_evidences:
            if r.bbox and len(r.bbox) == 4:
                ymin, xmin, ymax, xmax = r.bbox
                ymin, xmin = max(0, int(ymin)), max(0, int(xmin))
                ymax, xmax = min(np_img.shape[0], int(ymax)), min(np_img.shape[1], int(xmax))
                if ymax > ymin and xmax > xmin:
                    crop = np_img[ymin:ymax, xmin:xmax]
                    crop_path = str((out_p / f"crop_{r.region_id}.png").resolve())
                    cv2.imwrite(crop_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                    visuals[f"crop_{r.region_id}"] = crop_path
                    r.crop_path = crop_path

                    try:
                        import matplotlib.pyplot as plt
                        jet = plt.get_cmap('jet')
                        rh, rw = np_img.shape[:2]
                        ry, rx = np.ogrid[:rh, :rw]
                        rcy, rcx = (ymin + ymax) / 2.0, (xmin + xmax) / 2.0
                        r_sigma_y = max(1.0, (ymax - ymin) / 2.5) ** 2
                        r_sigma_x = max(1.0, (xmax - xmin) / 2.5) ** 2
                        dist = ((rx - rcx)**2 / (2 * r_sigma_x)) + ((ry - rcy)**2 / (2 * r_sigma_y))
                        r_hm = np.exp(-dist)
                        r_hm = (r_hm - r_hm.min()) / (r_hm.max() - r_hm.min() + 1e-8)
                        c_hm = (jet(r_hm)[:, :, :3] * 255).astype(np.uint8)
                        blended = (np_img.astype(np.float32) * 0.45 + c_hm.astype(np.float32) * 0.55).astype(np.uint8)
                        r_hm_path = str((out_p / f"heatmap_{r.region_id}.png").resolve())
                        cv2.imwrite(r_hm_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
                        visuals[f"heatmap_{r.region_id}"] = r_hm_path
                        r.heatmap_path = r_hm_path
                    except Exception:
                        pass

        # 7. Selective Regional Grad-CAM
        if self.enable_gradcam and (len(suspicious_regions) > 0 or "UNCERTAIN" in primary_global_verdict.upper()):
            cam_dict = self.gradcam_runner.run_selective_gradcam(
                np_img=np_img,
                region_evidences=region_evidences,
                output_dir=out_p / "gradcam",
            )
            visuals.update(cam_dict)

        # 8. Synthesize Human-Readable Summary
        if suspicious_regions:
            summary = (
                f"Regional analysis isolated manipulation evidence in: [{', '.join(suspicious_regions)}]. "
                f"Other regions tested authentic/neutral. Consistency status: {consistency_res.status.value}."
            )
        else:
            summary = (
                f"All analyzed regions [{', '.join(r.region_id for r in region_evidences)}] "
                f"exhibit authentic photographic characteristics. Consistency status: {consistency_res.status.value}."
            )

        return SecondaryAnalysisResult(
            enabled=True,
            trigger_mode=self.trigger_mode.value,
            trigger_fired=True,
            regions=region_evidences,
            regional_consistency=consistency_res,
            suspicious_regions=suspicious_regions,
            visualizations=visuals,
            summary_explanation=summary,
        )
