"""
Ablation Study Orchestrator for Explainable Deepfake Detection.

Compares 4 experimental configurations:
  Experiment 1: Primary System Only (No regional analysis, no Grad-CAM)
  Experiment 2: Primary + Secondary System (Classifier-only regional analysis: w_cls=1.0, w_freq=0.0)
  Experiment 3: Primary + Secondary + Frequency Fusion (Multi-modal regional analysis: w_cls=0.6, w_freq=0.4)
  Experiment 4: Full Multi-Agent System (Primary + Secondary + Frequency Fusion + Grad-CAM Attribution)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from integration.complete_pipeline import CompleteDeepfakeDetectionPipeline
from secondary_system.models import SecondaryTriggerMode



def run_ablation_experiment(
    image_path: str,
    output_dir: str = "outputs/ablation",
) -> Dict[str, Any]:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"RUNNING ABLATION STUDY ON: {Path(image_path).name}")
    print(f"Path: {image_path}")
    print("=" * 80)

    configs = [
        {
            "id": "exp1_primary_only",
            "name": "Exp 1: Primary System Only",
            "enable_secondary": False,
            "classifier_weight": 0.60,
            "frequency_weight": 0.40,
            "enable_gradcam": False,
        },
        {
            "id": "exp2_secondary_classifier_only",
            "name": "Exp 2: Primary + Secondary (Classifier-Only Regions)",
            "enable_secondary": True,
            "classifier_weight": 1.00,
            "frequency_weight": 0.00,
            "enable_gradcam": False,
        },
        {
            "id": "exp3_secondary_freq_fusion",
            "name": "Exp 3: Primary + Secondary + Frequency Regional Fusion",
            "enable_secondary": True,
            "classifier_weight": 0.60,
            "frequency_weight": 0.40,
            "enable_gradcam": False,
        },
        {
            "id": "exp4_full_multimodal",
            "name": "Exp 4: Full System (Primary + Secondary + Freq + Grad-CAM)",
            "enable_secondary": True,
            "classifier_weight": 0.60,
            "frequency_weight": 0.40,
            "enable_gradcam": True,
        },
    ]

    results = []

    for cfg in configs:
        print(f"\n--- Running [{cfg['name']}] ---")
        pipeline = CompleteDeepfakeDetectionPipeline(
            secondary_trigger_mode=SecondaryTriggerMode.ALWAYS,
            classifier_weight=cfg["classifier_weight"],
            frequency_weight=cfg["frequency_weight"],
            enable_secondary=cfg["enable_secondary"],
            enable_gradcam=cfg["enable_gradcam"],
        )

        exp_out_dir = out_path / cfg["id"]
        t0 = time.perf_counter()
        res = pipeline.analyze(image_path, output_dir=exp_out_dir)
        elapsed_sec = time.perf_counter() - t0

        primary = res.get("primary_system", {})
        sec = res.get("secondary_system", {})

        consistency_status = "N/A"
        region_count = 0
        susp_count = 0
        if sec and sec.get("executed"):
            region_count = sec.get("region_count", 0)
            susp_count = sec.get("suspicious_region_count", 0)
            if sec.get("regional_consistency"):
                consistency_status = sec["regional_consistency"]["status"]

        record = {
            "experiment_id": cfg["id"],
            "experiment_name": cfg["name"],
            "latency_seconds": round(elapsed_sec, 3),
            "global_verdict": primary.get("global_verdict", "N/A"),
            "global_confidence": primary.get("confidence_display", "N/A"),
            "regions_evaluated": region_count,
            "suspicious_regions": susp_count,
            "consistency_status": consistency_status,
            "gradcam_generated": "gradcam_regional" in res.get("visualizations", {}) or "gradcam_global" in res.get("visualizations", {}),
            "artifacts_count": len(res.get("visualizations", {})),
        }
        results.append(record)

        print(f"  Verdict:            {record['global_verdict']}")
        print(f"  Confidence:         {record['global_confidence']}")
        print(f"  Regions Analyzed:   {record['regions_evaluated']} (Suspicious: {record['suspicious_regions']})")
        print(f"  Consistency Status: {record['consistency_status']}")
        print(f"  Latency:            {record['latency_seconds']:.2f}s")

    # Generate Markdown Table
    md_table = "| Experiment | Global Verdict | Confidence | Regions | Consistency | Latency |\n"
    md_table += "| :--- | :--- | :--- | :---: | :---: | :---: |\n"
    for r in results:
        md_table += f"| **{r['experiment_name']}** | {r['global_verdict']} | {r['global_confidence']} | {r['regions_evaluated']} (Susp: {r['suspicious_regions']}) | {r['consistency_status']} | {r['latency_seconds']:.2f}s |\n"

    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY TABLE")
    print("=" * 80)
    print(md_table)

    summary_payload = {
        "target_image": str(image_path),
        "results": results,
    }

    # Save to disk
    json_path = out_path / "ablation_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    md_path = out_path / "ablation_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Ablation Study Summary\n\nTarget: `{Path(image_path).name}`\n\n{md_table}\n")

    print(f"[OK] Ablation results saved to: {json_path}")
    print(f"[OK] Markdown report saved to:   {md_path}")
    return summary_payload


def main():
    parser = argparse.ArgumentParser(description="Run Ablation Study across 4 detection configurations.")
    parser.add_argument(
        "image_path",
        nargs="?",
        default=r"C:\Users\KEERTHANA S.T\OneDrive\Pictures\Saved Pictures\img2.jpeg",
        help="Path to input image for ablation testing",
    )
    parser.add_argument("--output_dir", default="outputs/ablation", help="Directory for ablation results")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"[ERROR] Target image does not exist: {args.image_path}")
        sys.exit(1)

    run_ablation_experiment(args.image_path, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
