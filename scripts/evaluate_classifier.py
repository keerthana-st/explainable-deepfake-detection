"""
CLI Runner for Classifier Agent 4-Category Evaluation.

Evaluates the Classifier Agent explicitly across:
1. Known-real ordinary photos
2. Known GAN-based fakes
3. Known modern diffusion fakes (ChatGPT/DALL-E 3, Midjourney)
4. Compressed real photos (social media simulation)

Prints a category-separated confusion matrix per project specification.
"""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.classifier.agent import ClassifierAgent
from modules.classifier.dataset import generate_four_category_benchmark
from modules.classifier.evaluation import evaluate_classifier_agent


def main():
    parser = argparse.ArgumentParser(description="Evaluate Classifier Agent across 4 mandatory categories")
    parser.add_argument("--samples-per-cat", type=int, default=20, help="Number of benchmark samples per category")
    parser.add_argument("--output-json", type=str, default="data/classifier_evaluation.json", help="Path to export evaluation metrics")
    args = parser.parse_args()

    bench_dir = PROJECT_ROOT / "data" / "classifier_benchmark"
    print("==============================================================================")
    print("         GENERATING 4-CATEGORY BENCHMARK DATASET")
    print("==============================================================================")
    print(f"Directory: {bench_dir}")
    print(f"Samples per category: {args.samples_per_cat} (Total {args.samples_per_cat * 4} images)")

    benchmark_paths = generate_four_category_benchmark(bench_dir, samples_per_category=args.samples_per_cat)

    print("\nRunning Classifier Agent inference across all categories...")
    agent = ClassifierAgent()
    matrix = evaluate_classifier_agent(agent, benchmark_paths)

    # Print category-separated report
    matrix.print_report()

    # Export metrics JSON
    json_path = PROJECT_ROOT / args.output_json
    json_path.parent.mkdir(parents=True, exist_ok=True)
    export_dict = {
        cat: {
            "ground_truth": m.ground_truth,
            "total_samples": m.total_samples,
            "correct": m.correct_predictions,
            "incorrect": m.incorrect_predictions,
            "accuracy": round(m.accuracy, 4),
            "mean_confidence_fake": round(m.mean_confidence, 4),
            "pred_real": m.predicted_real_count,
            "pred_fake": m.predicted_fake_count,
        }
        for cat, m in matrix.categories.items()
    }
    json_path.write_text(json.dumps(export_dict, indent=2), encoding="utf-8")
    print(f"\nEvaluation metrics exported to: {json_path}")


if __name__ == "__main__":
    main()
