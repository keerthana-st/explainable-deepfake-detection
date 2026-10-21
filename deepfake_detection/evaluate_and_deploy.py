"""
Evaluate and Deploy the Fine-Tuned Model.
Loads deepfake_detection/models/best_model.pth, evaluates test set,
generates confusion matrix and ROC plots, and deploys to the multi-agent system.
"""

import json
import shutil
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from deepfake_detection.src.dataset import build_dataloaders
from deepfake_detection.src.evaluate import evaluate_test_set
from deepfake_detection.src.model import EfficientNetDeepfakeClassifier
from deepfake_detection.src.predict import predict_image


def main():
    print("\n============================================================")
    print("      EVALUATING AND DEPLOYING FINE-TUNED MODEL")
    print("============================================================")

    models_dir = PROJECT_ROOT / "deepfake_detection" / "models"
    results_dir = PROJECT_ROOT / "deepfake_detection" / "results"
    dataset_dir = PROJECT_ROOT / "deepfake_detection" / "dataset"
    ckpt_path = models_dir / "best_model.pth"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    # 1. Load checkpoint metadata and temperature
    ckpt = torch.load(ckpt_path, map_location="cpu")
    calibrated_temp = float(ckpt.get("temperature", 1.0))
    print(f"  • Checkpoint Epoch:   {ckpt.get('epoch')}")
    print(f"  • Validation F1:      {ckpt.get('val_f1', 0.0):.4f}")
    print(f"  • Validation Loss:    {ckpt.get('val_loss', 0.0):.4f}")
    print(f"  • Calibrated Temp T:  {calibrated_temp:.4f}")

    # 2. Load model with trained weights
    model = EfficientNetDeepfakeClassifier(pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # 3. Build test dataloader
    _, _, test_loader, _ = build_dataloaders(
        dataset_dir=dataset_dir,
        batch_size=32,
        num_workers=0,
    )

    # 4. Run test set forensic evaluation
    metrics = evaluate_test_set(
        model=model,
        test_loader=test_loader,
        temperature=calibrated_temp,
        results_dir=results_dir,
    )

    # 5. Deploy to multi-agent system paths
    system_models_dir = PROJECT_ROOT / "models"
    system_models_dir.mkdir(parents=True, exist_ok=True)
    target_pt = system_models_dir / "classifier_deepfake_weights.pt"
    target_pth = system_models_dir / "efficientnet_deepfake.pth"

    shutil.copy2(ckpt_path, target_pt)
    shutil.copy2(ckpt_path, target_pth)

    print("\n============================================================")
    print("      DEPLOYMENT TO MULTI-AGENT SYSTEM COMPLETE")
    print("============================================================")
    print(f"  [OK] Fine-tuned checkpoint: {ckpt_path}")
    print(f"  [OK] Live system weights:   {target_pt}")
    print(f"  [OK] Live system weights:   {target_pth}")
    print(f"  [OK] Test Accuracy:         {metrics['accuracy']*100:.2f}%")
    print(f"  [OK] Test Precision:        {metrics['precision']*100:.2f}%")
    print(f"  [OK] Test Fake-Recall:      {metrics['recall']*100:.2f}%")
    print(f"  [OK] Test Specificity:      {metrics['specificity']*100:.2f}%")
    print(f"  [OK] Test F1-Score:         {metrics['f1_score']:.4f}")
    print(f"  [OK] Test ROC-AUC:          {metrics['roc_auc']:.4f}")
    print(f"  [OK] False Negative Rate:   {metrics['false_negative_rate']*100:.2f}%")
    print(f"  [OK] False Positive Rate:   {metrics['false_positive_rate']*100:.2f}%")
    print("============================================================\n")

    # 6. Verify single image inference
    print("[Verification] Running predict_image on real and fake benchmarks...")
    real_sample = "C:/Users/KEERTHANA S.T/OneDrive/Pictures/Saved Pictures/img2.jpeg"
    fake_sample = "D:/dog.jpg"

    if Path(real_sample).exists():
        res_real = predict_image(real_sample, model_path=ckpt_path)
        print(f"  * Real Test ({Path(real_sample).name}): Predicted={res_real['prediction']} (Fake Prob={res_real['fake_probability']:.4f}, Confidence={res_real['confidence_percentage']})")

    if Path(fake_sample).exists():
        res_fake = predict_image(fake_sample, model_path=ckpt_path)
        print(f"  * Fake Test ({Path(fake_sample).name}): Predicted={res_fake['prediction']} (Fake Prob={res_fake['fake_probability']:.4f}, Confidence={res_fake['confidence_percentage']})")


if __name__ == "__main__":
    main()
