"""
Deepfake Detection Training and Fine-Tuning Package.
"""

from .dataset import (
    DeepfakeDataset,
    build_dataloaders,
    create_leakage_free_splits,
    download_kaggle_dataset,
    get_data_transforms,
    inspect_and_clean_dataset,
)
from .evaluate import evaluate_test_set
from .gradcam import EfficientNetGradCAM
from .model import EfficientNetDeepfakeClassifier, resolve_label_mapping
from .predict import predict_image
from .train import train_deepfake_classifier
from .utils import (
    TemperatureScaler,
    get_device,
    plot_confusion_matrix_heatmap,
    plot_roc_curve_chart,
    plot_training_history,
    set_seed,
)

__all__ = [
    "DeepfakeDataset",
    "build_dataloaders",
    "create_leakage_free_splits",
    "download_kaggle_dataset",
    "get_data_transforms",
    "inspect_and_clean_dataset",
    "evaluate_test_set",
    "EfficientNetGradCAM",
    "EfficientNetDeepfakeClassifier",
    "resolve_label_mapping",
    "predict_image",
    "train_deepfake_classifier",
    "TemperatureScaler",
    "get_device",
    "plot_confusion_matrix_heatmap",
    "plot_roc_curve_chart",
    "plot_training_history",
    "set_seed",
]
