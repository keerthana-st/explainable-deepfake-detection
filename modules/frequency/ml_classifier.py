"""
Explainable Tabular ML Classifier for Frequency Agent.

Implements a Cost-Sensitive ElasticNet Logistic Regression model operating on
the 28-dimensional frequency feature vector.

Features:
- Strict per-feature explainability (signed feature coefficients and attributions)
- Calibrated Indifference Zone ([0.35, 0.65]) yielding INCONCLUSIVE_FREQUENCY
- False-positive cost weighting penalizing false alarms on compressed photos
- Pure NumPy implementation for zero dependency bloat and instant CPU inference
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class FrequencyMLClassifier:
    """
    Cost-sensitive, regularized linear/tabular classifier for 28-D frequency features.
    """

    def __init__(
        self,
        indifference_low: float = 0.35,
        indifference_high: float = 0.65,
        threshold: float = 0.50,
    ):
        self.indifference_low = indifference_low
        self.indifference_high = indifference_high
        self.threshold = threshold

        # Feature normalization parameters
        self.means: Optional[np.ndarray] = None
        self.stds: Optional[np.ndarray] = None

        # Model weights (bias and 28 feature coefficients)
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.feature_names: Optional[List[str]] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        l1_ratio: float = 0.25,
        alpha: float = 0.01,
        learning_rate: float = 0.05,
        max_iter: int = 1000,
        fake_weight: float = 1.0,
        real_weight: float = 2.0,  # Penalize false alarms on real compressed photos
    ):
        """
        Fit ElasticNet Logistic Regression via coordinate / gradient descent with cost-sensitivity.
        """
        self.feature_names = feature_names
        N, D = X.shape

        # Standardize features
        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0) + 1e-6
        X_norm = (X - self.means) / self.stds

        # Sample weights
        sample_weights = np.where(y == 1, fake_weight, real_weight).astype(np.float32)
        sample_weights /= np.mean(sample_weights)

        # Initialize weights
        w = np.zeros(D, dtype=np.float64)
        b = 0.0

        for it in range(max_iter):
            # Sigmoid probabilities
            logits = np.dot(X_norm, w) + b
            logits = np.clip(logits, -20.0, 20.0)
            probs = 1.0 / (1.0 + np.exp(-logits))

            # Weighted residuals
            res = (probs - y) * sample_weights

            # Gradients
            dw = np.dot(X_norm.T, res) / N
            db = float(np.mean(res))

            # ElasticNet regularization gradient (L2 smooth + L1 soft threshold)
            l2_grad = alpha * (1.0 - l1_ratio) * w
            dw += l2_grad

            # Update bias
            b -= learning_rate * db

            # Update weights with proximal soft-thresholding for L1
            w -= learning_rate * dw
            l1_penalty = learning_rate * alpha * l1_ratio
            w = np.sign(w) * np.maximum(0.0, np.abs(w) - l1_penalty)

        self.weights = w
        self.bias = b

    def predict_probability(self, x_vec: np.ndarray) -> float:
        """Compute calibrated probability that frequency signature is FAKE."""
        if self.weights is None or self.means is None:
            # Fallback heuristic if not trained yet
            return 0.50
        x_norm = (x_vec - self.means) / self.stds
        logit = float(np.dot(x_norm, self.weights) + self.bias)
        logit = np.clip(logit, -20.0, 20.0)
        return float(1.0 / (1.0 + np.exp(-logit)))

    def explain(self, x_vec: np.ndarray, top_k: int = 4) -> Dict[str, Any]:
        """
        Provide signed per-feature attributions explaining why the model made its decision.
        """
        prob = self.predict_probability(x_vec)

        if self.weights is None or self.means is None or self.feature_names is None:
            return {
                "probability": prob,
                "verdict": "INCONCLUSIVE_FREQUENCY",
                "attributions": [],
                "summary": "Model uncalibrated.",
            }

        x_norm = (x_vec - self.means) / self.stds
        attributions = x_norm * self.weights

        # Sort features by magnitude of contribution
        ranked_idx = np.argsort(np.abs(attributions))[::-1]

        top_features = []
        for idx in ranked_idx[:top_k]:
            fname = self.feature_names[idx]
            impact = float(attributions[idx])
            direction = "Towards Fake / Manipulated" if impact > 0 else "Towards Authentic / Real"
            top_features.append(
                {
                    "feature": fname,
                    "raw_value": float(x_vec[idx]),
                    "z_score": float(x_norm[idx]),
                    "coefficient": float(self.weights[idx]),
                    "impact": impact,
                    "direction": direction,
                }
            )

        if prob > self.indifference_high:
            verdict = "MANIPULATED_FREQUENCY"
            reasoning = (
                f"Elevated frequency anomalies detected (Probability: {prob:.4f} > {self.indifference_high}). "
                f"Primary drivers: {top_features[0]['feature']} ({top_features[0]['direction']})."
            )
        elif prob < self.indifference_low:
            verdict = "AUTHENTIC_FREQUENCY"
            reasoning = (
                f"Natural frequency distribution verified (Probability: {prob:.4f} < {self.indifference_low}). "
                f"Primary corroboration: {top_features[0]['feature']} ({top_features[0]['direction']})."
            )
        else:
            verdict = "INCONCLUSIVE_FREQUENCY"
            reasoning = (
                f"Frequency indicators are in the calibrated indifference zone "
                f"({prob:.4f} in [{self.indifference_low}, {self.indifference_high}]). "
                f"Frequency domain cannot rule out diffusion smoothing or compression ambiguity."
            )

        return {
            "probability": prob,
            "verdict": verdict,
            "attributions": top_features,
            "reasoning": reasoning,
        }

    def save_model(self, file_path: Path):
        """Save model parameters to JSON."""
        data = {
            "bias": float(self.bias),
            "weights": self.weights.tolist() if self.weights is not None else [],
            "means": self.means.tolist() if self.means is not None else [],
            "stds": self.stds.tolist() if self.stds is not None else [],
            "feature_names": self.feature_names or [],
            "indifference_low": self.indifference_low,
            "indifference_high": self.indifference_high,
            "threshold": self.threshold,
        }
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    def load_model(self, file_path: Path):
        """Load model parameters from JSON."""
        with open(file_path, "r") as f:
            data = json.load(f)
        self.bias = float(data["bias"])
        self.weights = np.array(data["weights"], dtype=np.float64)
        self.means = np.array(data["means"], dtype=np.float64)
        self.stds = np.array(data["stds"], dtype=np.float64)
        self.feature_names = data.get("feature_names", [])
        self.indifference_low = data.get("indifference_low", 0.35)
        self.indifference_high = data.get("indifference_high", 0.65)
        self.threshold = data.get("threshold", 0.50)
