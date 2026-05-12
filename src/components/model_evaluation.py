"""
Model Evaluation Component
Handles evaluation and metrics calculation for fraud detection models
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc
)
from typing import Dict, Any, Tuple


class ModelEvaluation:
    """
    Component for evaluating machine learning models
    """

    def __init__(self):
        """Initialize model evaluation component"""
        pass

    def evaluate_binary_classifier(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        Evaluate binary classification model

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (for AUC metrics)

        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {
            "classification_report": classification_report(y_true, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
        }

        if y_prob is not None:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)

            # Precision-Recall AUC
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            metrics["pr_auc"] = auc(recall, precision)

        return metrics

    def calculate_fraud_capture_at_k(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        k_values: list = None
    ) -> Dict[float, float]:
        """
        Calculate fraud capture rate at different percentile thresholds

        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            k_values: List of k values (percentiles) to evaluate

        Returns:
            Dictionary with fraud capture rates for each k
        """
        if k_values is None:
            k_values = [0.01, 0.03, 0.05, 0.10]

        results = {}

        for k in k_values:
            df_eval = pd.DataFrame({
                "y": y_true,
                "prob": y_prob
            }).sort_values("prob", ascending=False)

            cutoff = int(len(df_eval) * k)
            fraud_captured = df_eval.head(cutoff)["y"].sum()
            total_fraud = y_true.sum()

            capture_rate = fraud_captured / total_fraud if total_fraud > 0 else 0
            results[k] = capture_rate

        return results

    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        target_percentile: float = 99
    ) -> float:
        """
        Find optimal threshold based on percentile

        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            target_percentile: Target percentile for threshold

        Returns:
            Optimal threshold value
        """
        threshold = np.percentile(y_prob, target_percentile)
        return threshold

    def evaluate_threshold_performance(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float
    ) -> Dict[str, Any]:
        """
        Evaluate model performance at specific threshold

        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            threshold: Decision threshold

        Returns:
            Dictionary with threshold-based evaluation metrics
        """
        y_pred_threshold = (y_prob >= threshold).astype(int)

        metrics = self.evaluate_binary_classifier(y_true, y_pred_threshold, y_prob)
        metrics["threshold"] = threshold

        return metrics

    def print_evaluation_report(self, model_name: str, metrics: Dict[str, Any]):
        """
        Print formatted evaluation report

        Args:
            model_name: Name of the model
            metrics: Evaluation metrics dictionary
        """
        print(f"\n{'='*60}")
        print(f"{model_name.upper()} EVALUATION REPORT")
        print('='*60)

        if "roc_auc" in metrics:
            print(".4f")
        if "pr_auc" in metrics:
            print(".4f")

        print("\nClassification Report:")
        print(classification_report(
            y_true=None, y_pred=None,
            labels=list(range(len(metrics["classification_report"]) - 3)),
            target_names=["Non-Fraud", "Fraud"],
            output_dict=False
        ))

        print("\nConfusion Matrix:")
        cm = metrics["confusion_matrix"]
        print(f"[[{cm[0][0]:4d} {cm[0][1]:4d}]")
        print(f" [{cm[1][0]:4d} {cm[1][1]:4d}]]")
        print("(True Negative  False Positive)")
        print("(False Negative True Positive )")