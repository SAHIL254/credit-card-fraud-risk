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

from src.logger import get_logger
from src.exception import ModelEvaluationError


class ModelEvaluation:
    """
    Component for evaluating machine learning models
    """

    def __init__(self):
        """Initialize model evaluation component"""
        self.logger = get_logger(__name__)
        self.logger.info("ModelEvaluation initialized")

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
        try:
            self.logger.info("Evaluating binary classifier...")
            metrics = {
                "classification_report": classification_report(y_true, y_pred, output_dict=True),
                "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
            }

            if y_prob is not None:
                metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
                precision, recall, _ = precision_recall_curve(y_true, y_prob)
                metrics["pr_auc"] = auc(recall, precision)

            self.logger.info("✅ Model evaluation completed")
            return metrics

        except Exception as e:
            self.logger.error(f"Model evaluation failed: {str(e)}")
            raise ModelEvaluationError(f"Model evaluation failed: {str(e)}") from e

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
        try:
            if k_values is None:
                k_values = [0.01, 0.03, 0.05, 0.10]

            self.logger.info(f"Calculating fraud capture at k = {k_values}")

            df_eval = pd.DataFrame({
                "y": y_true,
                "prob": y_prob
            }).sort_values("prob", ascending=False)

            results = {}
            for k in k_values:
                cutoff = int(len(df_eval) * k)
                fraud_captured = df_eval.head(cutoff)["y"].sum()
                total_fraud = y_true.sum()
                capture_rate = fraud_captured / total_fraud if total_fraud > 0 else 0
                results[k] = capture_rate

            self.logger.info("✅ Fraud capture calculation completed")
            return results

        except Exception as e:
            self.logger.error(f"Fraud capture calculation failed: {str(e)}")
            raise ModelEvaluationError(f"Fraud capture calculation failed: {str(e)}") from e

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
        try:
            self.logger.info(f"Printing evaluation report for {model_name}")
            self.logger.info('=' * 60)
            self.logger.info(f"{model_name.upper()} EVALUATION REPORT")
            self.logger.info('=' * 60)

            if "roc_auc" in metrics:
                self.logger.info(f"ROC-AUC Score: {metrics['roc_auc']:.4f}")
            if "pr_auc" in metrics:
                self.logger.info(f"PR-AUC Score: {metrics['pr_auc']:.4f}")

            self.logger.info("Classification Report:")
            cr = metrics.get("classification_report", {})
            # Log header
            self.logger.info(f"{'label':<15}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}")
            for label, vals in cr.items():
                # skip aggregate rows if they don't have support
                try:
                    precision = vals.get('precision', 0.0)
                    recall = vals.get('recall', 0.0)
                    f1 = vals.get('f1-score', 0.0)
                    support = int(vals.get('support', 0))
                except Exception:
                    continue
                # log formatted line
                self.logger.info(f"{label:<15}{precision:10.2f}{recall:10.2f}{f1:10.2f}{support:10d}")

            self.logger.info("Confusion Matrix:")
            cm = metrics["confusion_matrix"]
            self.logger.info(f"[[{cm[0][0]:4d} {cm[0][1]:4d}]")
            self.logger.info(f" [{cm[1][0]:4d} {cm[1][1]:4d}]]")
            self.logger.info("(True Negative  False Positive)")
            self.logger.info("(False Negative True Positive )")

        except Exception as e:
            self.logger.error(f"Error printing evaluation report: {str(e)}")
            raise ModelEvaluationError(f"Error printing evaluation report: {str(e)}") from e