"""
Training Pipeline
Orchestrates the complete model training workflow
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from components.data_ingestion import DataIngestion
from components.data_transformation import DataTransformation
from components.feature_engineering import FeatureEngineering
from components.model_trainer import ModelTrainer
from components.model_evaluation import ModelEvaluation


class TrainingPipeline:
    """
    Complete training pipeline for credit card fraud detection
    """

    def __init__(self):
        """Initialize training pipeline"""
        self.data_ingestion = DataIngestion()
        self.feature_engineering = FeatureEngineering()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()
        self.model_evaluation = ModelEvaluation()

        # Create artifacts directory
        self.artifacts_dir = Path(__file__).parent.parent.parent / "artifacts"
        self.artifacts_dir.mkdir(exist_ok=True)

    def run_pipeline(self):
        """
        Execute the complete training pipeline
        """
        print("=" * 80)
        print("CREDIT CARD FRAUD DETECTION - TRAINING PIPELINE")
        print("=" * 80)

        try:
            # 1. Data Ingestion
            print("\n[1/8] Data Ingestion...")
            df = self.data_ingestion.load_data()
            self.data_ingestion.validate_data(df)

            # 2. Feature Engineering
            print("\n[2/8] Feature Engineering...")
            df_processed = self.feature_engineering.prepare_features(df)

            # 3. Data Preparation
            print("\n[3/8] Data Preparation...")
            X, y = self.data_transformation.prepare_features_and_target(df_processed)

            # 4. Data Splitting
            print("\n[4/8] Data Splitting...")
            X_train, X_test, y_train, y_test = self.data_transformation.split_data(X, y)

            # 5. Data Preprocessing
            print("\n[5/8] Data Preprocessing...")
            preprocessor = self.data_transformation.create_preprocessor()
            X_train_p, X_test_p = self.data_transformation.preprocess_data(X_train, X_test, preprocessor)

            # 6. Model Training
            print("\n[6/8] Model Training...")
            X_train_normal = X_train_p[y_train == 0]  # Normal transactions only
            models = self.model_trainer.train_all_models(X_train_p, y_train, X_train_normal)

            # 7. Model Evaluation
            print("\n[7/8] Model Evaluation...")

            # Evaluate Random Forest (primary model)
            rf_prob = models["random_forest"].predict_proba(X_test_p)[:, 1]
            rf_pred = models["random_forest"].predict(X_test_p)
            rf_metrics = self.model_evaluation.evaluate_binary_classifier(y_test, rf_pred, rf_prob)

            # Evaluate Logistic Regression
            lr_prob = models["logistic_regression"].predict_proba(X_test_p)[:, 1]
            lr_pred = models["logistic_regression"].predict(X_test_p)
            lr_metrics = self.model_evaluation.evaluate_binary_classifier(y_test, lr_pred, lr_prob)

            # Evaluate Isolation Forest
            iso_pred = models["isolation_forest"].predict(X_test_p)
            iso_pred = (iso_pred == -1).astype(int)
            iso_metrics = self.model_evaluation.evaluate_binary_classifier(y_test, iso_pred)

            # Fraud capture analysis
            fraud_capture = self.model_evaluation.calculate_fraud_capture_at_k(y_test, rf_prob)

            # Print results
            self.model_evaluation.print_evaluation_report("Random Forest", rf_metrics)
            self.model_evaluation.print_evaluation_report("Logistic Regression", lr_metrics)
            self.model_evaluation.print_evaluation_report("Isolation Forest", iso_metrics)

            print("\nFraud Capture Analysis (Random Forest):")
            for k, capture_rate in fraud_capture.items():
                print(f"  Top {int(k*100)}%: {capture_rate*100:.1f}% fraud captured")

            # 8. Save Artifacts
            print("\n[8/8] Saving Artifacts...")
            self._save_artifacts(
                models["random_forest"],
                preprocessor,
                X.columns.tolist(),
                models["random_forest"].predict_proba(X_train_p)[:, 1]
            )

            print("\n" + "=" * 80)
            print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            print("=" * 80)

            return {
                "models": models,
                "preprocessor": preprocessor,
                "feature_columns": X.columns.tolist(),
                "metrics": {
                    "random_forest": rf_metrics,
                    "logistic_regression": lr_metrics,
                    "isolation_forest": iso_metrics,
                    "fraud_capture": fraud_capture
                }
            }

        except Exception as e:
            print(f"\n❌ Pipeline failed with error: {str(e)}")
            raise

    def _save_artifacts(self, model, preprocessor, feature_columns, train_risk_scores):
        """
        Save trained model and preprocessing artifacts

        Args:
            model: Trained model
            preprocessor: Fitted preprocessor
            feature_columns: List of feature column names
            train_risk_scores: Risk scores from training data
        """
        # Save model
        model_path = self.artifacts_dir / "fraud_rf_model.pkl"
        joblib.dump(model, model_path)

        # Save preprocessor
        preprocessor_path = self.artifacts_dir / "preprocessor.pkl"
        joblib.dump(preprocessor, preprocessor_path)

        # Save feature columns
        feature_columns_path = self.artifacts_dir / "feature_columns.pkl"
        joblib.dump(feature_columns, feature_columns_path)

        # Save reference scores
        reference_scores_path = self.artifacts_dir / "reference_scores.npy"
        np.save(reference_scores_path, train_risk_scores)

        print("✅ All artifacts saved successfully!")


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    results = pipeline.run_pipeline()