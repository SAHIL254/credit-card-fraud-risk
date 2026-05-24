"""
Prediction Pipeline
Handles real-time fraud risk prediction for new transactions
"""

import sys
from pathlib import Path
# Ensure project root is on sys.path when running module directly
proj_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(proj_root))

import numpy as np
import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import json

from src.logger import get_logger
from src.exception import PredictionError, ArtifactNotFoundError
from src.utils import (
    ARTIFACTS_DIR,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    FEATURE_COLUMNS_PATH,
    REFERENCE_SCORES_PATH,
    HIGH_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD
)


class PredictionPipeline:
    """
    Pipeline for making fraud risk predictions on new transactions
    """

    def __init__(self):
        """
        Initialize prediction pipeline
        """

        self.logger = get_logger(__name__)

        # Artifacts directory
        self.artifacts_dir = Path(ARTIFACTS_DIR)

        # Risk thresholds
        self.high_risk_threshold = HIGH_RISK_THRESHOLD
        self.medium_risk_threshold = MEDIUM_RISK_THRESHOLD

        # Load artifacts
        self._load_artifacts()



        self.logger.info("PredictionPipeline initialized")

    def _load_artifacts(self):
        """Load trained model and preprocessing artifacts"""
        try:
            self.logger.info("Loading prediction artifacts...")

            # load preprocessor, feature columns, reference scores first
            self.preprocessor = joblib.load(PREPROCESSOR_PATH)
            self.feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
            self.reference_scores = np.load(REFERENCE_SCORES_PATH)

            # Discover model candidates in artifacts directory
            artifacts_path = Path(ARTIFACTS_DIR)
            model_files = list(artifacts_path.glob("*model*.pkl"))

            if not model_files:
                # fallback to legacy single model path
                model_files = [Path(MODEL_PATH)] if Path(MODEL_PATH).exists() else []

            if not model_files:
                raise FileNotFoundError("No model artifact found in artifacts directory")

            # If only one model, load it
            if len(model_files) == 1:
                self.model = joblib.load(str(model_files[0]))
                self.logger.info(f"Loaded model: {model_files[0].name}")
            else:
                # Multiple models: attempt to pick best using heuristics
                scores_path = artifacts_path / "model_scores.json"
                best_model_path = None
                if scores_path.exists():
                    import json
                    with open(scores_path, "r", encoding="utf-8") as fh:
                        scores = json.load(fh)
                    # scores expected as {"model_filename": score}
                    best_name = max(scores.items(), key=lambda x: x[1])[0]
                    candidate = artifacts_path / best_name
                    if candidate.exists():
                        best_model_path = candidate

                # If no scores.json or candidate missing, prefer RandomForest if present
                if best_model_path is None:
                    rf_candidates = [p for p in model_files if "random" in p.name.lower() or "rf" in p.name.lower()]
                    if rf_candidates:
                        best_model_path = rf_candidates[0]

                # last resort: pick first file
                if best_model_path is None:
                    best_model_path = model_files[0]

                self.model = joblib.load(str(best_model_path))
                self.logger.info(f"Selected best model: {best_model_path.name}")

            self.logger.info("✅ Prediction pipeline artifacts loaded successfully")
        except FileNotFoundError as e:
            self.logger.error(f"Required artifact not found: {e}")
            raise ArtifactNotFoundError(f"Required artifact not found: {e}") from e
        except Exception as e:
            self.logger.error(f"Error loading artifacts: {str(e)}")
            raise PredictionError(f"Error loading artifacts: {str(e)}") from e

    def predict_risk(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict fraud risk for a single transaction

        Args:
            transaction_data: Dictionary with transaction details

        Returns:
            Dictionary with risk assessment
        """
        try:
            self.logger.info("Processing fraud risk prediction request")

            # Prepare input features
            input_df = self._prepare_transaction_features(transaction_data)

            # Transform features
            X_processed = self.preprocessor.transform(input_df)

            # Get prediction
            risk_score = self.model.predict_proba(X_processed)[0][1]

            # Calculate percentile
            percentile = (self.reference_scores < risk_score).mean() * 100

            # Determine risk level
            risk_level = self._classify_risk_level(percentile)

            result = {
                "risk_score": float(risk_score),
                "percentile": float(percentile),
                "risk_level": risk_level,
                "timestamp": datetime.now().isoformat()
            }

            self.logger.info(f"Prediction completed - Risk Level: {risk_level['label']}")
            return result

        except PredictionError:
            raise
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise PredictionError(f"Prediction failed: {str(e)}") from e

    def _prepare_transaction_features(self, transaction_data: Dict[str, Any]):
        """
        Prepare transaction data for model prediction
        """

        # Extract transaction details
        amount = transaction_data.get("Amount") or transaction_data.get("amount", 0)

        transaction_type = (
            transaction_data.get("TransactionType")
            or transaction_data.get("transaction_type", "purchase")
        )

        location = (
            transaction_data.get("Location")
            or transaction_data.get("location", "New York")
        )

        # Extract temporal features
        hour = transaction_data.get("hour", datetime.now().hour)
        day = transaction_data.get("day", datetime.now().day)
        weekday = transaction_data.get("weekday", datetime.now().weekday())

        # Handle datetime if provided
        transaction_time = (
            transaction_data.get("transaction_time")
            or transaction_data.get("TransactionTime")
        )

        if transaction_time:

            if isinstance(transaction_time, str):
                transaction_time = datetime.fromisoformat(transaction_time)

            hour = transaction_time.hour
            day = transaction_time.day
            weekday = transaction_time.weekday()

        # Derived features
        is_weekend = int(weekday in [5, 6])

        is_night = int(hour >= 0 and hour <= 5)

        high_amount = int(
            amount > 1000
        )  # you can adjust threshold based on training

        # Initialize all expected columns
        input_df = pd.DataFrame(
            np.zeros((1, len(self.feature_columns))),
            columns=self.feature_columns
        )

        # Fill numerical/base features
        base_features = {
            "Amount": amount,
            "hour": hour,
            "day": day,
            "weekday": weekday,
            "is_weekend": is_weekend,
            "is_night": is_night,
            "high_amount": high_amount
        }

        for col, value in base_features.items():
            if col in input_df.columns:
                input_df[col] = value

        # TransactionType encoding
        transaction_col = f"TransactionType_{transaction_type}"

        if transaction_col in input_df.columns:
            input_df[transaction_col] = 1

        # Location encoding
        location_col = f"Location_{location}"

        if location_col in input_df.columns:
            input_df[location_col] = 1

        return input_df

    def _classify_risk_level(self, percentile: float) -> Dict[str, str]:
        """
        Classify risk level based on percentile

        Args:
            percentile: Risk percentile (0-100)

        Returns:
            Dictionary with risk level information
        """
        if percentile >= self.high_risk_threshold:
            return {
                "level": "high",
                "label": "🔴 HIGH RISK",
                "color": "red"
            }
        elif percentile >= self.medium_risk_threshold:
            return {
                "level": "medium",
                "label": "🟠 MEDIUM RISK",
                "color": "orange"
            }
        else:
            return {
                "level": "low",
                "label": "🟢 LOW RISK",
                "color": "green"
            }

    def batch_predict(self, transactions: list) -> list:
        """
        Predict risk for multiple transactions

        Args:
            transactions: List of transaction dictionaries

        Returns:
            List of prediction results
        """
        self.logger.info(f"Processing batch prediction for {len(transactions)} transactions")
        results = []
        for i, transaction in enumerate(transactions):
            try:
                result = self.predict_risk(transaction)
                results.append(result)
            except Exception as e:
                self.logger.warning(f"Failed to predict transaction {i}: {str(e)}")
                results.append({
                    "error": str(e),
                    "transaction": transaction
                })

        self.logger.info(f"Batch prediction completed: {len(results)} results")
        return results

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model

        Returns:
            Dictionary with model information
        """
        return {
            "feature_count": len(self.feature_columns),
            "features": self.feature_columns,
            "reference_scores_shape": self.reference_scores.shape,
            "high_risk_threshold": self.high_risk_threshold,
            "medium_risk_threshold": self.medium_risk_threshold
        }