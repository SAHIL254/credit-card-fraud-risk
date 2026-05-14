"""
Model Trainer Component
Handles training of machine learning models for fraud detection
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from typing import Dict, Any

from src.logger import get_logger
from src.exception import ModelTrainingError
from src.utils import (
    ISO_FOREST_PARAMS,
    LOG_REG_PARAMS,
    RF_PARAMS
)


class ModelTrainer:
    """
    Component for training machine learning models
    """

    def __init__(self):
        """Initialize model trainer component"""
        self.logger = get_logger(__name__)
        self.logger.info("ModelTrainer initialized")

    def train_isolation_forest(self, X_train_normal: np.ndarray) -> IsolationForest:
        """
        Train Isolation Forest model on normal (non-fraud) transactions

        Args:
            X_train_normal: Training data with only non-fraud samples

        Returns:
            Trained IsolationForest model
        """
        try:
            self.logger.info("Training Isolation Forest...")
            model = IsolationForest(**ISO_FOREST_PARAMS)
            model.fit(X_train_normal)
            self.logger.info("✅ Isolation Forest trained successfully")
            return model
        except Exception as e:
            self.logger.error(f"Isolation Forest training failed: {str(e)}")
            raise ModelTrainingError(f"Isolation Forest training failed: {str(e)}") from e

    def train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
        """
        Train Logistic Regression model

        Args:
            X_train: Training features
            y_train: Training target

        Returns:
            Trained LogisticRegression model
        """
        try:
            self.logger.info("Training Logistic Regression...")
            model = LogisticRegression(**LOG_REG_PARAMS)
            model.fit(X_train, y_train)
            self.logger.info("✅ Logistic Regression trained successfully")
            return model
        except Exception as e:
            self.logger.error(f"Logistic Regression training failed: {str(e)}")
            raise ModelTrainingError(f"Logistic Regression training failed: {str(e)}") from e

    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
        """
        Train Random Forest model

        Args:
            X_train: Training features
            y_train: Training target

        Returns:
            Trained RandomForestClassifier model
        """
        try:
            self.logger.info("Training Random Forest...")
            model = RandomForestClassifier(**RF_PARAMS)
            model.fit(X_train, y_train)
            self.logger.info("✅ Random Forest trained successfully")
            return model
        except Exception as e:
            self.logger.error(f"Random Forest training failed: {str(e)}")
            raise ModelTrainingError(f"Random Forest training failed: {str(e)}") from e

    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray, X_train_normal: np.ndarray) -> Dict[str, Any]:
        """
        Train all models (Isolation Forest, Logistic Regression, Random Forest)

        Args:
            X_train: Training features
            y_train: Training target
            X_train_normal: Normal transaction features for Isolation Forest

        Returns:
            Dictionary containing all trained models
        """
        try:
            self.logger.info("Training all models...")

            models = {}

            # Train Isolation Forest
            models["isolation_forest"] = self.train_isolation_forest(X_train_normal)

            # Train Logistic Regression
            models["logistic_regression"] = self.train_logistic_regression(X_train, y_train)

            # Train Random Forest
            models["random_forest"] = self.train_random_forest(X_train, y_train)

            self.logger.info(f"✅ All models trained: {list(models.keys())}")
            return models

        except Exception as e:
            self.logger.error(f"Model training pipeline failed: {str(e)}")
            raise ModelTrainingError(f"Model training pipeline failed: {str(e)}") from e