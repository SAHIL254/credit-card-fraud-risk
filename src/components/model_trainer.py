"""
Model Trainer Component
Handles training of machine learning models for fraud detection
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from typing import Dict, Any, Tuple


class ModelTrainer:
    """
    Component for training various machine learning models
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize model trainer

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state

        # Model configurations
        self.model_configs = {
            "isolation_forest": {
                "n_estimators": 200,
                "contamination": 0.01,
                "random_state": self.random_state
            },
            "logistic_regression": {
                "max_iter": 1000,
                "class_weight": "balanced"
            },
            "random_forest": {
                "n_estimators": 300,
                "class_weight": "balanced",
                "random_state": self.random_state,
                "n_jobs": -1
            }
        }

    def train_isolation_forest(self, X_train_normal: np.ndarray) -> IsolationForest:
        """
        Train Isolation Forest model on normal (non-fraud) transactions

        Args:
            X_train_normal: Training data with only non-fraud samples

        Returns:
            Trained IsolationForest model
        """
        config = self.model_configs["isolation_forest"]
        model = IsolationForest(**config)
        model.fit(X_train_normal)

        print("✅ Isolation Forest trained successfully")
        return model

    def train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
        """
        Train Logistic Regression model

        Args:
            X_train: Training features
            y_train: Training target

        Returns:
            Trained LogisticRegression model
        """
        config = self.model_configs["logistic_regression"]
        model = LogisticRegression(**config)
        model.fit(X_train, y_train)

        print("✅ Logistic Regression trained successfully")
        return model

    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
        """
        Train Random Forest model

        Args:
            X_train: Training features
            y_train: Training target

        Returns:
            Trained RandomForestClassifier model
        """
        config = self.model_configs["random_forest"]
        model = RandomForestClassifier(**config)
        model.fit(X_train, y_train)

        print("✅ Random Forest trained successfully")
        return model

    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray, X_train_normal: np.ndarray = None) -> Dict[str, Any]:
        """
        Train all available models

        Args:
            X_train: Training features
            y_train: Training target
            X_train_normal: Normal transactions for Isolation Forest (optional)

        Returns:
            Dictionary with trained models
        """
        models = {}

        # Train Logistic Regression
        models["logistic_regression"] = self.train_logistic_regression(X_train, y_train)

        # Train Random Forest
        models["random_forest"] = self.train_random_forest(X_train, y_train)

        # Train Isolation Forest (if normal data provided)
        if X_train_normal is not None:
            models["isolation_forest"] = self.train_isolation_forest(X_train_normal)

        print(f"✅ All models trained: {list(models.keys())}")
        return models