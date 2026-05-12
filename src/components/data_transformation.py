"""
Data Transformation Component
Handles data preprocessing and cleaning operations
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


class DataTransformation:
    """
    Component for transforming and preprocessing credit card fraud data
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize data transformation component

        Args:
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.numeric_features = ["Amount", "MerchantID", "hour", "day", "weekday"]
        self.target_column = "IsFraud"

    def split_data(self, X: pd.DataFrame, y: pd.Series):
        """
        Split data into training and testing sets with stratification

        Args:
            X: Features DataFrame
            y: Target Series

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state
        )

        print(f"✅ Data split completed:")
        print(f"   Train: {X_train.shape[0]} samples")
        print(f"   Test: {X_test.shape[0]} samples")

        return X_train, X_test, y_train, y_test

    def create_preprocessor(self) -> ColumnTransformer:
        """
        Create data preprocessor with StandardScaler for numeric features

        Returns:
            ColumnTransformer object
        """
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_features)
            ],
            remainder="passthrough"
        )
        return preprocessor

    def preprocess_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame, preprocessor: ColumnTransformer):
        """
        Apply preprocessing to training and testing data

        Args:
            X_train: Training features
            X_test: Testing features
            preprocessor: Fitted preprocessor

        Returns:
            Tuple of (X_train_processed, X_test_processed)
        """
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        print("✅ Data preprocessing completed")
        print(f"   Processed train shape: {X_train_processed.shape}")
        print(f"   Processed test shape: {X_test_processed.shape}")

        return X_train_processed, X_test_processed

    def prepare_features_and_target(self, df: pd.DataFrame):
        """
        Prepare features (X) and target (y) from processed DataFrame

        Args:
            df: Processed DataFrame with features and target

        Returns:
            Tuple of (X, y)
        """
        X = df.drop(self.target_column, axis=1)
        y = df[self.target_column]

        print(f"✅ Features and target prepared:")
        print(f"   Features shape: {X.shape}")
        print(f"   Target distribution: {y.value_counts().to_dict()}")

        return X, y