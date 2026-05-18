"""
Data Transformation Component
Handles data preprocessing and cleaning operations
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from src.logger import get_logger
from src.exception import DataTransformationError
from src.utils import NUMERIC_FEATURES, TEST_SIZE, RANDOM_STATE, TARGET_COLUMN


class DataTransformation:
    """
    Component for transforming and preprocessing credit card fraud data
    """

    def __init__(self, test_size: float = None, random_state: int = None):
        """
        Initialize data transformation component

        Args:
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
        """
        self.logger = get_logger(__name__)
        self.test_size = test_size if test_size is not None else TEST_SIZE
        self.random_state = random_state if random_state is not None else RANDOM_STATE
        self.numeric_features = NUMERIC_FEATURES
        self.target_column = TARGET_COLUMN
        self.logger.info("DataTransformation initialized")

    def split_data(self, X: pd.DataFrame, y: pd.Series):
        """
        Split data into training and testing sets with stratification

        Args:
            X: Features DataFrame
            y: Target Series

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        try:
            self.logger.info("Splitting data into train and test sets")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                stratify=y,
                random_state=self.random_state
            )

            self.logger.info(f"✅ Data split completed")
            self.logger.info(f"   Train: {X_train.shape[0]} samples")
            self.logger.info(f"   Test: {X_test.shape[0]} samples")

            return X_train, X_test, y_train, y_test
        except Exception as e:
            self.logger.error(f"Error splitting data: {str(e)}")
            raise DataTransformationError(f"Data split failed: {str(e)}") from e

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

    def preprocess_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame, preprocessor: ColumnTransformer = None):
        """
        Apply preprocessing to training and testing data

        Args:
            X_train: Training features
            X_test: Testing features
            preprocessor: Fitted preprocessor (optional, will create if None)

        Returns:
            Tuple of (X_train_processed, X_test_processed, preprocessor)
        """
        try:
            self.logger.info("Preprocessing data")
            if preprocessor is None:
                preprocessor = self.create_preprocessor()
            
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)

            self.logger.info("✅ Data preprocessing completed")
            self.logger.info(f"   Processed train shape: {X_train_processed.shape}")
            self.logger.info(f"   Processed test shape: {X_test_processed.shape}")

            return X_train_processed, X_test_processed, preprocessor
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {str(e)}")
            raise DataTransformationError(f"Data preprocessing failed: {str(e)}") from e

    def prepare_features_and_target(self, df: pd.DataFrame):
        """
        Prepare features (X) and target (y) from processed DataFrame

        Args:
            df: Processed DataFrame with features and target

        Returns:
            Tuple of (X, y)
        """
        try:
            self.logger.info("Preparing features and target")
            X = df.drop(self.target_column, axis=1)
            y = df[self.target_column]

            self.logger.info(f"✅ Features and target prepared")
            self.logger.info(f"   Features shape: {X.shape}")
            self.logger.info(f"   Target distribution: {y.value_counts().to_dict()}")

            return X, y
        except Exception as e:
            self.logger.error(f"Error preparing features and target: {str(e)}")
            raise DataTransformationError(f"Feature preparation failed: {str(e)}") from e