"""
Feature Engineering Component
Handles feature extraction and preprocessing for fraud detection
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List

from src.logger import get_logger
from src.exception import FeatureEngineeringError
from src.utils import NUMERIC_FEATURES, CATEGORICAL_FEATURES, ID_COLUMN, DATE_COLUMN


class FeatureEngineering:
    """
    Component for feature engineering and preprocessing
    """

    def __init__(self):
        """Initialize feature engineering component"""
        self.logger = get_logger(__name__)
        self.numeric_features = NUMERIC_FEATURES
        self.categorical_features = CATEGORICAL_FEATURES
        self.id_column = ID_COLUMN
        self.date_column = DATE_COLUMN
        self.logger.info("FeatureEngineering initialized")

    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features from datetime column

        Args:
            df: Input DataFrame with datetime column

        Returns:
            DataFrame with added temporal features
        """
        try:
            self.logger.info("Extracting temporal features...")
            df = df.copy()

            if self.date_column not in df.columns:
                raise FeatureEngineeringError(f"Date column '{self.date_column}' not found")

            df[self.date_column] = pd.to_datetime(df[self.date_column], errors='coerce')
            if df[self.date_column].isnull().any():
                self.logger.warning("Some date values could not be parsed")

            df["hour"] = df[self.date_column].dt.hour
            df["day"] = df[self.date_column].dt.day
            df["weekday"] = df[self.date_column].dt.weekday
            df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
            # additional temporal flags
            df["is_night"] = ((df["hour"] >= 0) & (df["hour"] <= 5)).astype(int)

            df = df.drop(columns=[self.date_column])
            self.logger.info("✅ Temporal features extracted")
            return df

        except Exception as e:
            self.logger.error(f"Error extracting temporal features: {str(e)}")
            raise FeatureEngineeringError(f"Temporal feature extraction failed: {str(e)}") from e

    def apply_one_hot_encoding(self, df: pd.DataFrame, drop_first: bool = True) -> pd.DataFrame:
        """
        Apply one-hot encoding to categorical columns

        Args:
            df: Input DataFrame
            drop_first: Whether to drop first category to avoid multicollinearity

        Returns:
            DataFrame with one-hot encoded categorical variables
        """
        try:
            self.logger.info(f"Applying one-hot encoding to {self.categorical_features}...")
            df = df.copy()
            df = pd.get_dummies(df, columns=self.categorical_features, drop_first=drop_first)
            self.logger.info(f"✅ One-hot encoding applied. New shape: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Error in one-hot encoding: {str(e)}")
            raise FeatureEngineeringError(f"One-hot encoding failed: {str(e)}") from e

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete feature preparation pipeline

        Args:
            df: Raw input DataFrame

        Returns:
            Processed DataFrame ready for modeling
        """
        try:
            self.logger.info("Starting feature preparation pipeline...")

            # Drop ID column if exists
            if self.id_column in df.columns:
                df = df.drop(columns=[self.id_column])
                self.logger.info(f"✅ Dropped {self.id_column} column")

            # Extract temporal features
            df = self.extract_temporal_features(df)

            # Aggregated merchant/location features (computed on available df)
            if "MerchantID" in df.columns:
                df["merchant_fraud_rate"] = df.groupby("MerchantID")["IsFraud"].transform("mean")
                df["amount_vs_merchant_mean"] = df["Amount"] / df.groupby("MerchantID")["Amount"].transform("mean")
                df["merchant_txn_count"] = df.groupby("MerchantID")["MerchantID"].transform("count")
                df["merchant_fraud_rate"] = df["merchant_fraud_rate"].fillna(0)
                df["amount_vs_merchant_mean"] = df["amount_vs_merchant_mean"].fillna(0)
                df["merchant_txn_count"] = df["merchant_txn_count"].fillna(0)

            if "Location" in df.columns:
                df["location_fraud_rate"] = df.groupby("Location")["IsFraud"].transform("mean")
                df["location_fraud_rate"] = df["location_fraud_rate"].fillna(0)

            # high amount flag
            df["high_amount"] = (df["Amount"] > df["Amount"].quantile(0.95)).astype(int)

            # Apply one-hot encoding (keep all categories to align with saved feature columns)
            df = self.apply_one_hot_encoding(df, drop_first=False)

            self.logger.info(f"✅ Feature preparation completed. Final shape: {df.shape}")
            return df

        except Exception as e:
            self.logger.error(f"Feature preparation failed: {str(e)}")
            raise FeatureEngineeringError(f"Feature preparation failed: {str(e)}") from e

    def prepare_user_input(
        self,
        amount: float,
        merchant_id: int,
        transaction_type: str,
        location: str,
        transaction_time: datetime,
        feature_columns: List[str],
        reference_day: int = None
    ) -> pd.DataFrame:
        """
        Prepare user input from Streamlit app for prediction

        Args:
            amount: Transaction amount
            merchant_id: Merchant ID
            transaction_type: Type of transaction (purchase/refund)
            location: Transaction location
            transaction_time: Time of transaction
            feature_columns: List of expected feature columns
            reference_day: Day for feature extraction (default: today)

        Returns:
            DataFrame prepared for model prediction
        """
        if reference_day is None:
            now = datetime.now()
        else:
            now = datetime(datetime.now().year, datetime.now().month, reference_day)

        hour = transaction_time.hour
        day = now.day
        weekday = now.weekday()
        is_weekend = int(weekday in [5, 6])

        # Base input dictionary
        input_dict = {
            "Amount": amount,
            "MerchantID": merchant_id,
            "hour": hour,
            "day": day,
            "weekday": weekday,
            "is_weekend": is_weekend
        }

        # Initialize all expected columns with 0
        input_df = pd.DataFrame(
            np.zeros((1, len(feature_columns))),
            columns=feature_columns
        )

        # Fill numeric values
        for col in input_dict:
            if col in input_df.columns:
                input_df[col] = input_dict[col]

        # One-hot encoding for TransactionType
        if transaction_type == "refund":
            if "TransactionType_refund" in input_df.columns:
                input_df["TransactionType_refund"] = 1

        # One-hot encoding for Location
        loc_col = f"Location_{location}"
        if loc_col in input_df.columns:
            input_df[loc_col] = 1

        return input_df