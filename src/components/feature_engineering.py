"""
Feature Engineering Component
Handles feature extraction and preparation for credit card fraud detection
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List


class FeatureEngineering:
    """
    Component for engineering features from raw credit card transaction data
    """

    def __init__(self):
        """Initialize feature engineering component"""
        self.numeric_features = ["Amount", "MerchantID", "hour", "day", "weekday"]
        self.categorical_features = ["TransactionType", "Location"]
        self.target_column = "IsFraud"
        self.id_column = "TransactionID"
        self.date_column = "TransactionDate"

    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features from datetime column

        Args:
            df: Input DataFrame with TransactionDate column

        Returns:
            DataFrame with added temporal features
        """
        df = df.copy()
        df[self.date_column] = pd.to_datetime(df[self.date_column])

        df["hour"] = df[self.date_column].dt.hour
        df["day"] = df[self.date_column].dt.day
        df["weekday"] = df[self.date_column].dt.weekday
        df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

        df = df.drop(columns=[self.date_column])

        print("✅ Temporal features extracted")
        return df

    def apply_one_hot_encoding(self, df: pd.DataFrame, drop_first: bool = True) -> pd.DataFrame:
        """
        Apply one-hot encoding to categorical columns

        Args:
            df: Input DataFrame
            drop_first: Whether to drop first category to avoid multicollinearity

        Returns:
            DataFrame with one-hot encoded categorical variables
        """
        df = df.copy()
        df = pd.get_dummies(df, columns=self.categorical_features, drop_first=drop_first)

        print(f"✅ One-hot encoding applied. New shape: {df.shape}")
        return df

    def prepare_features(self, df: pd.DataFrame, drop_id: bool = True) -> pd.DataFrame:
        """
        Complete feature preparation pipeline

        Args:
            df: Raw DataFrame
            drop_id: Whether to drop ID column

        Returns:
            Processed DataFrame with engineered features
        """
        df = df.copy()

        # Drop ID column if requested
        if drop_id and self.id_column in df.columns:
            df = df.drop(columns=[self.id_column])
            print(f"✅ Dropped {self.id_column} column")

        # Extract temporal features
        df = self.extract_temporal_features(df)

        # Apply one-hot encoding
        df = self.apply_one_hot_encoding(df)

        print(f"✅ Feature preparation completed. Final shape: {df.shape}")
        return df

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