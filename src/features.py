"""
Feature engineering and preprocessing pipeline for Telco churn data.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import load_clean

# ------------------------------------------------------------------
# Column definitions (after cleaning)
# ------------------------------------------------------------------
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
TARGET = "Churn"


# ------------------------------------------------------------------
# Engineered features
# ------------------------------------------------------------------
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived columns:
      - AvgChargesPerMonth: TotalCharges / tenure (0-safe)
      - TenureGroup: binned tenure (0-1yr, 1-2yr, 2-4yr, 4-6yr)
    Returns a new DataFrame with the extra columns appended.
    """
    df = df.copy()

    # AvgChargesPerMonth — guard against division by zero (tenure == 0)
    df["AvgChargesPerMonth"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"],  # fallback: use current monthly rate
    )

    # TenureGroup — closed-right bins
    bins = [0, 12, 24, 48, 72]
    labels = ["0-1yr", "1-2yr", "2-4yr", "4-6yr"]
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=True,
    ).astype(str)

    return df


# ------------------------------------------------------------------
# Preprocessor
# ------------------------------------------------------------------
def build_preprocessor(
    extra_numeric: list[str] | None = None,
    extra_categorical: list[str] | None = None,
) -> ColumnTransformer:
    """
    Return a ColumnTransformer with:
      - StandardScaler  on numeric columns
      - OneHotEncoder   on categorical columns (unknown → ignore)

    Pass extra_numeric / extra_categorical to include engineered features.
    """
    numeric = NUMERIC_COLS + (extra_numeric or [])
    categorical = CATEGORICAL_COLS + (extra_categorical or [])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


# ------------------------------------------------------------------
# Train / test split
# ------------------------------------------------------------------
def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified 80/20 split.
    Returns X_train, X_test, y_train, y_test.
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------
if __name__ == "__main__":
    df = load_clean()
    df = add_engineered_features(df)

    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train size : {X_train.shape[0]:,}  |  Test size: {X_test.shape[0]:,}")
    print(f"New columns: AvgChargesPerMonth, TenureGroup")

    preprocessor = build_preprocessor(
        extra_numeric=["AvgChargesPerMonth"],
        extra_categorical=["TenureGroup"],
    )
    X_train_t = preprocessor.fit_transform(X_train, y_train)
    X_test_t = preprocessor.transform(X_test)
    print(f"Transformed train shape: {X_train_t.shape}")
    print(f"Transformed test shape : {X_test_t.shape}")
    print("Preprocessing pipeline OK")
