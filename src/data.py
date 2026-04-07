"""
Download and clean the IBM Telco Customer Churn dataset.
"""

import pathlib
import requests
import pandas as pd

RAW_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
RAW_PATH = pathlib.Path("data/raw/Telco-Customer-Churn.csv")
CLEAN_PATH = pathlib.Path("data/processed/telco_churn_clean.csv")

# Columns that encode "No internet service" or "No phone service" as a third value
_MULTI_NO_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "MultipleLines",
]


def download_raw(url: str = RAW_URL, dest: pathlib.Path = RAW_PATH) -> pathlib.Path:
    """Download raw CSV if it doesn't already exist."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        dest.write_bytes(response.content)
    return dest


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning steps and return a cleaned DataFrame."""
    df = df.copy()

    # 1. TotalCharges: blank strings → 0, cast to float
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    # 2. Drop customerID (not a feature)
    df = df.drop(columns=["customerID"])

    # 3. Churn: Yes/No → 1/0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # 4. Collapse "No internet service" / "No phone service" → "No"
    for col in _MULTI_NO_COLS:
        if col in df.columns:
            df[col] = df[col].replace(
                {"No internet service": "No", "No phone service": "No"}
            )

    # 5. SeniorCitizen: 0/1 → No/Yes
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    return df


def load_clean(force_download: bool = False) -> pd.DataFrame:
    """Return cleaned DataFrame, downloading + cleaning if necessary."""
    if not CLEAN_PATH.exists() or force_download:
        raw_path = download_raw()
        df_raw = pd.read_csv(raw_path)
        df_clean = clean(df_raw)
        CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(CLEAN_PATH, index=False)
    else:
        df_clean = pd.read_csv(CLEAN_PATH)
    return df_clean


if __name__ == "__main__":
    df = load_clean(force_download=True)
    churn_rate = df["Churn"].mean() * 100
    print(f"Shape: {df.shape}")
    print(f"Churn rate: {churn_rate:.1f}%")
