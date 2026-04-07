# Telco Customer Churn Prediction

A machine learning project to predict customer churn using the IBM Telco Customer Churn dataset.

## Problem Statement

Predict whether a telecom customer will cancel their subscription based on their demographics, services, and billing information. Early identification of at-risk customers allows the business to intervene with retention offers before losing them.

## Dataset

- **Source:** IBM Telco Customer Churn dataset
- **Size:** 7,043 customers, 20 features
- **Target:** Churn (1 = left, 0 = stayed)
- **Class balance:** ~26.5% churn rate

## Project Structure

```
churn-prediction/
├── data/
│   ├── raw/                  # Original downloaded CSV
│   └── processed/            # Cleaned CSV ready for modeling
├── notebooks/                # Jupyter notebooks for EDA and experiments
├── src/
│   ├── data.py               # Data download and cleaning pipeline
│   └── features.py           # Feature engineering and preprocessing
├── tests/                    # Unit tests
├── requirements.txt
└── README.md
```

## Setup

```bash
# Clone the repo
git clone https://github.com/Zitsuu/churn-prediction.git
cd churn-prediction

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Download and clean the dataset
python src/data.py
# Output: Shape: (7043, 20) | Churn rate: 26.5%

# Run feature engineering pipeline
python src/features.py
# Output: Transformed train/test shapes
```

## Tech Stack

- **Data:** pandas, numpy
- **ML:** scikit-learn, xgboost, lightgbm
- **Tuning:** optuna
- **Explainability:** shap
- **Tracking:** mlflow
- **App:** streamlit
- **Testing:** pytest
