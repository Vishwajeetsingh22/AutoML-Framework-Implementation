"""
data_preprocessing.py
----------------------
Generic, dataset-agnostic preprocessing pipeline used by the AutoML
framework. Given a dataset config (path, target column, columns to
drop, columns where 0 actually means "missing"), it:

  1. Loads the CSV
  2. Drops irrelevant identifier columns (e.g. Loan_ID)
  3. Replaces implausible zeros with NaN where configured
  4. Imputes missing values (median for numeric, mode for categorical)
  5. One-hot encodes categorical features
  6. Label-encodes the target if it isn't already numeric
  7. Splits into train/test sets (stratified)
  8. Scales numeric features with StandardScaler

This lets the same AutoML engine (src/automl.py) run unmodified on the
Diabetes, Iris, and Loan datasets (or any other similarly-shaped
tabular classification dataset).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ---------------------------------------------------------------------------
# Dataset configuration: add a new entry here to plug in a new dataset
# ---------------------------------------------------------------------------
DATASET_CONFIGS = {
    "diabetes": {
        "path": "../data/diabetes.csv",
        "target": "Outcome",
        "drop_cols": [],
        # These columns use 0 to mean "missing" (not physiologically valid)
        "zero_as_missing": ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"],
    },
    "iris": {
        "path": "../data/iris.csv",
        "target": "species",
        "drop_cols": [],
        "zero_as_missing": [],
    },
    "loan": {
        "path": "../data/loan.csv",
        "target": "Loan_Status",
        "drop_cols": ["Loan_ID"],
        "zero_as_missing": [],
    },
}


def load_dataset(name: str) -> pd.DataFrame:
    config = DATASET_CONFIGS[name]
    df = pd.read_csv(config["path"])
    return df


def clean_and_impute(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Drop irrelevant columns, mark implausible zeros as missing, impute."""
    df = df.copy()

    for col in config.get("drop_cols", []):
        if col in df.columns:
            df = df.drop(columns=[col])

    for col in config.get("zero_as_missing", []):
        df[col] = df[col].replace(0, np.nan)

    for col in df.columns:
        if col == config["target"]:
            continue
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0])

    return df


def encode_features(df: pd.DataFrame, target: str):
    """One-hot encode categorical predictors; label-encode a non-numeric target."""
    X = df.drop(columns=[target])
    y = df[target]

    categorical_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    label_encoder = None
    if not pd.api.types.is_numeric_dtype(y):
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y), index=y.index, name=target)

    return X, y, label_encoder


def preprocess(dataset_name: str, test_size: float = 0.2, random_state: int = 42):
    """
    Full preprocessing pipeline for a named dataset in DATASET_CONFIGS.

    Returns
    -------
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names, label_encoder
    """
    config = DATASET_CONFIGS[dataset_name]
    df = load_dataset(dataset_name)
    df = clean_and_impute(df, config)
    X, y, label_encoder = encode_features(df, config["target"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, list(X.columns), label_encoder


if __name__ == "__main__":
    for name in DATASET_CONFIGS:
        X_train, X_test, y_train, y_test, scaler, feature_names, le = preprocess(name)
        print(f"[{name}] train={X_train.shape}, test={X_test.shape}, "
              f"features={len(feature_names)}, classes={sorted(set(y_train) | set(y_test))}")
