"""
preprocessing.py
----------------
Data preprocessing script for Ames Housing price regression.

This script:
- Cleans the dataset (handles duplicates, missing values, and outliers)
- Encodes categorical and numeric features
- Scales numerical features
- Performs log-transformation on the target to handle imbalance
- Splits data into train/validation/test sets with stratification
- Saves processed data and preprocessing pipeline for later model training
"""

import argparse
import json
import os
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from scipy import sparse
import joblib


# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------

def safe_mkdir(path: str):
    """Safely create a directory if it doesn't already exist."""
    os.makedirs(path, exist_ok=True)


def cap_outliers_iqr(df: pd.DataFrame, numeric_cols: List[str], cap_factor: float = 1.5) -> pd.DataFrame:
    """
    Caps numeric columns to the IQR range [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    Prevents extreme values from skewing models.
    """
    capped = df.copy()
    for col in numeric_cols:
        s = capped[col]
        if s.dtype.kind not in "biufc":
            continue
        if s.notna().sum() == 0:
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0 or np.isnan(iqr):
            continue
        lo = q1 - cap_factor * iqr
        hi = q3 + cap_factor * iqr
        capped[col] = s.clip(lower=lo, upper=hi)
    return capped


def stratify_bins(y: pd.Series, n_bins: int = 10) -> np.ndarray:
    """Create stratification bins from continuous target (y)."""
    try:
        return pd.qcut(y, q=n_bins, labels=False, duplicates="drop").astype(int).values
    except ValueError:
        # Fallback if too many duplicate values
        ranks = y.rank(method="first")
        return pd.qcut(ranks, q=n_bins, labels=False, duplicates="drop").astype(int).values


def split_data(
    X: pd.DataFrame,
    y_log: pd.Series,
    y_raw: pd.Series,
    test_size: float = 0.15,
    valid_size: float = 0.15,
    random_state: int = 42
):
    """
    Splits dataset into Train / Validation / Test sets.
    Uses stratified splitting on y_log to maintain balance in price ranges.
    """
    y_bins = stratify_bins(y_log, n_bins=10)

    # First split off test set
    X_train_valid, X_test, y_log_train_valid, y_log_test, y_raw_train_valid, y_raw_test = train_test_split(
        X, y_log, y_raw, test_size=test_size, random_state=random_state, stratify=y_bins
    )

    # Now split remaining into train and validation
    bins_train_valid = stratify_bins(y_log_train_valid, n_bins=10)
    valid_fraction_of_trainvalid = valid_size / (1.0 - test_size)
    X_train, X_valid, y_log_train, y_log_valid, y_raw_train, y_raw_valid = train_test_split(
        X_train_valid, y_log_train_valid, y_raw_train_valid,
        test_size=valid_fraction_of_trainvalid,
        random_state=random_state,
        stratify=bins_train_valid
    )

    return (X_train, X_valid, X_test,
            y_log_train, y_log_valid, y_log_test,
            y_raw_train, y_raw_valid, y_raw_test)


def make_pipeline_named(*steps):
    """Creates a named pipeline (helps avoid name collisions)."""
    named_steps = []
    for i, step in enumerate(steps):
        name = step.__class__.__name__.lower()
        name = f"{name}_{i}"
        named_steps.append((name, step))
    return Pipeline(named_steps)


def build_preprocessor(X_train: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Builds a ColumnTransformer that:
      - Imputes and scales numeric features
      - Imputes and encodes categorical features

    Compatible with both older and newer versions of scikit-learn.
    """
    numeric_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    numeric_pipe = SimpleImputer(strategy="median")
    numeric_scaler = StandardScaler(with_mean=True, with_std=True)
    cat_pipe = SimpleImputer(strategy="most_frequent")

    # Compatibility: try 'sparse_output' (new sklearn) or fallback to 'sparse' (old sklearn)
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True, min_frequency=0.01)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True, min_frequency=0.01)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", make_pipeline_named(numeric_pipe, numeric_scaler), numeric_cols),
            ("cat", make_pipeline_named(cat_pipe, ohe), categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return preprocessor, numeric_cols, categorical_cols


def get_feature_names(preprocessor: ColumnTransformer,
                      numeric_cols: List[str],
                      categorical_cols: List[str]) -> List[str]:
    """Extracts feature names from the fitted ColumnTransformer."""
    feature_names = []
    feature_names.extend(numeric_cols)

    # Extract one-hot encoded categorical feature names
    ohe = None
    for name, trans, cols in preprocessor.transformers_:
        if name == "cat":
            ohe = trans.named_steps[list(trans.named_steps.keys())[-1]]
            cat_cols = cols
            break

    if ohe is not None:
        try:
            ohe_feature_names = ohe.get_feature_names_out(cat_cols).tolist()
        except TypeError:
            ohe_feature_names = []
            for i, col in enumerate(cat_cols):
                cats = ohe.categories_[i]
                ohe_feature_names += [f"{col}_{c}" for c in cats]
        feature_names.extend(ohe_feature_names)

    return feature_names


# -------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess Ames Housing dataset for regression models.")
    parser.add_argument("--data", default="data/ames.csv", help="Path to the raw CSV dataset")
    parser.add_argument("--results_dir", default="results", help="Directory to save outputs")
    parser.add_argument("--cap_outliers", action="store_true", help="Apply IQR capping to numeric features")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    safe_mkdir(args.results_dir)

    # --------------------
    # 1️⃣ Load and inspect data
    # --------------------
    df = pd.read_csv(args.data)
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])

    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {before - len(df)} duplicate rows.")

    # Ensure target exists
    if "SalePrice" not in df.columns:
        raise ValueError("Expected target column 'SalePrice' not found in dataset.")

    # Remove rows with missing or invalid SalePrice
    df = df[df["SalePrice"].notna()]
    df = df[df["SalePrice"] > 0]

    # --------------------
    # 2️⃣ Handle predictors & target
    # --------------------
    X_all = df.drop(columns=["SalePrice"])
    y_raw = df["SalePrice"].astype(float)

    numeric_cols_all = [c for c in X_all.columns if pd.api.types.is_numeric_dtype(X_all[c])]

    # Optional outlier capping
    if args.cap_outliers and len(numeric_cols_all) > 0:
        X_all = cap_outliers_iqr(X_all, numeric_cols_all)

    # Log-transform target to fix right skew (common in price data)
    y_log = np.log1p(y_raw)

    # --------------------
    # 3️⃣ Split data (train/valid/test)
    # --------------------
    (X_train, X_valid, X_test,
     y_log_train, y_log_valid, y_log_test,
     y_raw_train, y_raw_valid, y_raw_test) = split_data(
        X_all, pd.Series(y_log), pd.Series(y_raw),
        test_size=0.15, valid_size=0.15, random_state=args.random_state
    )

    # --------------------
    # 4️⃣ Build & fit preprocessor
    # --------------------
    preprocessor, num_cols, cat_cols = build_preprocessor(X_train)
    preprocessor.fit(X_train)

    # --------------------
    # 5️⃣ Transform data
    # --------------------
    X_train_t = preprocessor.transform(X_train)
    X_valid_t = preprocessor.transform(X_valid)
    X_test_t = preprocessor.transform(X_test)

    # --------------------
    # 6️⃣ Save processed artifacts
    # --------------------
    def save_matrix(path, mat):
        if sparse.issparse(mat):
            sparse.save_npz(path, mat.tocsr())
        else:
            np.save(path, mat)

    # Save feature matrices
    save_matrix(os.path.join(args.results_dir, "X_train.npz"), X_train_t)
    save_matrix(os.path.join(args.results_dir, "X_valid.npz"), X_valid_t)
    save_matrix(os.path.join(args.results_dir, "X_test.npz"), X_test_t)

    # Save target arrays
    np.save(os.path.join(args.results_dir, "y_train_log.npy"), y_log_train.values)
    np.save(os.path.join(args.results_dir, "y_valid_log.npy"), y_log_valid.values)
    np.save(os.path.join(args.results_dir, "y_test_log.npy"), y_log_test.values)

    np.save(os.path.join(args.results_dir, "y_train_raw.npy"), y_raw_train.values)
    np.save(os.path.join(args.results_dir, "y_valid_raw.npy"), y_raw_valid.values)
    np.save(os.path.join(args.results_dir, "y_test_raw.npy"), y_raw_test.values)

    # Save feature names and preprocessor
    feature_names = get_feature_names(preprocessor, num_cols, cat_cols)
    with open(os.path.join(args.results_dir, "feature_names.json"), "w") as f:
        json.dump(feature_names, f, indent=2)

    joblib.dump(preprocessor, os.path.join(args.results_dir, "preprocessor.joblib"))

    # Optional: save readable cleaned CSVs for debugging
    X_train.assign(SalePrice=y_raw_train).to_csv(os.path.join(args.results_dir, "cleaned_train.csv"), index=False)
    X_valid.assign(SalePrice=y_raw_valid).to_csv(os.path.join(args.results_dir, "cleaned_valid.csv"), index=False)
    X_test.assign(SalePrice=y_raw_test).to_csv(os.path.join(args.results_dir, "cleaned_test.csv"), index=False)

    # --------------------
    # ✅ Final Summary
    # --------------------
    print("=== Preprocessing Completed ===")
    print(f"Train shape: {X_train.shape}")
    print(f"Valid shape: {X_valid.shape}")
    print(f"Test shape : {X_test.shape}")
    print(f"Numeric cols: {len(num_cols)} | Categorical cols: {len(cat_cols)}")
    print(f"Saved results to: {args.results_dir}")


if __name__ == "__main__":
    main()
