"""
preprocessing.py
----------------
Performs all data preparation steps before model training:
- Loads the Ames housing dataset
- Cleans missing and invalid values
- Optionally caps predictor outliers
- Encodes categorical features, scales numerics
- Splits into train/validation/test sets
- Saves processed datasets and preprocessing pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import json
import argparse

# ------------------------- Utility functions -------------------------

def cap_outliers(df):
    """
    Cap numeric columns using the IQR rule:
    Values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] are clipped to the boundary.
    Prevents extreme outliers from skewing model training.
    """
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower, upper)
    return df


def build_preprocessor(X_train):
    """
    Construct a ColumnTransformer that:
      - Imputes and scales numeric columns
      - Imputes and one-hot encodes categorical columns
    Returns both the transformer and column name lists.
    """
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True, min_frequency=0.01)
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", ohe)
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    return preprocessor, num_cols, cat_cols


# ------------------------- Main pipeline -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--cap_outliers", action="store_true")
    args = parser.parse_args()

    data_path = Path(args.data)
    res = Path(args.results_dir)
    res.mkdir(exist_ok=True)

    # 1️⃣ Load data
    df = pd.read_csv(data_path)
    print(f"Loaded dataset with {len(df)} rows and {df.shape[1]} columns.")

    # 2️⃣ Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {before - len(df)} duplicate rows.")

    # 3️⃣ Remove invalid target rows
    before = len(df)
    df = df[df["SalePrice"].notna() & (df["SalePrice"] > 0)]
    print(f"Removed {before - len(df)} rows with missing/invalid SalePrice.")

    # 4️⃣ Separate predictors and target
    X_all = df.drop("SalePrice", axis=1)
    y_raw = df["SalePrice"].astype(float)
    y_log = np.log1p(y_raw)

    # 5️⃣ Optional outlier capping
    if args.cap_outliers:
        X_all = cap_outliers(X_all)
        print("Capped predictor outliers using IQR method.")

    # 6️⃣ Stratified splitting by target bins (to preserve target distribution)
    y_bins = pd.qcut(y_log, q=10, duplicates="drop")
    X_temp, X_test, y_temp, y_test = train_test_split(X_all, y_log, test_size=0.2, stratify=y_bins, random_state=42)
    y_bins_temp = pd.qcut(y_temp, q=10, duplicates="drop")
    X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.2, stratify=y_bins_temp, random_state=42)

    # Save raw targets for metric comparison later
    y_train_raw = np.expm1(y_train)
    y_valid_raw = np.expm1(y_valid)
    y_test_raw = np.expm1(y_test)

    # 7️⃣ Build preprocessing pipeline
    preprocessor, num_cols, cat_cols = build_preprocessor(X_train)
    preprocessor.fit(X_train)
    print("Preprocessing pipeline fitted on training data.")

    # 8️⃣ Transform splits
    Xt_train = preprocessor.transform(X_train)
    Xt_valid = preprocessor.transform(X_valid)
    Xt_test = preprocessor.transform(X_test)

    # 9️⃣ Save all results
    def save_array(path, arr):
        if hasattr(arr, "toarray"):
            np.savez_compressed(path, arr)
        else:
            np.save(path, arr)

    save_array(res / "X_train", Xt_train)
    save_array(res / "X_valid", Xt_valid)
    save_array(res / "X_test", Xt_test)

    np.save(res / "y_train_log.npy", y_train)
    np.save(res / "y_valid_log.npy", y_valid)
    np.save(res / "y_test_log.npy", y_test)
    np.save(res / "y_train_raw.npy", y_train_raw)
    np.save(res / "y_valid_raw.npy", y_valid_raw)
    np.save(res / "y_test_raw.npy", y_test_raw)

    # Save fitted preprocessor and feature names
    joblib.dump(preprocessor, res / "preprocessor.joblib")
    feature_names = num_cols + preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out(cat_cols).tolist()
    (res / "feature_names.json").write_text(json.dumps(feature_names, indent=2))

    print(f"Saved all processed data and artifacts to {res.resolve()}")

if __name__ == "__main__":
    main()
