import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

TARGET_COL = "Churn"

def load_raw(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df

def clean_basic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Standardize TotalCharges: convert to numeric; coerce errors -> NaN then impute later
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Strip whitespace from object columns
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip()

    # Drop duplicate rows if any
    df = df.drop_duplicates()

    # Handle missing: for numeric, use median; for object, use most frequent
    for c in df.columns:
        if df[c].dtype.kind in "biufc":
            df[c] = df[c].fillna(df[c].median())
        else:
            df[c] = df[c].fillna(df[c].mode(dropna=True).iloc[0] if not df[c].mode(dropna=True).empty else "")

    return df

def split_Xy(df: pd.DataFrame):
    if TARGET_COL not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COL}' not found.")
    y = df[TARGET_COL].map({"No": 0, "Yes": 1}).astype(int)
    X = df.drop(columns=[TARGET_COL, "customerID"], errors="ignore")
    return X, y

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor

def make_train_test(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
