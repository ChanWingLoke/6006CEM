"""
train_models.py
---------------
Step 3: Train, evaluate, and tune two regression models:
1. Random Forest Regressor (Traditional ML)
2. MLP Regressor (Deep Learning)

Inputs (from Step 2, in --results_dir):
- X_train.npz / X_valid.npz / X_test.npz
- y_train_log.npy / y_valid_log.npy / y_test_log.npy
- feature_names.json (optional, for feature importances)
- preprocessor.joblib (optional, for inference)

Outputs:
- results/metrics_comparison.csv
- results/model_rf.joblib
- results/model_mlp.joblib
- results/feature_importance_rf.csv
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, KFold

# ---------------------------
# Helper functions
# ---------------------------

def load_matrix(path: Path):
    """
    Robust loader for both sparse (.npz) and dense (.npy) matrices.
    """
    path = Path(path)
    candidates = []
    if path.suffix:
        candidates.append(path)
    candidates.extend([path.with_suffix(".npz"), path.with_suffix(".npy")])

    for p in candidates:
        if not p.exists():
            continue
        try:
            return sparse.load_npz(p)
        except Exception:
            try:
                return np.load(p, allow_pickle=False)
            except Exception:
                pass
    raise FileNotFoundError(f"Could not load matrix for {path}")


def to_dense_if_needed(X, model_type="rf"):
    """
    RandomForest and MLP require dense arrays.
    """
    if sparse.issparse(X):
        return X.toarray()
    return X


def eval_regression(y_true_log, y_pred_log):
    """
    Evaluate models on both log-space (R²) and raw prices (MAE/RMSE).
    """
    r2_log = r2_score(y_true_log, y_pred_log)
    y_true_raw = np.expm1(y_true_log)
    y_pred_raw = np.expm1(y_pred_log)

    mae = mean_absolute_error(y_true_raw, y_pred_raw)
    try:
        rmse = mean_squared_error(y_true_raw, y_pred_raw, squared=False)
    except TypeError:
        rmse = np.sqrt(mean_squared_error(y_true_raw, y_pred_raw))
    return {"R2_log": r2_log, "MAE_raw": mae, "RMSE_raw": rmse}


def save_feature_importance(model, feature_names, out_csv: Path):
    """
    Save feature importances for tree-based models.
    """
    if hasattr(model, "feature_importances_"):
        imp = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        imp.to_csv(out_csv, index=False)

# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Train & evaluate RF and MLP regressors.")
    parser.add_argument("--results_dir", default="results", help="Directory for input/output artifacts")
    parser.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning")
    parser.add_argument("--cv_splits", type=int, default=5)
    parser.add_argument("--n_iter", type=int, default=40)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    res = Path(args.results_dir)
    res.mkdir(parents=True, exist_ok=True)

    # -----------------------
    # Load data
    # -----------------------
    X_train = load_matrix(res / "X_train")
    X_valid = load_matrix(res / "X_valid")
    X_test = load_matrix(res / "X_test")

    y_train_log = np.load(res / "y_train_log.npy")
    y_valid_log = np.load(res / "y_valid_log.npy")
    y_test_log  = np.load(res / "y_test_log.npy")

    # Load feature names (optional)
    feature_names = []
    fn_path = res / "feature_names.json"
    if fn_path.exists():
        feature_names = json.loads(fn_path.read_text())

    metrics_rows = []

    # ======================================================
    # 1) Random Forest (Traditional ML)
    # ======================================================
    print("Training Random Forest Regressor...")
    rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        n_jobs=args.n_jobs,
        random_state=args.random_state,
        bootstrap=True,
    )

    rf.fit(to_dense_if_needed(X_train, "rf"), y_train_log)

    pred_val_rf = rf.predict(to_dense_if_needed(X_valid, "rf"))
    pred_tst_rf = rf.predict(to_dense_if_needed(X_test, "rf"))

    m_val = eval_regression(y_valid_log, pred_val_rf)
    m_tst = eval_regression(y_test_log, pred_tst_rf)

    joblib.dump(rf, res / "model_rf.joblib")
    metrics_rows.append({"Model": "RandomForest", "Split": "valid", **m_val})
    metrics_rows.append({"Model": "RandomForest", "Split": "test",  **m_tst})

    if feature_names:
        save_feature_importance(rf, feature_names, res / "feature_importance_rf.csv")

    # ======================================================
    # 2) MLP Regressor (Deep Learning)
    # ======================================================
    print("Training MLP Regressor (Deep Learning)...")
    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=0.001,
        learning_rate="adaptive",
        max_iter=300,
        early_stopping=True,
        random_state=args.random_state,
    )

    mlp.fit(to_dense_if_needed(X_train, "mlp"), y_train_log)

    pred_val_mlp = mlp.predict(to_dense_if_needed(X_valid, "mlp"))
    pred_tst_mlp = mlp.predict(to_dense_if_needed(X_test, "mlp"))

    m_val = eval_regression(y_valid_log, pred_val_mlp)
    m_tst = eval_regression(y_test_log, pred_tst_mlp)

    joblib.dump(mlp, res / "model_mlp.joblib")
    metrics_rows.append({"Model": "MLPRegressor", "Split": "valid", **m_val})
    metrics_rows.append({"Model": "MLPRegressor", "Split": "test",  **m_tst})

    # ======================================================
    # Optional: Hyperparameter Tuning
    # ======================================================
    if args.tune:
        print(">>> Running RandomizedSearchCV for RandomForest and MLP...")

        cv = KFold(n_splits=args.cv_splits, shuffle=True, random_state=args.random_state)

        # ---- Random Forest tuning
        rf_param_grid = {
            "n_estimators": [200, 400, 600, 800],
            "max_depth": [None, 10, 15, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 0.5, None],
        }

        rf_search = RandomizedSearchCV(
            rf,
            rf_param_grid,
            n_iter=args.n_iter,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=args.n_jobs,
            random_state=args.random_state,
            verbose=1,
        )

        rf_search.fit(to_dense_if_needed(X_train, "rf"), y_train_log)
        rf_best = rf_search.best_estimator_

        pred_tst_rf_tuned = rf_best.predict(to_dense_if_needed(X_test, "rf"))
        m_tst_rf_tuned = eval_regression(y_test_log, pred_tst_rf_tuned)
        metrics_rows.append({"Model": "RandomForest_TUNED", "Split": "test", **m_tst_rf_tuned})
        joblib.dump(rf_best, res / "model_rf_tuned.joblib")

        # ---- MLP tuning
        mlp_param_grid = {
            "hidden_layer_sizes": [(64, 32), (128, 64, 32), (256, 128)],
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate_init": [0.001, 0.01],
        }

        mlp_search = RandomizedSearchCV(
            mlp,
            mlp_param_grid,
            n_iter=args.n_iter,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=args.n_jobs,
            random_state=args.random_state,
            verbose=1,
        )

        mlp_search.fit(to_dense_if_needed(X_train, "mlp"), y_train_log)
        mlp_best = mlp_search.best_estimator_

        pred_tst_mlp_tuned = mlp_best.predict(to_dense_if_needed(X_test, "mlp"))
        m_tst_mlp_tuned = eval_regression(y_test_log, pred_tst_mlp_tuned)
        metrics_rows.append({"Model": "MLPRegressor_TUNED", "Split": "test", **m_tst_mlp_tuned})
        joblib.dump(mlp_best, res / "model_mlp_tuned.joblib")

    # -----------------------
    # Save metrics
    # -----------------------
    df_metrics = pd.DataFrame(metrics_rows)
    df_metrics.to_csv(res / "metrics_comparison.csv", index=False)

    print("\n=== Model Performance Summary ===")
    print(df_metrics)
    print(f"\nMetrics saved to {res / 'metrics_comparison.csv'}")
    print("Done.")


if __name__ == "__main__":
    main()
