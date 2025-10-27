"""
train_models.py
---------------
Trains and evaluates regression models for the Ames Housing dataset.

Models used:
  - Random Forest (traditional ML)
  - MLP Regressor (deep learning)

Features:
  - Supports baseline training and hyperparameter tuning (--tune)
  - Cross-validation with RandomizedSearchCV
  - Saves models, best parameters, and evaluation metrics
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import argparse

# ------------------ Helper Functions ------------------

def load_matrix(path):
    """Load .npy or .npz (sparse) matrix safely."""
    path = Path(path)
    if path.with_suffix(".npy").exists():
        return np.load(path.with_suffix(".npy"), allow_pickle=True)
    if path.with_suffix(".npz").exists():
        from scipy import sparse
        return sparse.load_npz(path.with_suffix(".npz"))
    raise FileNotFoundError(f"Could not load matrix for {path}")

def to_dense_if_needed(X, tag=""):
    """Convert sparse matrices to dense if model requires it."""
    from scipy import sparse
    if sparse.issparse(X):
        print(f"[{tag}] converting sparse -> dense for model compatibility")
        return X.toarray()
    return X

def eval_regression(y_true_log, y_pred_log):
    """
    Return R2 on log target, and MAE/RMSE in raw dollars.
    """
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    y_true_raw = np.expm1(y_true_log)
    y_pred_raw = np.expm1(y_pred_log)

    try:
        rmse = mean_squared_error(y_true_raw, y_pred_raw, squared=False)
    except TypeError:
        rmse = mean_squared_error(y_true_raw, y_pred_raw) ** 0.5

    return {
        "R2_log": r2_score(y_true_log, y_pred_log),
        "MAE_raw": mean_absolute_error(y_true_raw, y_pred_raw),
        "RMSE_raw": rmse,
    }

# ------------------ Main Training Pipeline ------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning")
    parser.add_argument("--n_iter", type=int, default=20, help="Number of parameter combinations for RandomizedSearchCV")
    parser.add_argument("--cv_splits", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    res = Path(args.results_dir)
    fn_path = res / "feature_names.json"
    feature_names = json.loads(fn_path.read_text())

    # Load data matrices and targets
    X_train = load_matrix(res / "X_train")
    X_valid = load_matrix(res / "X_valid")
    X_test = load_matrix(res / "X_test")

    y_train_log = np.load(res / "y_train_log.npy")
    y_valid_log = np.load(res / "y_valid_log.npy")
    y_test_log = np.load(res / "y_test_log.npy")

    y_train_raw = np.load(res / "y_train_raw.npy")
    y_valid_raw = np.load(res / "y_valid_raw.npy")
    y_test_raw = np.load(res / "y_test_raw.npy")

    metrics_rows = []

    # ------------------ Baseline Random Forest ------------------
    print("Training Random Forest Regressor (Traditional ML)...")
    rf = RandomForestRegressor(random_state=args.random_state, n_estimators=200)
    rf.fit(X_train, y_train_log)
    joblib.dump(rf, res / "model_rf.joblib")

    # Evaluate baseline RF
    pred_val_rf = rf.predict(X_valid)
    pred_test_rf = rf.predict(X_test)
    metrics_rows.append({"Model": "RandomForest", "Split": "valid", **eval_regression(y_valid_log, pred_val_rf)})
    metrics_rows.append({"Model": "RandomForest", "Split": "test", **eval_regression(y_test_log, pred_test_rf)})

    # ------------------ Baseline MLP ------------------
    print("Training MLP Regressor (Deep Learning)...")
    mlp = MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=500, random_state=args.random_state)
    mlp.fit(to_dense_if_needed(X_train, "mlp"), y_train_log)
    joblib.dump(mlp, res / "model_mlp.joblib")

    pred_val_mlp = mlp.predict(to_dense_if_needed(X_valid, "mlp"))
    pred_test_mlp = mlp.predict(to_dense_if_needed(X_test, "mlp"))
    metrics_rows.append({"Model": "MLPRegressor", "Split": "valid", **eval_regression(y_valid_log, pred_val_mlp)})
    metrics_rows.append({"Model": "MLPRegressor", "Split": "test", **eval_regression(y_test_log, pred_test_mlp)})

    # ------------------ Hyperparameter Tuning (if enabled) ------------------
    if args.tune:
        print(">>> Running RandomizedSearchCV for RandomForest and MLP...")

        # Parameter grids
        rf_param_grid = {
            "n_estimators": [200, 400, 600, 800],
            "max_depth": [None, 10, 15, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 0.5, None],
        }
        mlp_param_grid = {
            "hidden_layer_sizes": [(64, 32), (128, 64, 32), (256, 128, 64)],
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate_init": [0.001, 0.01],
        }

        # Random Forest Tuning
        rf_search = RandomizedSearchCV(rf, rf_param_grid, n_iter=args.n_iter, scoring="neg_root_mean_squared_error", cv=args.cv_splits, n_jobs=args.n_jobs, random_state=args.random_state, verbose=1)
        rf_search.fit(to_dense_if_needed(X_train, "rf"), y_train_log)
        rf_best = rf_search.best_estimator_
        joblib.dump(rf_best, res / "model_rf_tuned.joblib")

        print("\nBest parameters for RandomForest:")
        print(rf_search.best_params_)
        with open(res / "best_params_rf.json", "w") as f:
            json.dump({"best_params": rf_search.best_params_, "cv_best_score_neg_rmse": float(rf_search.best_score_)}, f, indent=2)

        # MLP Tuning
        mlp_search = RandomizedSearchCV(mlp, mlp_param_grid, n_iter=args.n_iter, scoring="neg_root_mean_squared_error", cv=args.cv_splits, n_jobs=args.n_jobs, random_state=args.random_state, verbose=1)
        mlp_search.fit(to_dense_if_needed(X_train, "mlp"), y_train_log)
        mlp_best = mlp_search.best_estimator_
        joblib.dump(mlp_best, res / "model_mlp_tuned.joblib")

        print("\nBest parameters for MLP Regressor:")
        print(mlp_search.best_params_)
        with open(res / "best_params_mlp.json", "w") as f:
            json.dump({"best_params": mlp_search.best_params_, "cv_best_score_neg_rmse": float(mlp_search.best_score_)}, f, indent=2)

        # Evaluate tuned models
        pred_test_rf_tuned = rf_best.predict(X_test)
        pred_test_mlp_tuned = mlp_best.predict(to_dense_if_needed(X_test, "mlp"))
        metrics_rows.append({"Model": "RandomForest_TUNED", "Split": "test", **eval_regression(y_test_log, pred_test_rf_tuned)})
        metrics_rows.append({"Model": "MLPRegressor_TUNED", "Split": "test", **eval_regression(y_test_log, pred_test_mlp_tuned)})

    # ------------------ Save metrics ------------------
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(res / "metrics_comparison.csv", index=False)
    print("\n=== Model Performance Summary ===")
    print(metrics.head())
    print(f"\nMetrics saved to {res / 'metrics_comparison.csv'}")

if __name__ == "__main__":
    main()
