"""
train_models.py
---------------
Step 3: Train, evaluate (and optionally tune) three regressors:
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

Inputs (from Step 2, in --results_dir):
- X_train.npz / X_valid.npz / X_test.npz
- y_train_log.npy / y_valid_log.npy / y_test_log.npy
- y_train_raw.npy / y_valid_raw.npy / y_test_raw.npy
- feature_names.json (for feature importances)
- preprocessor.joblib (not directly used here but useful downstream)

Outputs:
- results/metrics_comparison.csv
- results/model_linear.joblib
- results/model_rf.joblib
- results/model_xgb.joblib
- results/feature_importance_rf.csv
- results/feature_importance_xgb.csv
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from scipy import sparse

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, KFold

# XGBoost
from xgboost import XGBRegressor


# ---------------------------
# Utility helpers
# ---------------------------

def load_matrix(path: Path):
    """
    Robust loader:
    - If it's a sparse .npz, return CSR matrix
    - If it's a dense .npy, return ndarray
    - Tries .npz, then .npy if extensionless
    """
    path = Path(path)
    candidates = []

    # if user passed a path with suffix, try that first
    if path.suffix:
        candidates.append(path)

    # then try explicit .npz and .npy variants
    candidates.extend([path.with_suffix(".npz"), path.with_suffix(".npy")])

    tried = []
    for p in candidates:
        if not p.exists():
            tried.append(f"{p} (missing)")
            continue
        # try sparse first
        try:
            return sparse.load_npz(p)
        except Exception:
            # fall back to dense
            try:
                return np.load(p, allow_pickle=False)
            except Exception as e:
                tried.append(f"{p} ({type(e).__name__}: {e})")
                continue

    raise FileNotFoundError(f"Could not load matrix. Tried: {', '.join(tried)}")


def to_dense_if_needed(X, for_model: str):
    """
    RandomForest cannot handle sparse matrices -> make dense.
    Linear and XGB accept sparse CSR (fine to keep).
    """
    if for_model.lower() in {"rf", "randomforest"}:
        if sparse.issparse(X):
            return X.toarray()
    return X


def eval_regression(y_true_log, y_pred_log):
    """
    R2 in log-space; MAE/RMSE in raw price.
    Compatible with old scikit-learn that lacks squared=.
    """
    # R² on log targets (stable variance metric)
    r2_log = r2_score(y_true_log, y_pred_log)

    # Back-transform to raw prices
    y_true_raw = np.expm1(y_true_log)
    y_pred_raw = np.expm1(y_pred_log)

    mae = mean_absolute_error(y_true_raw, y_pred_raw)

    # Try new API first; fall back to sqrt(MSE)
    try:
        rmse = mean_squared_error(y_true_raw, y_pred_raw, squared=False)
    except TypeError:
        rmse = np.sqrt(mean_squared_error(y_true_raw, y_pred_raw))

    return {"R2_log": r2_log, "MAE_raw": mae, "RMSE_raw": rmse}


def save_feature_importance(model, feature_names, out_csv: Path):
    """
    Save feature importances if available (RF/XGB).
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        df_imp = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        ).sort_values("importance", ascending=False)
        df_imp.to_csv(out_csv, index=False)


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Train & evaluate baseline and tuned regressors.")
    parser.add_argument("--results_dir", default="results", help="Directory that contains Step 2 artifacts and where outputs will be saved.")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning (RF & XGB) via RandomizedSearchCV.")
    parser.add_argument("--cv_splits", type=int, default=5, help="CV splits for tuning.")
    parser.add_argument("--n_iter", type=int, default=40, help="Number of parameter samples for random search.")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Parallel jobs for CV.")
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    res = Path(args.results_dir)
    res.mkdir(parents=True, exist_ok=True)

    # ---- Load data
    X_train = load_matrix(res / "X_train.npz")
    X_valid = load_matrix(res / "X_valid.npz")
    X_test  = load_matrix(res / "X_test.npz")

    y_train_log = np.load(res / "y_train_log.npy")
    y_valid_log = np.load(res / "y_valid_log.npy")
    y_test_log  = np.load(res / "y_test_log.npy")

    # these are only for reference / optional checks
    # y_train_raw = np.load(res / "y_train_raw.npy")
    # y_valid_raw = np.load(res / "y_valid_raw.npy")
    # y_test_raw  = np.load(res / "y_test_raw.npy")

    # feature names (for importances)
    feature_names = []
    fn_path = res / "feature_names.json"
    if fn_path.exists():
        feature_names = json.loads(fn_path.read_text())

    metrics_rows = []

    # ======================================================
    # 1) Linear Regression (baseline)
    # ======================================================
    lin = LinearRegression(n_jobs=None)  # LinearRegression has no n_jobs param; keep default
    lin.fit(X_train, y_train_log)
    pred_val_lin = lin.predict(X_valid)
    pred_tst_lin = lin.predict(X_test)

    m_val = eval_regression(y_valid_log, pred_val_lin)
    m_tst = eval_regression(y_test_log, pred_tst_lin)

    joblib.dump(lin, res / "model_linear.joblib")
    metrics_rows.append({"Model": "LinearRegression", "Split": "valid", **m_val})
    metrics_rows.append({"Model": "LinearRegression", "Split": "test",  **m_tst})

    # ======================================================
    # 2) Random Forest
    # ======================================================
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
    # 3) XGBoost (version-flexible)
    # ======================================================
    from xgboost import XGBRegressor

    xgb = XGBRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        tree_method="hist",
        # put eval_metric here for older XGB that doesn't accept it in fit()
        eval_metric="rmse",
    )

    # Try modern API first; fall back if older xgboost version
    try:
        xgb.fit(
            X_train,
            y_train_log,
            eval_set=[(X_valid, y_valid_log)],
            early_stopping_rounds=50,
            verbose=False,
        )
    except TypeError:
        # Very old versions: no early_stopping_rounds/verbose in fit
        try:
            xgb.fit(X_train, y_train_log, eval_set=[(X_valid, y_valid_log)])
        except TypeError:
            # Oldest: no eval_set either
            xgb.fit(X_train, y_train_log)

    pred_val_xgb = xgb.predict(X_valid)
    pred_tst_xgb = xgb.predict(X_test)

    m_val = eval_regression(y_valid_log, pred_val_xgb)
    m_tst = eval_regression(y_test_log, pred_tst_xgb)

    joblib.dump(xgb, res / "model_xgb.joblib")
    metrics_rows.append({"Model": "XGBoost", "Split": "valid", **m_val})
    metrics_rows.append({"Model": "XGBoost", "Split": "test",  **m_tst})

    if feature_names:
        save_feature_importance(xgb, feature_names, res / "feature_importance_xgb.csv")

    # ======================================================
    # Optional: Hyperparameter Tuning (RF & XGB)
    # ======================================================
    if args.tune:
        print(">>> Tuning RandomForest and XGBoost with RandomizedSearchCV...")
        cv = KFold(n_splits=args.cv_splits, shuffle=True, random_state=args.random_state)

        # ----- Random Forest search
        rf_base = RandomForestRegressor(random_state=args.random_state, n_jobs=args.n_jobs)
        rf_space = {
            "n_estimators": [200, 400, 600, 800, 1000],
            "max_depth": [None, 8, 12, 16, 24],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 0.5, 0.7, None],
            "bootstrap": [True, False],
        }
        rf_search = RandomizedSearchCV(
            estimator=rf_base,
            param_distributions=rf_space,
            n_iter=args.n_iter,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            verbose=1,
        )
        rf_search.fit(to_dense_if_needed(X_train, "rf"), y_train_log)
        rf_best = rf_search.best_estimator_

        # Evaluate on valid
        pred_val_rf_tuned = rf_best.predict(to_dense_if_needed(X_valid, "rf"))
        m_val_rf_tuned = eval_regression(y_valid_log, pred_val_rf_tuned)
        metrics_rows.append({"Model": "RandomForest_TUNED", "Split": "valid", **m_val_rf_tuned})

        # Refit on train+valid, test
        X_trv = to_dense_if_needed(sparse.vstack([X_train, X_valid]) if sparse.issparse(X_train) else np.vstack([X_train, X_valid]), "rf")
        y_trv = np.concatenate([y_train_log, y_valid_log])
        rf_best.fit(X_trv, y_trv)
        pred_tst_rf_tuned = rf_best.predict(to_dense_if_needed(X_test, "rf"))
        m_tst_rf_tuned = eval_regression(y_test_log, pred_tst_rf_tuned)
        metrics_rows.append({"Model": "RandomForest_TUNED", "Split": "test", **m_tst_rf_tuned})
        joblib.dump(rf_best, res / "model_rf_tuned.joblib")

        # ----- XGBoost search
        xgb_base = XGBRegressor(
            objective="reg:squarederror",
            n_jobs=args.n_jobs,
            random_state=args.random_state,
            tree_method="hist",
        )
        xgb_space = {
            "n_estimators": [400, 600, 800, 1000, 1200],
            "learning_rate": np.linspace(0.02, 0.2, 10),
            "max_depth": [3, 4, 5, 6, 8, 10],
            "subsample": np.linspace(0.6, 1.0, 5),
            "colsample_bytree": np.linspace(0.6, 1.0, 5),
            "reg_alpha": [0.0, 0.01, 0.1, 0.5, 1.0],
            "reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
        }
        xgb_search = RandomizedSearchCV(
            estimator=xgb_base,
            param_distributions=xgb_space,
            n_iter=args.n_iter,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            verbose=1,
        )
        xgb_search.fit(X_train, y_train_log)
        xgb_best = xgb_search.best_estimator_

        # Evaluate on valid
        pred_val_xgb_tuned = xgb_best.predict(X_valid)
        m_val_xgb_tuned = eval_regression(y_valid_log, pred_val_xgb_tuned)
        metrics_rows.append({"Model": "XGBoost_TUNED", "Split": "valid", **m_val_xgb_tuned})

        # Refit on train+valid with early-stopping using a small valid fold from train+valid (optional)
        X_trv = sparse.vstack([X_train, X_valid]) if sparse.issparse(X_train) else np.vstack([X_train, X_valid])
        y_trv = np.concatenate([y_train_log, y_valid_log])
        xgb_best.fit(
            X_trv,
            y_trv,
            eval_set=[(X_valid, y_valid_log)],
            eval_metric="rmse",
            early_stopping_rounds=50,
            verbose=False,
        )
        pred_tst_xgb_tuned = xgb_best.predict(X_test)
        m_tst_xgb_tuned = eval_regression(y_test_log, pred_tst_xgb_tuned)
        metrics_rows.append({"Model": "XGBoost_TUNED", "Split": "test", **m_tst_xgb_tuned})
        joblib.dump(xgb_best, res / "model_xgb_tuned.joblib")

    # ---- Save metrics table
    df_metrics = pd.DataFrame(metrics_rows)
    df_metrics.to_csv(res / "metrics_comparison.csv", index=False)

    # tiny printout
    print("\n=== Metrics (head) ===")
    print(df_metrics.head(10))
    print(f"\nSaved metrics to {res / 'metrics_comparison.csv'}")
    print("Done.")


if __name__ == "__main__":
    main()
