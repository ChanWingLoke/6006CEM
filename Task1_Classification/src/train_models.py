import argparse
import os
import json
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report

from .preprocess import load_raw, clean_basic, split_Xy, build_preprocessor, make_train_test
from .imbalance_handler import smote_balance, get_class_weights, describe_distribution
from .models import get_random_forest, get_keras_classifier
from .evaluate import evaluate_binary, save_metrics, plot_confusion_matrix, plot_roc

RESULTS_DIR = "results"

def run_pipeline(data_path: str, balance: str = "smote", tune: bool = False, random_state: int = 42):
    # Load & preprocess
    raw = load_raw(data_path)
    df = clean_basic(raw)
    X, y = split_Xy(df)

    X_train, X_test, y_train, y_test = make_train_test(X, y, test_size=0.2, random_state=random_state)

    # Preprocessor
    preprocessor = build_preprocessor(X_train)

    # Report distribution
    print("Train distribution:", describe_distribution(y_train, "train"))
    print("Test distribution :", describe_distribution(y_test, "test"))

    # ===== Random Forest =====
    rf = get_random_forest()
    rf_pipe = Pipeline([("prep", preprocessor), ("model", rf)])

    rf_param_grid = {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [None, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", None],
    }

    if tune:
        rf_grid = GridSearchCV(rf_pipe, rf_param_grid, cv=3, n_jobs=-1, scoring="f1", verbose=1)
        rf_grid.fit(X_train, y_train)
        rf_best = rf_grid.best_estimator_
        print("RF best params:", rf_grid.best_params_)
    else:
        rf_pipe.fit(X_train, y_train)
        rf_best = rf_pipe

    # Evaluate RF
    rf_pred = rf_best.predict(X_test)
    try:
        rf_proba = rf_best.predict_proba(X_test)[:,1]
    except Exception:
        rf_proba = None
    rf_metrics = evaluate_binary(y_test, rf_pred, rf_proba)
    print("Random Forest metrics:", rf_metrics)

    # Save RF metrics & plots
    save_metrics(rf_metrics, os.path.join(RESULTS_DIR, "metrics", "random_forest_metrics.json"))
    plot_confusion_matrix(y_test, rf_pred, os.path.join(RESULTS_DIR, "confusion_matrices", "rf_confusion.png"), "RF Confusion Matrix")
    if rf_proba is not None:
        plot_roc(y_test, rf_proba, os.path.join(RESULTS_DIR, "confusion_matrices", "rf_roc.png"), "RF ROC")

    # ===== Neural Network (Keras via scikeras) =====
    # For NN we may use SMOTE or class weights
    if balance.lower() == "smote":
        # Fit preprocessor on train, transform, then SMOTE
        X_train_trans = preprocessor.fit_transform(X_train)
        X_test_trans = preprocessor.transform(X_test)
        X_train_bal, y_train_bal = smote_balance(X_train_trans, y_train)
        class_weight = None
    else:
        # Use class weights
        X_train_trans = preprocessor.fit_transform(X_train)
        X_test_trans = preprocessor.transform(X_test)
        X_train_bal, y_train_bal = X_train_trans, y_train
        class_weight = get_class_weights(y_train)

    input_dim = X_train_bal.shape[1]

    nn = get_keras_classifier(input_dim)

    nn_param_grid = {
        "hidden_units": [64, 128],
        "dropout": [0.2, 0.3],
        "lr": [1e-3, 5e-4],
        "epochs": [20, 30],
        "batch_size": [64, 128],
    }

    if tune:
        nn_grid = GridSearchCV(nn, nn_param_grid, cv=3, n_jobs=1, scoring="f1", verbose=1)  # n_jobs=1 due to TF
        nn_grid.fit(X_train_bal, y_train_bal, class_weight=class_weight)
        nn_best = nn_grid.best_estimator_
        print("NN best params:", nn_grid.best_params_)
        # Save tuned weights if available (optional)
    else:
        nn.fit(X_train_bal, y_train_bal, class_weight=class_weight, verbose=0)
        nn_best = nn

    # Evaluate NN
    nn_pred = nn_best.predict(X_test_trans)
    # scikeras returns class predictions; try to also get probabilities
    try:
        nn_proba = nn_best.predict_proba(X_test_trans)[:,1]
    except Exception:
        nn_proba = None

    nn_metrics = evaluate_binary(y_test, nn_pred, nn_proba)
    print("Neural Net metrics:", nn_metrics)

    save_metrics(nn_metrics, os.path.join(RESULTS_DIR, "metrics", "neural_net_metrics.json"))
    plot_confusion_matrix(y_test, nn_pred, os.path.join(RESULTS_DIR, "confusion_matrices", "nn_confusion.png"), "NN Confusion Matrix")
    if nn_proba is not None:
        plot_roc(y_test, nn_proba, os.path.join(RESULTS_DIR, "confusion_matrices", "nn_roc.png"), "NN ROC")

    # Summary
    summary = {"random_forest": rf_metrics, "neural_network": nn_metrics}
    with open(os.path.join(RESULTS_DIR, "metrics", "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Train churn models (RF + NN) with imbalance handling.")
    parser.add_argument("--data", required=True, help="Path to WA_Fn-UseC_-Telco-Customer-Churn.csv")
    parser.add_argument("--balance", default="smote", choices=["smote", "class_weight"], help="Imbalance strategy for NN")
    parser.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning")
    args = parser.parse_args()

    run_pipeline(data_path=args.data, balance=args.balance, tune=args.tune)

if __name__ == "__main__":
    main()
