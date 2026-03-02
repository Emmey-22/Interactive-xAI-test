# framingham_xgb_train.py
# Step 1: Train an XGBoost risk model on Framingham (TenYearCHD) with proper tuning + artifacts saved.

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
)

from xgboost import XGBClassifier


RANDOM_STATE = 42


def find_best_threshold(y_true, y_prob, metric="f1"):
    """
    Selects an operating threshold on validation probabilities.
    Options:
      - metric="f1": maximize F1
      - metric="youden": maximize (TPR - FPR) indirectly not implemented here (needs ROC curve)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # precision_recall_curve returns thresholds of length n-1
    # Align arrays for threshold-based scoring
    precision = precision[:-1]
    recall = recall[:-1]

    if metric == "f1":
        f1 = (2 * precision * recall) / (precision + recall + 1e-12)
        best_idx = int(np.nanargmax(f1))
        return float(thresholds[best_idx]), float(f1[best_idx])

    raise ValueError("Unsupported metric for threshold selection.")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "framingham1.csv")
    out_dir = os.environ.get("ARTIFACT_DIR", os.path.join(base_dir, "artifacts"))
    os.makedirs(out_dir, exist_ok=True)
    model_version = os.environ.get("MODEL_VERSION", datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"))

    # -----------------------------
    # 1) Load data + basic checks
    # -----------------------------
    df = pd.read_csv(data_path)

    if "TenYearCHD" not in df.columns:
        raise ValueError("Target column TenYearCHD not found in dataset.")

    # Separate features/target
    X = df.drop(columns=["TenYearCHD"])
    y = df["TenYearCHD"].astype(int)

    # Diagnostics
    missing = X.isna().sum().sort_values(ascending=False)
    class_counts = y.value_counts()
    pos_rate = y.mean()

    print("Dataset shape:", df.shape)
    print("\nMissing values (non-zero):")
    print(missing[missing > 0])
    print("\nClass distribution:")
    print(class_counts)
    print("Positive rate:", round(float(pos_rate), 4))

    # -----------------------------
    # 2) Split: train / val / test
    # -----------------------------
    # 80/20 train-test, then 80/20 train-val => 64/16/20 overall
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.20, stratify=y_trainval, random_state=RANDOM_STATE
    )

    # -----------------------------
    # 3) Preprocessing
    # -----------------------------
    # Binary-ish columns in Framingham (0/1): impute most_frequent
    binary_cols = [
        "male",
        "currentSmoker",
        "BPMeds",
        "prevalentStroke",
        "prevalentHyp",
        "diabetes",
    ]

    # Education is ordinal (1-4), impute most_frequent to preserve category-like behavior
    ordinal_cols = ["education"]

    # Continuous/quantitative columns: impute median (robust)
    continuous_cols = [c for c in X.columns if c not in set(binary_cols + ordinal_cols)]

    # ColumnTransformer lets us apply different imputers
    preprocessor = ColumnTransformer(
        transformers=[
            ("bin", SimpleImputer(strategy="most_frequent"), binary_cols),
            ("ord", SimpleImputer(strategy="most_frequent"), ordinal_cols),
            ("cont", SimpleImputer(strategy="median"), continuous_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # -----------------------------
    # 4) Model + imbalance handling
    # -----------------------------
    # scale_pos_weight ~ (#neg / #pos) is common for imbalanced binary classification in XGBoost
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = (neg / max(pos, 1))

    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",  # PR-AUC is informative under imbalance
        random_state=RANDOM_STATE,
        n_jobs=1,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
    )

    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", base_model),
    ])

    # -----------------------------
    # 5) Hyperparameter tuning
    # -----------------------------
    # Use StratifiedKFold and optimize ROC-AUC (standard for medical risk prediction).
    # PR-AUC is also computed later; we keep ROC-AUC for tuning stability.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    param_distributions = {
        "model__n_estimators": [200, 400, 600, 800, 1000],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "model__max_depth": [2, 3, 4, 5, 6],
        "model__min_child_weight": [1, 2, 5, 10],
        "model__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "model__gamma": [0.0, 0.1, 0.2, 0.5, 1.0],
        "model__reg_alpha": [0.0, 0.001, 0.01, 0.1, 1.0],
        "model__reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
    }

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=40,                 # increase to 80+ if you have time
        scoring="roc_auc",
        n_jobs=1,
        cv=cv,
        verbose=1,
        random_state=RANDOM_STATE,
        refit=True,
    )

    print("\nTuning hyperparameters with RandomizedSearchCV...")
    search.fit(X_train, y_train)

    best_pipe = search.best_estimator_
    best_params = search.best_params_
    best_cv_auc = float(search.best_score_)

    print("\nBest CV ROC-AUC:", round(best_cv_auc, 4))
    print("Best params:", best_params)

    # -----------------------------
    # 6) Final fit with early stopping (train -> val)
    # -----------------------------
    # Early stopping is fitted on validation set after preprocessing.
    # We must transform data explicitly to pass eval_set to XGBoost.
    prep = best_pipe.named_steps["prep"]
    model = best_pipe.named_steps["model"]

    X_train_p = prep.fit_transform(X_train, y_train)
    X_val_p = prep.transform(X_val)
    X_test_p = prep.transform(X_test)

    model.set_params(
        n_estimators=5000,  # large cap; early stopping will decide
        early_stopping_rounds=50,
    )

    print("\nFitting final model with early stopping...")
    model.fit(
        X_train_p,
        y_train,
        eval_set=[(X_val_p, y_val)],
        verbose=False,
    )

    # -----------------------------
    # 7) Validation metrics + threshold tuning
    # -----------------------------
    val_prob = model.predict_proba(X_val_p)[:, 1]
    val_roc = roc_auc_score(y_val, val_prob)
    val_pr = average_precision_score(y_val, val_prob)

    best_thr, best_val_f1 = find_best_threshold(y_val, val_prob, metric="f1")

    print("\nValidation ROC-AUC:", round(float(val_roc), 4))
    print("Validation PR-AUC :", round(float(val_pr), 4))
    print("Best threshold (F1):", round(best_thr, 4), "Best val F1:", round(best_val_f1, 4))

    # -----------------------------
    # 8) Test metrics (report honestly)
    # -----------------------------
    test_prob = model.predict_proba(X_test_p)[:, 1]
    test_pred = (test_prob >= best_thr).astype(int)

    test_roc = roc_auc_score(y_test, test_prob)
    test_pr = average_precision_score(y_test, test_prob)
    test_bal_acc = balanced_accuracy_score(y_test, test_pred)
    cm = confusion_matrix(y_test, test_pred)

    print("\nTEST RESULTS (threshold tuned on val):")
    print("Test ROC-AUC:", round(float(test_roc), 4))
    print("Test PR-AUC :", round(float(test_pr), 4))
    print("Test Balanced Acc:", round(float(test_bal_acc), 4))
    print("\nConfusion matrix:\n", cm)
    print("\nClassification report:\n", classification_report(y_test, test_pred, digits=4))

    # -----------------------------
    # 9) Save artifacts
    # -----------------------------
    # Save preprocessor + model + chosen threshold + metadata
    artifacts = {
        "threshold": best_thr,
        "best_cv_roc_auc": best_cv_auc,
        "val_roc_auc": float(val_roc),
        "val_pr_auc": float(val_pr),
        "test_roc_auc": float(test_roc),
        "test_pr_auc": float(test_pr),
        "test_balanced_accuracy": float(test_bal_acc),
        "scale_pos_weight_train": float(scale_pos_weight),
        "features": list(X.columns),
        "binary_cols": binary_cols,
        "ordinal_cols": ordinal_cols,
        "continuous_cols": continuous_cols,
        "best_params": best_params,
        "random_state": RANDOM_STATE,
        "model_version": model_version,
        "trained_at_utc": datetime.utcnow().isoformat() + "Z",
        "artifact_dir": out_dir,
    }

    joblib.dump(prep, os.path.join(out_dir, "preprocessor.joblib"))
    joblib.dump(model, os.path.join(out_dir, "xgb_model.joblib"))

    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(artifacts, f, indent=2)

    print("\nSaved artifacts to:", out_dir)
    print(" - preprocessor.joblib")
    print(" - xgb_model.joblib")
    print(" - metadata.json")


if __name__ == "__main__":
    main()
