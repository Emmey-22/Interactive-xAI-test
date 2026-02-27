# next_step_screening_calibrate.py
import os, json, joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix

RANDOM_STATE = 42

def pick_threshold_for_target_recall(y_true, y_prob, target_recall=0.85):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    precision, recall = precision[:-1], recall[:-1]
    eligible = np.where(recall >= target_recall)[0]
    if len(eligible) == 0:
        # fallback: choose threshold with max recall (screening-first)
        idx = int(np.argmax(recall))
        return float(thresholds[idx]), float(precision[idx]), float(recall[idx]), "max_recall_fallback"
    # choose highest threshold that still meets recall target (reduces false positives)
    idx = int(eligible[-1])
    return float(thresholds[idx]), float(precision[idx]), float(recall[idx]), "recall_target"

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "framingham1.csv")
    art_dir = os.path.join(base_dir, "framingham_xgb_artifacts")

    prep = joblib.load(os.path.join(art_dir, "preprocessor.joblib"))
    base_model = joblib.load(os.path.join(art_dir, "xgb_model.joblib"))
    # CalibratedClassifierCV refits the estimator without eval_set; disable early stopping.
    base_model.set_params(early_stopping_rounds=None)

    df = pd.read_csv(data_path)
    X = df.drop(columns=["TenYearCHD"])
    y = df["TenYearCHD"].astype(int)

    # same split logic as training
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.20, stratify=y_trainval, random_state=RANDOM_STATE
    )

    X_train_p = prep.fit_transform(X_train, y_train)
    X_val_p = prep.transform(X_val)
    X_test_p = prep.transform(X_test)

    # calibration (sigmoid)
    cal = CalibratedClassifierCV(base_model, method="sigmoid", cv=3)
    cal.fit(X_train_p, y_train)

    val_prob = cal.predict_proba(X_val_p)[:, 1]
    test_prob = cal.predict_proba(X_test_p)[:, 1]

    # discrimination metrics (still report)
    val_roc = roc_auc_score(y_val, val_prob)
    val_pr = average_precision_score(y_val, val_prob)
    test_roc = roc_auc_score(y_test, test_prob)
    test_pr = average_precision_score(y_test, test_prob)

    # screening threshold (choose recall target)
    screening_target_recall = 0.85
    thr, prec, rec, mode = pick_threshold_for_target_recall(y_val, val_prob, screening_target_recall)

    test_pred = (test_prob >= thr).astype(int)
    cm = confusion_matrix(y_test, test_pred)

    print("CALIBRATED DISCRIMINATION")
    print(f"Val ROC-AUC/PR-AUC:  {val_roc:.4f} / {val_pr:.4f}")
    print(f"Test ROC-AUC/PR-AUC: {test_roc:.4f} / {test_pr:.4f}")
    print("")
    print("SCREENING OPERATING POINT")
    print(f"Target recall: {screening_target_recall:.2f}")
    print(f"Chosen threshold: {thr:.4f} ({mode})")
    print(f"Val precision/recall at thr: {prec:.4f} / {rec:.4f}")
    print("")
    print("TEST CONFUSION MATRIX (screening threshold)")
    print(cm)

    # save
    joblib.dump(cal, os.path.join(art_dir, "xgb_calibrated_screening.joblib"))
    with open(os.path.join(art_dir, "screening_config.json"), "w") as f:
        json.dump(
            {
                "screening_target_recall": screening_target_recall,
                "screening_threshold": thr,
                "threshold_mode": mode,
                "val_roc_auc": float(val_roc),
                "val_pr_auc": float(val_pr),
                "test_roc_auc": float(test_roc),
                "test_pr_auc": float(test_pr),
                "features": list(X.columns),
            },
            f,
            indent=2,
        )

    print("\nSaved:")
    print("- xgb_calibrated_screening.joblib")
    print("- screening_config.json")

if __name__ == "__main__":
    main()
