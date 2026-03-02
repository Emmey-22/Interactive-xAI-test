
# framingham_step2_shap.py
import os
import json
import joblib
import numpy as np
import pandas as pd
import shap

RANDOM_STATE = 42

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "framingham1.csv")
    art_dir = os.environ.get("ARTIFACT_DIR", os.path.join(script_dir, "artifacts"))
    if not os.path.exists(os.path.join(art_dir, "preprocessor.joblib")):
        raise FileNotFoundError(
            f"Missing preprocessor artifact in {art_dir}. "
            "Run training and calibration with matching ARTIFACT_DIR first."
        )
    out_dir = os.path.join(art_dir, "shap_outputs")
    os.makedirs(out_dir, exist_ok=True)

    # Load artifacts
    prep = joblib.load(os.path.join(art_dir, "preprocessor.joblib"))
    base_model = joblib.load(os.path.join(art_dir, "xgb_model.joblib"))
    cal_model = joblib.load(os.path.join(art_dir, "xgb_calibrated_screening.joblib"))

    with open(os.path.join(art_dir, "screening_config.json"), "r") as f:
        cfg = json.load(f)

    screening_threshold = float(cfg["screening_threshold"])
    feature_names = list(prep.get_feature_names_out())

    # Load data
    df = pd.read_csv(data_path)
    X = df.drop(columns=["TenYearCHD"])
    y = df["TenYearCHD"].astype(int)

    # Reuse the fitted preprocessor from training artifacts.
    X_p = prep.transform(X)

    # -----------------------------
    # 1) Global SHAP (TreeExplainer)
    # -----------------------------
    explainer = shap.TreeExplainer(base_model)
    shap_values = explainer.shap_values(X_p)

    # Global importance: mean(|SHAP|)
    mean_abs = np.mean(np.abs(shap_values), axis=0)

    global_imp = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs
    }).sort_values("mean_abs_shap", ascending=False)

    global_path = os.path.join(out_dir, "global_shap_importance.csv")
    global_imp.to_csv(global_path, index=False)
    print("Saved global SHAP importance:", global_path)

    # -----------------------------
    # 2) Local SHAP for a sample case
    # -----------------------------
    # Example: explain the first row (you will replace index with the user's input later)
    idx = 0
    x_row = X.iloc[[idx]]
    x_row_p = prep.transform(x_row)

    # calibrated risk and screening decision
    risk = float(cal_model.predict_proba(x_row_p)[:, 1][0])
    flagged = bool(risk >= screening_threshold)

    # local shap values
    local_shap = shap_values[idx, :]
    local_df = pd.DataFrame({
        "feature": feature_names,
        "value": x_row.iloc[0].values,
        "shap": local_shap
    })

    # Rank by absolute contribution
    local_df["abs_shap"] = np.abs(local_df["shap"])
    local_df = local_df.sort_values("abs_shap", ascending=False)

    local_out = {
        "row_index": idx,
        "calibrated_risk": risk,
        "screening_threshold": screening_threshold,
        "screening_flagged": flagged,
        "top_positive": local_df[local_df["shap"] > 0].head(8)[["feature", "value", "shap"]].to_dict(orient="records"),
        "top_negative": local_df[local_df["shap"] < 0].head(8)[["feature", "value", "shap"]].to_dict(orient="records"),
        "top_all": local_df.head(12)[["feature", "value", "shap"]].to_dict(orient="records"),
    }

    local_path = os.path.join(out_dir, "local_explanation_example.json")
    with open(local_path, "w") as f:
        json.dump(local_out, f, indent=2)

    print("Saved local SHAP example explanation:", local_path)
    print("\nExample case:")
    print(" - Calibrated risk:", round(risk, 4))
    print(" - Screening flagged:", flagged)

    # -----------------------------
    # 3) Save a small background set for faster web explanations
    # -----------------------------
    # Use a small representative subset for SHAP background in deployment
    # (keeps explanations fast)
    rng = np.random.default_rng(RANDOM_STATE)
    bg_idx = rng.choice(len(X_p), size=min(200, len(X_p)), replace=False)
    bg = X_p[bg_idx]

    joblib.dump(bg, os.path.join(out_dir, "shap_background_200.joblib"))
    print("Saved SHAP background sample:", os.path.join(out_dir, "shap_background_200.joblib"))

if __name__ == "__main__":
    main()
