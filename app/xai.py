import os
import json
import joblib
import numpy as np
import pandas as pd
import shap

from .db import get_preferences, get_disputed_features, get_confusing_features

ART_DIR = os.environ.get("ART_DIR", "artifacts")

# Load once at startup
_prep = joblib.load(os.path.join(ART_DIR, "preprocessor.joblib"))
_base_model = joblib.load(os.path.join(ART_DIR, "xgb_model.joblib"))
_cal_model = joblib.load(os.path.join(ART_DIR, "xgb_calibrated_screening.joblib"))

with open(os.path.join(ART_DIR, "screening_config.json"), "r") as f:
    _cfg = json.load(f)

_THRESHOLD = float(_cfg["screening_threshold"])
_FEATURES = list(_cfg["features"])

# Optional background for faster/steadier SHAP
_bg_path = os.path.join(ART_DIR, "shap_outputs", "shap_background_200.joblib")
_BG = joblib.load(_bg_path) if os.path.exists(_bg_path) else None

_EXPLAINER = (
    shap.TreeExplainer(_base_model, data=_BG)
    if _BG is not None
    else shap.TreeExplainer(_base_model)
)

FEATURE_GLOSSARY = {
    "prevalentHyp": {"desc": "History of hypertension (0/1).", "unit": "binary"},
    "sysBP": {"desc": "Systolic blood pressure.", "unit": "mmHg"},
    "diaBP": {"desc": "Diastolic blood pressure.", "unit": "mmHg"},
    "totChol": {"desc": "Total cholesterol.", "unit": "mg/dL"},
    "glucose": {"desc": "Blood glucose level.", "unit": "mg/dL"},
    "BMI": {"desc": "Body mass index.", "unit": "kg/m^2"},
    "heartRate": {"desc": "Resting heart rate.", "unit": "beats/min"},
    "cigsPerDay": {"desc": "Cigarettes smoked per day.", "unit": "count"},
    "currentSmoker": {"desc": "Current smoking status (0/1).", "unit": "binary"},
    "diabetes": {"desc": "Diabetes diagnosis (0/1).", "unit": "binary"},
    "BPMeds": {"desc": "On blood pressure medication (0/1).", "unit": "binary"},
    "prevalentStroke": {"desc": "History of stroke (0/1).", "unit": "binary"},
    "age": {"desc": "Age.", "unit": "years"},
    "male": {"desc": "Sex: male (1) or female (0).", "unit": "binary"},
    "education": {"desc": "Education level (ordinal category).", "unit": "category"},
}

def _to_df(patient_dict):
    # Enforce exact column order
    return pd.DataFrame([patient_dict], columns=_FEATURES)

def _pack(df):
    return [
        {"feature": r["feature"], "value": r["value"], "shap": float(r["shap"])}
        for _, r in df.iterrows()
    ]

def predict(patient_dict):
    x = _to_df(patient_dict)
    x_p = _prep.transform(x)
    risk = float(_cal_model.predict_proba(x_p)[:, 1][0])
    flagged = bool(risk >= _THRESHOLD)
    return risk, flagged, _THRESHOLD

def explain(patient_dict, user_id=None):
    x = _to_df(patient_dict)
    x_p = _prep.transform(x)

    risk, flagged, thr = predict(patient_dict)

    shap_out = _EXPLAINER.shap_values(x_p)
    if isinstance(shap_out, list):
        shap_out = shap_out[0]

    shap_vals = np.array(shap_out)
    if shap_vals.ndim == 2:
        shap_vals = shap_vals[0]

    local = pd.DataFrame({
        "feature": _FEATURES,
        "value": x.iloc[0].values,
        "shap": shap_vals
    })
    local["abs_shap"] = np.abs(local["shap"])
    local = local.sort_values("abs_shap", ascending=False)

    # Preferences + feedback
    prefs = {"top_k": 8, "style": "simple"}
    disputed = []
    confusing = []

    if user_id:
        prefs = get_preferences(user_id)
        disputed = get_disputed_features(user_id)
        confusing = get_confusing_features(user_id)

    top_k = int(prefs.get("top_k", 8))

    # Transparency: show what was hidden
    hidden = local[local["feature"].isin(disputed)].head(top_k)

    # Main display excludes disputed features
    filtered = local[~local["feature"].isin(disputed)]

    top_pos = filtered[filtered["shap"] > 0].head(top_k)
    top_neg = filtered[filtered["shap"] < 0].head(top_k)

    clarifications = []
    for f in confusing:
        if f in FEATURE_GLOSSARY:
            item = {"feature": f}
            item.update(FEATURE_GLOSSARY[f])
            clarifications.append(item)

    meta = {
        "mode": "screening",
        "style": prefs.get("style", "simple"),
        "note": "Screening mode prioritizes sensitivity; false positives are expected.",
        "clarifications": clarifications,
        "disclaimer": "This tool is for screening support only and does not provide a medical diagnosis. Consult a qualified clinician for decisions."
    }

    return {
        "risk": risk,
        "threshold": thr,
        "flagged": flagged,
        "top_positive": _pack(top_pos),
        "top_negative": _pack(top_neg),
        "disputed_features": disputed,
        "hidden_contributors": _pack(hidden),
        "meta": meta,
    }