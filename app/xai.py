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
_MODEL_FEATURES = list(_prep.get_feature_names_out())
_METADATA_PATH = os.path.join(ART_DIR, "metadata.json")
_METADATA = {}
if os.path.exists(_METADATA_PATH):
    with open(_METADATA_PATH, "r") as f:
        _METADATA = json.load(f)

_MODEL_VERSION = (
    os.environ.get("MODEL_VERSION")
    or str(_METADATA.get("model_version") or "").strip()
    or str(_METADATA.get("trained_at_utc") or "").strip()
    or "unknown"
)

# Optional background for faster/steadier SHAP
_bg_path = os.path.join(ART_DIR, "shap_outputs", "shap_background_200.joblib")
_BG = joblib.load(_bg_path) if os.path.exists(_bg_path) else None

_CALIBRATED_ESTIMATORS = (
    [cc.estimator for cc in getattr(_cal_model, "calibrated_classifiers_", []) if hasattr(cc, "estimator")]
    or [_base_model]
)

_EXPLAINERS = [
    shap.TreeExplainer(model, data=_BG) if _BG is not None else shap.TreeExplainer(model)
    for model in _CALIBRATED_ESTIMATORS
]

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

def _to_1d_dense(x):
    if hasattr(x, "toarray"):
        arr = x.toarray()
    else:
        arr = np.asarray(x)
    if arr.ndim == 2:
        arr = arr[0]
    return np.asarray(arr, dtype=float)

def _extract_shap_vector(shap_out):
    if isinstance(shap_out, list):
        shap_out = shap_out[0]
    arr = np.asarray(shap_out)
    if arr.ndim == 2:
        arr = arr[0]
    return np.asarray(arr, dtype=float)

def _mean_ensemble_shap_values(x_p):
    vectors = [_extract_shap_vector(explainer.shap_values(x_p)) for explainer in _EXPLAINERS]
    return np.mean(np.vstack(vectors), axis=0)

def _predict_from_preprocessed(x_p):
    risk = float(_cal_model.predict_proba(x_p)[:, 1][0])
    flagged = bool(risk >= _THRESHOLD)
    return risk, flagged, _THRESHOLD

def get_model_info():
    return {
        "model_version": _MODEL_VERSION,
        "artifact_dir": ART_DIR,
        "feature_count": len(_MODEL_FEATURES),
        "features": list(_MODEL_FEATURES),
        "screening_threshold": _THRESHOLD,
        "screening_target_recall": _cfg.get("screening_target_recall"),
        "threshold_mode": _cfg.get("threshold_mode"),
        "metrics": {
            "val_roc_auc": _cfg.get("val_roc_auc", _METADATA.get("val_roc_auc")),
            "val_pr_auc": _cfg.get("val_pr_auc", _METADATA.get("val_pr_auc")),
            "test_roc_auc": _cfg.get("test_roc_auc", _METADATA.get("test_roc_auc")),
            "test_pr_auc": _cfg.get("test_pr_auc", _METADATA.get("test_pr_auc")),
        },
    }

def _pack(df, disputed_features=None):
    disputed_set = set(disputed_features or [])
    return [
        {
            "feature": r["feature"],
            "value": r["value"],
            "shap": float(r["shap"]),
            "disputed": r["feature"] in disputed_set,
        }
        for _, r in df.iterrows()
    ]

def predict(patient_dict):
    x = _to_df(patient_dict)
    x_p = _prep.transform(x)
    return _predict_from_preprocessed(x_p)

def explain(patient_dict, user_id=None, case_id=None):
    x = _to_df(patient_dict)
    x_p = _prep.transform(x)
    x_p_row = _to_1d_dense(x_p)

    risk, flagged, thr = _predict_from_preprocessed(x_p)
    shap_vals = _mean_ensemble_shap_values(x_p)

    local = pd.DataFrame({
        "feature": _MODEL_FEATURES,
        "value": x_p_row,
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
        disputed = get_disputed_features(user_id, case_id=case_id)
        confusing = get_confusing_features(user_id, case_id=case_id)

    top_k = max(1, min(20, int(prefs.get("top_k", 8))))
    style = prefs.get("style", "simple")
    if style not in ("simple", "detailed"):
        style = "simple"
    valid_feature_set = set(local["feature"])
    disputed = [f for f in disputed if f in valid_feature_set]
    confusing = [f for f in confusing if f in valid_feature_set]

    # Keep disputed features visible; show a dedicated disputed slice for transparency.
    disputed_slice = local[local["feature"].isin(disputed)].head(top_k)
    top_pos = local[local["shap"] > 0].head(top_k)
    top_neg = local[local["shap"] < 0].head(top_k)

    clarifications = []
    for f in confusing:
        if f in FEATURE_GLOSSARY:
            item = {"feature": f}
            item.update(FEATURE_GLOSSARY[f])
            clarifications.append(item)

    meta = {
        "mode": "screening",
        "style": style,
        "model_version": _MODEL_VERSION,
        "note": "Screening mode prioritizes sensitivity; false positives are expected.",
        "risk_source": "calibrated_screening_model",
        "explanation_source": "mean_tree_shap_over_calibration_estimators",
        "explanation_model_count": len(_EXPLAINERS),
        "explanation_note": "SHAP values come from tree models under calibration; calibrated probability is used for final risk.",
        "adaptation_scope": "case" if case_id else "user",
        "case_feedback_id": case_id,
        "clarifications": clarifications,
        "disclaimer": "This tool is for screening support only and does not provide a medical diagnosis. Consult a qualified clinician for decisions."
    }

    return {
        "risk": risk,
        "threshold": thr,
        "flagged": flagged,
        "case_id": case_id,
        "top_positive": _pack(top_pos, disputed_features=disputed),
        "top_negative": _pack(top_neg, disputed_features=disputed),
        "disputed_features": disputed,
        "hidden_contributors": _pack(disputed_slice, disputed_features=disputed),
        "meta": meta,
    }
