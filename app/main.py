import os
from fastapi import FastAPI, Query, Depends
from fastapi.middleware.cors import CORSMiddleware

from .schemas import PatientInput, PredictResponse, ExplainResponse, FeedbackRequest, PreferenceRequest
from .db import (
    init_db,
    insert_feedback,
    upsert_preferences,
    get_preferences,
    ensure_user,
    log_user_activity,
    apply_preference_from_feedback,
    feedback_summary,
    top_features_by_feedback,
    log_audit_event,
)
from .security import get_current_user, enforce_rate_limit
from .xai import predict, explain

app = FastAPI(title="Interactive XAI Screening API", version="1.1")

# Allow browser clients (React/Vite) to make cross-origin API calls.
# Use CORS_ORIGINS="https://your-frontend.vercel.app,https://another-origin"
# in deployment environments.
_origins_env = os.environ.get("CORS_ORIGINS", "").strip()
_origin_regex = os.environ.get("CORS_ORIGIN_REGEX", "").strip()
_default_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
_cors_origins = [o.strip() for o in _origins_env.split(",") if o.strip()] if _origins_env else _default_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_origin_regex=_origin_regex or None,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "Interactive XAI Screening API is running.",
        "docs": "/docs",
        "endpoints": ["/predict", "/explain", "/feedback", "/preferences", "/auth/me"],
    }


@app.on_event("startup")
def startup():
    init_db()


@app.get("/auth/me")
def api_auth_me(user_id: str = Depends(get_current_user)):
    ensure_user(user_id)
    return {"user_id": user_id}


@app.post("/predict", response_model=PredictResponse)
def api_predict(inp: PatientInput, user_id: str = Depends(get_current_user)):
    ensure_user(user_id)
    enforce_rate_limit(user_id, "predict", limit=120, window_s=60)
    log_user_activity(user_id, "predict")
    risk, flagged, thr = predict(inp.dict())
    log_audit_event(user_id, action="predict", details=f"flagged={flagged}")
    return PredictResponse(risk=risk, flagged=flagged, threshold=thr)


@app.post("/explain", response_model=ExplainResponse)
def api_explain(inp: PatientInput, user_id: str = Depends(get_current_user)):
    ensure_user(user_id)
    enforce_rate_limit(user_id, "explain", limit=90, window_s=60)
    log_user_activity(user_id, "explain")
    out = explain(inp.dict(), user_id=user_id)
    log_audit_event(user_id, action="explain")
    return out


@app.post("/feedback")
def api_feedback(req: FeedbackRequest, user_id: str = Depends(get_current_user)):
    insert_feedback(
        user_id=user_id,
        feedback_type=req.feedback_type,
        feature_name=req.feature_name,
        case_id=req.case_id,
        message=req.message,
    )

    # Auto preference updates
    if req.feedback_type in ("prefer_short", "prefer_long"):
        apply_preference_from_feedback(user_id, req.feedback_type)

    log_audit_event(user_id, action="feedback", details=f"type={req.feedback_type}")
    return {"status": "ok"}


@app.post("/preferences")
def api_set_prefs(req: PreferenceRequest, user_id: str = Depends(get_current_user)):
    upsert_preferences(user_id, req.top_k, req.style)
    log_audit_event(user_id, action="set_preferences", details=f"top_k={req.top_k},style={req.style}")
    return {"status": "ok"}


@app.get("/preferences")
def api_get_prefs(user_id: str = Depends(get_current_user)):
    return get_preferences(user_id)


@app.get("/analytics/summary")
def api_analytics_summary(user_id: str = Depends(get_current_user)):
    return {"summary": feedback_summary(user_id=user_id)}


@app.get("/analytics/top_features")
def api_top_features(feedback_type: str, limit: int = Query(10, ge=1, le=100), user_id: str = Depends(get_current_user)):
    return {"feedback_type": feedback_type, "top_features": top_features_by_feedback(feedback_type, limit, user_id)}
