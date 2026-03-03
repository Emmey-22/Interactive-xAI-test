import os
from uuid import uuid4
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from .schemas import (
    PatientInput,
    PredictResponse,
    ExplainResponse,
    FeedbackRequest,
    PreferenceRequest,
    FeedbackType,
)
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
)
from .xai import predict, explain, get_model_info
from .security import (
    validate_security_configuration,
    resolve_user_id,
    enforce_rate_limit,
)

app = FastAPI(title="Interactive XAI Screening API", version="1.0")

# Allow browser clients (React/Vite) to make cross-origin API calls.
# Use CORS_ORIGINS="https://your-frontend.vercel.app,https://another-origin"
# in deployment environments.
_origins_env = os.environ.get("CORS_ORIGINS", "").strip()
_origin_regex = os.environ.get("CORS_ORIGIN_REGEX", "").strip()
_default_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
env_origins = [o.strip() for o in _origins_env.split(",") if o.strip()]
_cors_origins = sorted(set(_default_origins + env_origins))

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
        "endpoints": [
            "/predict",
            "/explain",
            "/feedback",
            "/preferences",
            "/analytics/summary",
            "/analytics/top_features",
            "/model/info",
        ],
    }

@app.on_event("startup")
def startup():
    validate_security_configuration()
    init_db()

def _resolve_case_id(case_id: str = None) -> str:
    if case_id:
        return case_id
    return f"case_{uuid4().hex[:16]}"

@app.post("/predict", response_model=PredictResponse)
def api_predict(
    inp: PatientInput,
    request: Request,
    user_id: str = Query(None, min_length=1, max_length=128),
    case_id: str = Query(None, min_length=1, max_length=128),
):
    resolved_user_id = resolve_user_id(request, user_id)
    resolved_case_id = _resolve_case_id(case_id)
    enforce_rate_limit(resolved_user_id, "predict")
    ensure_user(resolved_user_id)
    log_user_activity(resolved_user_id, "predict", case_id=resolved_case_id)
    risk, flagged, thr = predict(inp.model_dump())
    model_info = get_model_info()
    return PredictResponse(
        risk=risk,
        flagged=flagged,
        threshold=thr,
        case_id=resolved_case_id,
        model_version=model_info.get("model_version"),
    )

@app.post("/explain", response_model=ExplainResponse)
def api_explain(
    inp: PatientInput,
    request: Request,
    user_id: str = Query(None, min_length=1, max_length=128),
    case_id: str = Query(None, min_length=1, max_length=128),
):
    resolved_user_id = resolve_user_id(request, user_id)
    resolved_case_id = _resolve_case_id(case_id)
    enforce_rate_limit(resolved_user_id, "explain")
    ensure_user(resolved_user_id)
    log_user_activity(resolved_user_id, "explain", case_id=resolved_case_id)
    return explain(inp.model_dump(), user_id=resolved_user_id, case_id=resolved_case_id)

@app.post("/feedback")
def api_feedback(req: FeedbackRequest, request: Request):
    resolved_user_id = resolve_user_id(request, req.user_id)
    enforce_rate_limit(resolved_user_id, "feedback")
    ensure_user(resolved_user_id)
    insert_feedback(
        user_id=resolved_user_id,
        feedback_type=req.feedback_type,
        feature_name=req.feature_name,
        case_id=req.case_id,
        message=req.message,
    )

    # Auto preference updates
    if req.feedback_type in ("prefer_short", "prefer_long"):
        apply_preference_from_feedback(resolved_user_id, req.feedback_type)

    return {"status": "ok"}

@app.post("/preferences")
def api_set_prefs(req: PreferenceRequest, request: Request):
    resolved_user_id = resolve_user_id(request, req.user_id)
    enforce_rate_limit(resolved_user_id, "preferences_write")
    ensure_user(resolved_user_id)
    upsert_preferences(resolved_user_id, req.top_k, req.style)
    return {"status": "ok"}

@app.get("/preferences")
def api_get_prefs(
    request: Request,
    user_id: str = Query(None, min_length=1, max_length=128),
):
    resolved_user_id = resolve_user_id(request, user_id)
    enforce_rate_limit(resolved_user_id, "preferences_read")
    ensure_user(resolved_user_id)
    return get_preferences(resolved_user_id)

@app.get("/analytics/summary")
def api_analytics_summary(
    request: Request,
    user_id: str = Query(None, min_length=1, max_length=128),
):
    resolved_user_id = resolve_user_id(request, user_id)
    enforce_rate_limit(resolved_user_id, "analytics_summary")
    ensure_user(resolved_user_id)
    return {"summary": feedback_summary(user_id=resolved_user_id)}

@app.get("/analytics/top_features")
def api_top_features(
    request: Request,
    feedback_type: FeedbackType = Query(...),
    limit: int = Query(10, ge=1, le=100),
    user_id: str = Query(None, min_length=1, max_length=128),
):
    resolved_user_id = resolve_user_id(request, user_id)
    enforce_rate_limit(resolved_user_id, "analytics_top_features")
    ensure_user(resolved_user_id)
    return {
        "feedback_type": feedback_type,
        "top_features": top_features_by_feedback(feedback_type, limit, resolved_user_id),
    }

@app.get("/model/info")
def api_model_info():
    return get_model_info()

@app.get("/__build")
def build_info():
    return {
        "service": "interactive-xai-api",
        "git_sha": os.getenv("GIT_SHA", "unknown"),
        "auth_required": os.getenv("AUTH_REQUIRED", "unset"),
    }
