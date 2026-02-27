from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class PatientInput(BaseModel):
    male: int = Field(..., ge=0, le=1)
    age: float
    education: Optional[int] = None
    currentSmoker: int = Field(..., ge=0, le=1)
    cigsPerDay: Optional[float] = None
    BPMeds: Optional[int] = None
    prevalentStroke: int = Field(..., ge=0, le=1)
    prevalentHyp: int = Field(..., ge=0, le=1)
    diabetes: int = Field(..., ge=0, le=1)
    totChol: Optional[float] = None
    sysBP: float
    diaBP: float
    BMI: Optional[float] = None
    heartRate: Optional[float] = None
    glucose: Optional[float] = None

class PredictResponse(BaseModel):
    risk: float
    threshold: float
    flagged: bool

class ShapItem(BaseModel):
    feature: str
    value: Any
    shap: float

class ExplainResponse(BaseModel):
    risk: float
    threshold: float
    flagged: bool
    top_positive: List[ShapItem]
    top_negative: List[ShapItem]
    disputed_features: List[str]
    meta: Dict[str, Any]
    hidden_contributors: List[ShapItem]

class FeedbackRequest(BaseModel):
    user_id: str
    feedback_type: str  # e.g., relevant, irrelevant, confusing, prefer_short, prefer_long
    feature_name: Optional[str] = None
    case_id: Optional[str] = None
    message: Optional[str] = None

class PreferenceRequest(BaseModel):
    user_id: str
    top_k: int = 8
    style: str = "simple"  # simple|detailed