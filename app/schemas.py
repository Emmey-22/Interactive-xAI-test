from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

FeedbackType = Literal["relevant", "irrelevant", "confusing", "prefer_short", "prefer_long"]
StyleType = Literal["simple", "detailed"]


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
    feedback_type: FeedbackType
    feature_name: Optional[str] = Field(default=None, min_length=1, max_length=64)
    case_id: Optional[str] = Field(default=None, min_length=1, max_length=128)
    message: Optional[str] = Field(default=None, max_length=1000)


class PreferenceRequest(BaseModel):
    top_k: int = Field(default=8, ge=1, le=10)
    style: StyleType = "simple"
