from pydantic import BaseModel, Field, model_validator
from typing import Optional, Dict, Any, List, Literal

FeatureName = Literal[
    "male",
    "age",
    "education",
    "currentSmoker",
    "cigsPerDay",
    "BPMeds",
    "prevalentStroke",
    "prevalentHyp",
    "diabetes",
    "totChol",
    "sysBP",
    "diaBP",
    "BMI",
    "heartRate",
    "glucose",
]

FeedbackType = Literal[
    "relevant",
    "irrelevant",
    "confusing",
    "prefer_short",
    "prefer_long",
]

StyleType = Literal["simple", "detailed"]

class PatientInput(BaseModel):
    male: int = Field(..., ge=0, le=1)
    age: float = Field(..., ge=18, le=110)
    education: Optional[int] = Field(default=None, ge=1, le=4)
    currentSmoker: int = Field(..., ge=0, le=1)
    cigsPerDay: Optional[float] = Field(default=None, ge=0, le=150)
    BPMeds: Optional[int] = Field(default=None, ge=0, le=1)
    prevalentStroke: int = Field(..., ge=0, le=1)
    prevalentHyp: int = Field(..., ge=0, le=1)
    diabetes: int = Field(..., ge=0, le=1)
    totChol: Optional[float] = Field(default=None, ge=50, le=1000)
    sysBP: float = Field(..., ge=60, le=300)
    diaBP: float = Field(..., ge=40, le=200)
    BMI: Optional[float] = Field(default=None, ge=10, le=80)
    heartRate: Optional[float] = Field(default=None, ge=30, le=250)
    glucose: Optional[float] = Field(default=None, ge=30, le=600)

class PredictResponse(BaseModel):
    risk: float
    threshold: float
    flagged: bool
    case_id: str
    model_version: Optional[str] = None

class ShapItem(BaseModel):
    feature: str
    value: Any
    shap: float
    disputed: bool = False

class ExplainResponse(BaseModel):
    risk: float
    threshold: float
    flagged: bool
    case_id: str
    top_positive: List[ShapItem]
    top_negative: List[ShapItem]
    disputed_features: List[str]
    meta: Dict[str, Any]
    hidden_contributors: List[ShapItem]

class FeedbackRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=128)
    feedback_type: FeedbackType
    feature_name: Optional[FeatureName] = None
    case_id: Optional[str] = Field(default=None, min_length=1, max_length=128)
    message: Optional[str] = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def validate_feature_requirement(self):
        needs_feature = self.feedback_type in ("relevant", "irrelevant", "confusing")
        if needs_feature and self.feature_name is None:
            raise ValueError("feature_name is required for relevant/irrelevant/confusing feedback.")
        if needs_feature and self.case_id is None:
            raise ValueError("case_id is required for relevant/irrelevant/confusing feedback.")
        return self

class PreferenceRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=128)
    top_k: int = Field(default=8, ge=1, le=20)
    style: StyleType = "simple"
