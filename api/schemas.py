"""
Pydantic schemas for the FastAPI endpoints.
"""
from pydantic import BaseModel, Field
from typing import Optional


class EmployeeFeatureRequest(BaseModel):
    """Request schema for a single employee risk prediction."""
    emp_id: str = Field(..., description="Employee ID (anonymized)")
    features: dict[str, float] = Field(
        ...,
        description="Feature dict matching the model's expected input columns",
        example={
            "tenure_years": 3.5,
            "avg_overtime_90d": 2.8,
            "absence_trend_90d": 4,
            "satisfaction_score": 2.5,
            "work_life_balance_score": 2.0,
            "manager_relationship_score": 3.0,
            "career_growth_score": 2.0,
            "compensation_fairness_score": 2.5,
            "composite_engagement_score": 2.4,
            "performance_rating": 3.2,
            "promotions_count": 0,
            "stagnation_flag": 1,
            "burnout_risk_score": 65.0,
            "underpaid_flag": 1,
        }
    )


class RiskDriver(BaseModel):
    feature: str
    value: float
    shap_impact: float
    direction: str  # "increases_risk" or "reduces_risk"


class RiskScoreResponse(BaseModel):
    emp_id: str
    flight_risk_probability: float = Field(..., ge=0, le=1)
    risk_tier: str  # "HIGH", "MEDIUM", "LOW"
    high_risk_flag: bool
    top_risk_drivers: list[RiskDriver]


class BurnoutResponse(BaseModel):
    emp_id: str
    burnout_anomaly_score: float
    burnout_flag: bool


class InterventionItem(BaseModel):
    intervention: str
    priority: str  # "HIGH", "MEDIUM", "LOW"


class InterventionResponse(BaseModel):
    emp_id: str
    interventions: list[InterventionItem]
    total_count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    burnout_detector_loaded: bool
