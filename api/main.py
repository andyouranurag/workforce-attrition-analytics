"""
FastAPI REST Layer
Exposes prediction endpoints for real-time flight risk scoring and intervention recommendations.
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import logging

from .schemas import (
    EmployeeFeatureRequest,
    RiskScoreResponse,
    BurnoutResponse,
    InterventionResponse,
    HealthResponse,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Workforce Attrition Analytics API",
    description="Real-time attrition risk scoring and burnout detection for HR teams",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production to Superset/BI tool domains
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Global model cache (loaded at startup)
_model = None
_explainer = None
_burnout_detector = None
_scaler = None


@app.on_event("startup")
async def load_models():
    """Load trained models on API startup."""
    global _model, _explainer, _burnout_detector, _scaler
    import xgboost as xgb
    import shap
    from pathlib import Path

    model_path = "./models/artifacts/attrition_model.json"
    if Path(model_path).exists():
        _model = xgb.XGBClassifier()
        _model.load_model(model_path)
        _explainer = shap.TreeExplainer(_model)
        logger.info("Attrition model loaded successfully")
    else:
        logger.warning(f"Model not found at {model_path} — run train_pipeline.py first")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """API health check — returns model load status."""
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
        burnout_detector_loaded=_burnout_detector is not None,
    )


@app.post("/predict/attrition", response_model=RiskScoreResponse)
async def predict_attrition(request: EmployeeFeatureRequest):
    """
    Predict flight risk probability for a single employee.
    Returns risk tier (HIGH/MEDIUM/LOW) and top SHAP-based risk drivers.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded — train and save first.")

    import pandas as pd
    from models.model import predict_risk_scores, get_top_risk_drivers

    X = pd.DataFrame([request.features])
    risk_df = predict_risk_scores(_model, X)

    drivers = []
    if _explainer is not None:
        drivers = get_top_risk_drivers(_explainer, X, top_n=5)

    return RiskScoreResponse(
        emp_id=request.emp_id,
        flight_risk_probability=float(risk_df["flight_risk_probability"].iloc[0]),
        risk_tier=str(risk_df["risk_tier"].iloc[0]),
        high_risk_flag=bool(risk_df["high_risk_flag"].iloc[0]),
        top_risk_drivers=drivers,
    )


@app.post("/predict/batch", response_model=list[RiskScoreResponse])
async def predict_batch(requests: list[EmployeeFeatureRequest]):
    """Batch predict for multiple employees (max 500 per call)."""
    if len(requests) > 500:
        raise HTTPException(status_code=400, detail="Max 500 employees per batch.")
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    import pandas as pd
    from models.model import predict_risk_scores

    rows = [r.features for r in requests]
    X = pd.DataFrame(rows)
    risk_df = predict_risk_scores(_model, X)

    results = []
    for i, req in enumerate(requests):
        results.append(RiskScoreResponse(
            emp_id=req.emp_id,
            flight_risk_probability=float(risk_df["flight_risk_probability"].iloc[i]),
            risk_tier=str(risk_df["risk_tier"].iloc[i]),
            high_risk_flag=bool(risk_df["high_risk_flag"].iloc[i]),
            top_risk_drivers=[],
        ))
    return results


@app.post("/interventions", response_model=InterventionResponse)
async def get_interventions(request: EmployeeFeatureRequest):
    """
    Generate HR intervention recommendations for an employee.
    Uses rule-based engine on feature profile.
    """
    from models.model import generate_interventions
    interventions = generate_interventions(request.features)
    return InterventionResponse(
        emp_id=request.emp_id,
        interventions=interventions,
        total_count=len(interventions),
    )
