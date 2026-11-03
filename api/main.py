"""
FastAPI REST Layer
Exposes prediction endpoints for real-time flight risk scoring and intervention recommendations.
"""
from contextlib import asynccontextmanager
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import xgboost as xgb

from .schemas import (
    EmployeeFeatureRequest,
    RiskScoreResponse,
    BurnoutResponse,
    InterventionResponse,
    HealthResponse,
)

logger = logging.getLogger(__name__)

# Global model cache
_model = None
_explainer = None
_expected_features = []


def _get_model():
    """Lazy load XGBoost model."""
    global _model, _expected_features
    if _model is None:
        model_path = Path("./models/artifacts/attrition_model.json")
        if model_path.exists():
            m = xgb.XGBClassifier()
            m.load_model(str(model_path))
            _expected_features = list(getattr(m, "feature_names_in_", []))
            _model = m
            logger.info("Attrition model loaded successfully.")
        else:
            logger.warning(f"Model not found at {model_path}")
    return _model


def _get_explainer():
    """Lazy load SHAP TreeExplainer on first prediction demand."""
    global _explainer
    m = _get_model()
    if _explainer is None and m is not None:
        try:
            import shap
            _explainer = shap.TreeExplainer(m)
            logger.info("SHAP explainer initialized.")
        except Exception as e:
            logger.warning(f"Could not initialize SHAP explainer: {e}")
    return _explainer


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure model file exists on startup."""
    _get_model()
    yield


app = FastAPI(
    title="Workforce Attrition Analytics API",
    description="Real-time attrition risk scoring and burnout detection for HR teams",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _prepare_feature_frame(feature_dicts: list[dict]) -> pd.DataFrame:
    """Align input dictionaries to the exact columns expected by the XGBoost model."""
    df = pd.DataFrame(feature_dicts)
    if _expected_features:
        for col in _expected_features:
            if col not in df.columns:
                df[col] = 0.0
        df = df[_expected_features]
    return df.fillna(0.0)


@app.get("/health", response_model=HealthResponse)
def health_check():
    """API health check — returns model load status."""
    return HealthResponse(
        status="ok",
        model_loaded=_get_model() is not None,
        burnout_detector_loaded=True,
    )


@app.post("/predict/attrition", response_model=RiskScoreResponse)
def predict_attrition(request: EmployeeFeatureRequest):
    """
    Predict flight risk probability for a single employee.
    Returns risk tier (HIGH/MEDIUM/LOW) and top SHAP-based risk drivers.
    """
    model = _get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded — train and save first.")

    from models.model import predict_risk_scores, get_top_risk_drivers

    X = _prepare_feature_frame([request.features])
    risk_df = predict_risk_scores(model, X)

    drivers = []
    explainer = _get_explainer()
    if explainer is not None:
        try:
            drivers = get_top_risk_drivers(explainer, X, top_n=5)
        except Exception as e:
            logger.warning(f"Error computing SHAP drivers: {e}")

    return RiskScoreResponse(
        emp_id=request.emp_id,
        flight_risk_probability=float(risk_df["flight_risk_probability"].iloc[0]),
        risk_tier=str(risk_df["risk_tier"].iloc[0]),
        high_risk_flag=bool(risk_df["high_risk_flag"].iloc[0]),
        top_risk_drivers=drivers,
    )


@app.post("/predict/batch", response_model=list[RiskScoreResponse])
def predict_batch(requests: list[EmployeeFeatureRequest]):
    """Batch predict for multiple employees (max 500 per call)."""
    if len(requests) > 500:
        raise HTTPException(status_code=400, detail="Max 500 employees per batch.")
    model = _get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    from models.model import predict_risk_scores

    X = _prepare_feature_frame([r.features for r in requests])
    risk_df = predict_risk_scores(model, X)

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
def get_interventions(request: EmployeeFeatureRequest):
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


