"""
Unit tests for API controller logic.
Direct functional tests avoid Windows TestClient socket permissions / loop deadlocks.
"""
import pytest
from api.main import (
    health_check,
    predict_attrition,
    get_interventions,
    _prepare_feature_frame,
)
from api.schemas import EmployeeFeatureRequest


def test_health_check():
    res = health_check()
    assert res.status == "ok"
    assert res.model_loaded is True


def test_prepare_feature_frame():
    sample = [{"tenure_years": 3.0, "avg_overtime_90d": 2.5}]
    df = _prepare_feature_frame(sample)
    assert len(df) == 1
    assert "tenure_years" in df.columns
    assert "avg_overtime_90d" in df.columns


def test_predict_attrition_logic():
    req = EmployeeFeatureRequest(
        emp_id="EMP0001",
        features={
            "tenure_years": 2.5,
            "avg_overtime_90d": 3.8,
            "absence_trend_90d": 4.0,
            "satisfaction_score": 2.1,
            "work_life_balance_score": 1.9,
            "base_salary": 750000.0,
        }
    )
    res = predict_attrition(req)
    assert res.emp_id == "EMP0001"
    assert 0.0 <= res.flight_risk_probability <= 1.0
    assert res.risk_tier in ["LOW", "MEDIUM", "HIGH"]
    assert isinstance(res.top_risk_drivers, list)


def test_interventions_logic():
    req = EmployeeFeatureRequest(
        emp_id="EMP0002",
        features={
            "avg_overtime_90d": 4.5,
            "absence_trend_90d": 6.0,
            "satisfaction_score": 1.8,
        }
    )
    res = get_interventions(req)
    assert res.emp_id == "EMP0002"
    assert res.total_count > 0
    assert len(res.interventions) > 0
