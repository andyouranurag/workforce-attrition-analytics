"""
Unit tests for the ML model — attrition classifier and burnout detector.
"""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.model import (
    predict_risk_scores,
    generate_interventions,
    get_top_risk_drivers,
    train_burnout_detector,
)


def make_feature_df(n=100):
    """Generate a minimal numeric feature DataFrame for model testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "tenure_years": np.random.uniform(0, 15, n),
        "distance_from_home": np.random.uniform(1, 80, n),
        "base_salary": np.random.randint(300000, 2000000, n),
        "dept_level_salary_pct": np.random.uniform(0, 1, n),
        "avg_overtime_90d": np.random.uniform(0, 5, n),
        "absence_trend_90d": np.random.randint(0, 15, n),
        "avg_hours_worked": np.random.uniform(5, 12, n),
        "avg_tasks_closed": np.random.uniform(0, 8, n),
        "avg_commits": np.random.uniform(0, 10, n),
        "late_login_pct": np.random.uniform(0, 0.5, n),
        "high_overtime_flag": np.random.randint(0, 2, n),
        "burnout_risk_score": np.random.uniform(0, 100, n),
        "satisfaction_score": np.random.uniform(1, 5, n),
        "work_life_balance_score": np.random.uniform(1, 5, n),
        "manager_relationship_score": np.random.uniform(1, 5, n),
        "career_growth_score": np.random.uniform(1, 5, n),
        "compensation_fairness_score": np.random.uniform(1, 5, n),
        "composite_engagement_score": np.random.uniform(1, 5, n),
        "performance_rating": np.random.uniform(1, 5, n),
        "projects_completed": np.random.randint(0, 12, n),
        "promotions_count": np.random.randint(0, 3, n),
        "training_hours": np.random.uniform(0, 80, n),
        "peer_review_score": np.random.uniform(1, 5, n),
        "stagnation_flag": np.random.randint(0, 2, n),
        "underpaid_flag": np.random.randint(0, 2, n),
    })


class TestTrainAttritionModel:

    def test_model_trains_without_error(self):
        """Training should complete on small synthetic dataset."""
        import xgboost as xgb
        from models.model import train_attrition_model

        X = make_feature_df(80)
        y = pd.Series(np.random.randint(0, 2, 80))
        # Ensure at least some positives for stratified split
        y.iloc[:16] = 1
        y.iloc[16:] = 0

        model, explainer, shap_values, metrics = train_attrition_model(X, y, n_folds=2)

        assert isinstance(model, xgb.XGBClassifier)
        assert "roc_auc" in metrics
        assert 0 <= metrics["roc_auc"] <= 1

    def test_metrics_contain_required_keys(self):
        from models.model import train_attrition_model

        X = make_feature_df(80)
        y = pd.Series([1] * 16 + [0] * 64)

        _, _, _, metrics = train_attrition_model(X, y, n_folds=2)
        for key in ["roc_auc", "average_precision", "cv_auc_mean", "confusion_matrix"]:
            assert key in metrics, f"Missing metric: {key}"


class TestPredictRiskScores:

    @pytest.fixture
    def trained_model(self):
        from models.model import train_attrition_model
        X = make_feature_df(80)
        y = pd.Series([1] * 16 + [0] * 64)
        model, _, _, _ = train_attrition_model(X, y, n_folds=2)
        return model

    def test_returns_correct_columns(self, trained_model):
        X = make_feature_df(10)
        result = predict_risk_scores(trained_model, X)
        assert "flight_risk_probability" in result.columns
        assert "risk_tier" in result.columns
        assert "high_risk_flag" in result.columns

    def test_probabilities_in_range(self, trained_model):
        X = make_feature_df(50)
        result = predict_risk_scores(trained_model, X)
        assert (result["flight_risk_probability"] >= 0).all()
        assert (result["flight_risk_probability"] <= 1).all()

    def test_risk_tiers_valid(self, trained_model):
        X = make_feature_df(50)
        result = predict_risk_scores(trained_model, X)
        assert set(result["risk_tier"].unique()).issubset({"LOW", "MEDIUM", "HIGH"})

    def test_high_risk_flag_matches_threshold(self, trained_model):
        X = make_feature_df(50)
        result = predict_risk_scores(trained_model, X, threshold=0.75)
        expected = (result["flight_risk_probability"] >= 0.75).astype(int)
        pd.testing.assert_series_equal(result["high_risk_flag"], expected, check_names=False)

    def test_row_count_matches_input(self, trained_model):
        X = make_feature_df(37)
        result = predict_risk_scores(trained_model, X)
        assert len(result) == 37


class TestGenerateInterventions:

    def test_high_overtime_triggers_intervention(self):
        row = {"avg_overtime_90d": 4.5}
        interventions = generate_interventions(row)
        assert any("overtime" in i["intervention"].lower() for i in interventions)

    def test_low_satisfaction_triggers_intervention(self):
        row = {"satisfaction_score": 1.8}
        interventions = generate_interventions(row)
        assert any("satisfaction" in i["intervention"].lower() for i in interventions)

    def test_no_triggers_returns_empty(self):
        row = {
            "avg_overtime_90d": 0.5,
            "absence_trend_90d": 0,
            "satisfaction_score": 4.5,
            "compensation_fairness_score": 4.0,
            "promotions_count": 1,
            "tenure_years": 2,
            "career_growth_score": 4.0,
            "manager_relationship_score": 4.5,
        }
        interventions = generate_interventions(row)
        assert interventions == []

    def test_multiple_triggers(self):
        row = {
            "avg_overtime_90d": 5.0,
            "absence_trend_90d": 8,
            "satisfaction_score": 1.5,
            "compensation_fairness_score": 1.8,
            "promotions_count": 0,
            "tenure_years": 4,
            "career_growth_score": 1.5,
            "manager_relationship_score": 1.0,
        }
        interventions = generate_interventions(row)
        assert len(interventions) > 3

    def test_intervention_has_required_keys(self):
        row = {"avg_overtime_90d": 4.0}
        interventions = generate_interventions(row)
        if interventions:
            assert "intervention" in interventions[0]
            assert "priority" in interventions[0]


class TestBurnoutDetector:

    def test_trains_and_returns_components(self):
        df = pd.DataFrame({
            "avg_overtime_90d": np.random.uniform(0, 5, 100),
            "absence_trend_90d": np.random.randint(0, 10, 100),
            "avg_hours_worked": np.random.uniform(6, 12, 100),
            "late_login_pct": np.random.uniform(0, 0.3, 100),
            "burnout_risk_score": np.random.uniform(0, 100, 100),
        })
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        detector, scaler = train_burnout_detector(df)
        assert isinstance(detector, IsolationForest)
        assert isinstance(scaler, StandardScaler)
