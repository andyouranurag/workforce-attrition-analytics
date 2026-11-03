"""
Unit tests for the feature engineering pipeline.
"""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from etl.features import generate_workforce_features, get_feature_columns
from etl.transform import (
    clean_employee_data,
    clean_telemetry_data,
    clean_survey_data,
    clean_performance_data,
    encode_categoricals,
)


class TestGenerateWorkforceFeatures:

    def test_output_has_all_employees(
        self, sample_employees, sample_telemetry, sample_surveys, sample_performance
    ):
        result = generate_workforce_features(
            sample_employees, sample_telemetry, sample_surveys, sample_performance
        )
        assert len(result) == len(sample_employees)

    def test_tenure_years_calculated(
        self, sample_employees, sample_telemetry, sample_surveys, sample_performance
    ):
        result = generate_workforce_features(
            sample_employees, sample_telemetry, sample_surveys, sample_performance
        )
        assert "tenure_years" in result.columns
        assert (result["tenure_years"] >= 0).all()

    def test_burnout_risk_score_in_range(
        self, sample_employees, sample_telemetry, sample_surveys, sample_performance
    ):
        result = generate_workforce_features(
            sample_employees, sample_telemetry, sample_surveys, sample_performance
        )
        assert "burnout_risk_score" in result.columns
        assert (result["burnout_risk_score"] >= 0).all()
        assert (result["burnout_risk_score"] <= 100).all()

    def test_composite_engagement_score_calculated(
        self, sample_employees, sample_telemetry, sample_surveys, sample_performance
    ):
        result = generate_workforce_features(
            sample_employees, sample_telemetry, sample_surveys, sample_performance
        )
        assert "composite_engagement_score" in result.columns

    def test_no_null_numeric_columns(
        self, sample_employees, sample_telemetry, sample_surveys, sample_performance
    ):
        result = generate_workforce_features(
            sample_employees, sample_telemetry, sample_surveys, sample_performance
        )
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        assert result[numeric_cols].isnull().sum().sum() == 0, \
            "Numeric columns should have no nulls after feature engineering"

    def test_high_overtime_flag_binary(
        self, sample_employees, sample_telemetry, sample_surveys, sample_performance
    ):
        result = generate_workforce_features(
            sample_employees, sample_telemetry, sample_surveys, sample_performance
        )
        assert set(result["high_overtime_flag"].unique()).issubset({0, 1})

    def test_salary_percentile_in_range(
        self, sample_employees, sample_telemetry, sample_surveys, sample_performance
    ):
        result = generate_workforce_features(
            sample_employees, sample_telemetry, sample_surveys, sample_performance
        )
        assert "dept_level_salary_pct" in result.columns
        valid = result["dept_level_salary_pct"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()


class TestGetFeatureColumns:

    def test_returns_list(self):
        cols = get_feature_columns()
        assert isinstance(cols, list)
        assert len(cols) > 0

    def test_no_duplicates(self):
        cols = get_feature_columns()
        assert len(cols) == len(set(cols))


class TestCleanEmployeeData:

    def test_removes_duplicates(self):
        df = pd.DataFrame({
            "emp_id": ["EMP001", "EMP001", "EMP002"],
            "hire_date": ["2021-01-01", "2021-01-01", "2020-06-15"],
            "department": ["Engineering", "Engineering", "Sales"],
            "job_role": ["Engineer", "Engineer", "Sales Rep"],
            "job_level": ["Mid", "Mid", "Junior"],
            "employment_type": ["Full-Time", "Full-Time", "Full-Time"],
            "gender": ["Male", "Male", "Female"],
            "marital_status": ["Single", "Single", "Married"],
            "base_salary": [700000, 700000, 500000],
        })
        result = clean_employee_data(df)
        assert result["emp_id"].nunique() == 2

    def test_salary_clipped_at_zero(self):
        df = pd.DataFrame({
            "emp_id": ["EMP001"],
            "hire_date": ["2021-01-01"],
            "department": ["Engineering"],
            "job_role": ["Engineer"],
            "job_level": ["Mid"],
            "employment_type": ["Full-Time"],
            "gender": ["Male"],
            "marital_status": ["Single"],
            "base_salary": [-50000],
        })
        result = clean_employee_data(df)
        assert result["base_salary"].iloc[0] == 0


class TestCleanTelemetryData:

    def test_hours_worked_clipped(self):
        df = pd.DataFrame({
            "emp_id": ["EMP001"],
            "log_date": ["2024-01-15"],
            "hours_worked": [25.0],  # Unrealistic
            "overtime_hours": [0],
            "tasks_closed": [3],
            "commits_count": [2],
            "unplanned_absence_flag": [False],
            "late_login_flag": [False],
        })
        result = clean_telemetry_data(df)
        assert result["hours_worked"].iloc[0] <= 16

    def test_drops_missing_emp_id(self):
        df = pd.DataFrame({
            "emp_id": [None, "EMP001"],
            "log_date": ["2024-01-15", "2024-01-16"],
            "hours_worked": [8, 8],
            "overtime_hours": [1, 1],
            "tasks_closed": [3, 3],
            "commits_count": [2, 2],
            "unplanned_absence_flag": [False, False],
            "late_login_flag": [False, False],
        })
        result = clean_telemetry_data(df)
        assert len(result) == 1


class TestEncodeCategoricals:

    def test_creates_dummy_columns(self, sample_employees):
        result = encode_categoricals(sample_employees.copy(), ["department"])
        assert "department_Engineering" in result.columns or \
               any("department_" in c for c in result.columns)

    def test_original_col_removed(self, sample_employees):
        result = encode_categoricals(sample_employees.copy(), ["department"])
        assert "department" not in result.columns
