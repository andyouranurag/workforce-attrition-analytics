"""
Shared pytest fixtures for the workforce attrition analytics test suite.
"""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_employees():
    """Minimal employee DataFrame for testing."""
    return pd.DataFrame({
        "emp_id": [f"EMP{i:04d}" for i in range(1, 21)],
        "hire_date": pd.date_range("2020-01-01", periods=20, freq="3ME"),
        "department": ["Engineering"] * 10 + ["Sales"] * 10,
        "job_role": ["Software Engineer"] * 10 + ["Sales Rep"] * 10,
        "job_level": ["Mid"] * 10 + ["Junior"] * 10,
        "base_salary": np.random.randint(500000, 1200000, 20),
        "manager_id": [None] * 20,
        "distance_from_home": np.random.uniform(1, 50, 20),
        "education_level": ["Bachelor"] * 20,
        "employment_type": ["Full-Time"] * 20,
        "gender": ["Male"] * 10 + ["Female"] * 10,
        "marital_status": ["Single"] * 20,
        "location": ["Bangalore"] * 20,
    })


@pytest.fixture
def sample_telemetry(sample_employees):
    """Minimal workload telemetry DataFrame."""
    rows = []
    for emp_id in sample_employees["emp_id"]:
        for day_offset in range(30):
            rows.append({
                "emp_id": emp_id,
                "log_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day_offset),
                "hours_worked": np.random.uniform(6, 10),
                "overtime_hours": np.random.uniform(0, 3),
                "tasks_closed": np.random.randint(0, 8),
                "tasks_opened": np.random.randint(0, 6),
                "commits_count": np.random.randint(0, 10),
                "tickets_resolved": np.random.randint(0, 5),
                "unplanned_absence_flag": np.random.random() < 0.05,
                "late_login_flag": np.random.random() < 0.08,
                "badge_swipe_count": np.random.randint(0, 4),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_surveys(sample_employees):
    """Minimal engagement survey DataFrame."""
    rows = []
    for emp_id in sample_employees["emp_id"]:
        rows.append({
            "emp_id": emp_id,
            "survey_date": pd.Timestamp("2024-06-01"),
            "satisfaction_score": np.random.uniform(1, 5),
            "work_life_balance_score": np.random.uniform(1, 5),
            "manager_relationship_score": np.random.uniform(1, 5),
            "career_growth_score": np.random.uniform(1, 5),
            "compensation_fairness_score": np.random.uniform(1, 5),
            "team_cohesion_score": np.random.uniform(1, 5),
            "survey_type": "pulse",
        })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_performance(sample_employees):
    """Minimal performance review DataFrame."""
    rows = []
    for emp_id in sample_employees["emp_id"]:
        rows.append({
            "emp_id": emp_id,
            "review_date": pd.Timestamp("2024-06-30"),
            "performance_rating": np.random.uniform(2, 5),
            "projects_completed": np.random.randint(1, 10),
            "promotions_count": np.random.randint(0, 2),
            "training_hours": np.random.uniform(0, 60),
            "peer_review_score": np.random.uniform(2, 5),
            "manager_rating": np.random.uniform(2, 5),
        })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_attrition(sample_employees):
    """Minimal attrition labels DataFrame."""
    return pd.DataFrame({
        "emp_id": sample_employees["emp_id"],
        "left_company_flag": [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        "exit_date": [None] * 20,
        "voluntary_flag": [None] * 20,
    })

