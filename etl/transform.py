"""
Data Transformation & Cleaning Layer
Standardizes data from multiple sources before feature engineering.
"""
import logging
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)


def clean_employee_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize and validate employee master data."""
    df = df.copy()

    # Normalize string columns
    str_cols = ["department", "job_role", "job_level", "employment_type", "gender", "marital_status"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    # Enforce salary floor
    df["base_salary"] = pd.to_numeric(df["base_salary"], errors="coerce").clip(lower=0)

    # Parse hire_date
    df["hire_date"] = pd.to_datetime(df["hire_date"], errors="coerce")

    # Remove duplicate employee IDs (keep latest)
    df = df.drop_duplicates(subset=["emp_id"], keep="last")

    logger.info(f"Employee data cleaned: {len(df)} records")
    return df


def clean_telemetry_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate daily workload telemetry."""
    df = df.copy()

    df["log_date"] = pd.to_datetime(df["log_date"], errors="coerce")
    df = df.dropna(subset=["emp_id", "log_date"])

    # Clip unrealistic values
    df["hours_worked"] = pd.to_numeric(df["hours_worked"], errors="coerce").clip(0, 16)
    df["overtime_hours"] = pd.to_numeric(df["overtime_hours"], errors="coerce").clip(0, 8)
    df["tasks_closed"] = pd.to_numeric(df["tasks_closed"], errors="coerce").clip(0, 50).fillna(0)
    df["commits_count"] = pd.to_numeric(df["commits_count"], errors="coerce").clip(0, 100).fillna(0)

    # Boolean flags
    for flag_col in ["unplanned_absence_flag", "late_login_flag"]:
        if flag_col in df.columns:
            df[flag_col] = df[flag_col].map(
                {True: True, False: False, "True": True, "False": False, 1: True, 0: False}
            ).fillna(False)

    # Remove duplicates (emp_id + date)
    df = df.drop_duplicates(subset=["emp_id", "log_date"], keep="last")

    logger.info(f"Telemetry cleaned: {len(df)} records")
    return df


def clean_survey_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate engagement survey data."""
    df = df.copy()
    df["survey_date"] = pd.to_datetime(df["survey_date"], errors="coerce")

    score_cols = [
        "satisfaction_score", "work_life_balance_score", "manager_relationship_score",
        "career_growth_score", "compensation_fairness_score", "team_cohesion_score",
    ]
    for col in score_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(1, 5)

    df = df.dropna(subset=["emp_id", "survey_date"])
    logger.info(f"Survey data cleaned: {len(df)} records")
    return df


def clean_performance_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate performance review data."""
    df = df.copy()
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")

    for col in ["performance_rating", "peer_review_score", "manager_rating"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(1, 5)

    df["projects_completed"] = pd.to_numeric(df["projects_completed"], errors="coerce").clip(0).fillna(0)
    df["promotions_count"] = pd.to_numeric(df["promotions_count"], errors="coerce").clip(0).fillna(0)
    df["training_hours"] = pd.to_numeric(df["training_hours"], errors="coerce").clip(0).fillna(0)

    df = df.dropna(subset=["emp_id", "review_date"])
    logger.info(f"Performance data cleaned: {len(df)} records")
    return df


def encode_categoricals(df: pd.DataFrame, cat_cols: Optional[list] = None) -> pd.DataFrame:
    """One-hot encode categorical columns for ML input."""
    if cat_cols is None:
        cat_cols = ["department", "job_role", "job_level", "employment_type", "marital_status"]

    available = [c for c in cat_cols if c in df.columns]
    df = pd.get_dummies(df, columns=available, drop_first=False, dtype=int)
    logger.info(f"Encoded categoricals: {available}")
    return df
