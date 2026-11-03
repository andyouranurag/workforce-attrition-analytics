"""
Feature Engineering Pipeline
Aggregates raw telemetry and survey metrics into ML-ready feature vectors.
Generates rolling behavioral windows to feed the attrition prediction model.
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from .config import PIPELINE_CONFIG

logger = logging.getLogger(__name__)


def generate_workforce_features(
    df_employee: pd.DataFrame,
    df_telemetry: pd.DataFrame,
    df_survey: pd.DataFrame,
    df_performance: pd.DataFrame,
    lookback_days: int = PIPELINE_CONFIG.lookback_days,
) -> pd.DataFrame:
    """
    Aggregates raw telemetry and survey metrics into rolling behavioral windows
    to feed the attrition prediction model.

    Args:
        df_employee:   dim_employee records
        df_telemetry:  fact_workload_telemetry records
        df_survey:     fact_engagement_survey records
        df_performance: fact_performance records
        lookback_days: Rolling window size for telemetry aggregation

    Returns:
        master_df: Wide feature DataFrame ready for model training
    """
    logger.info("Building workforce feature set...")

    # -- 1. Parse dates -------------------------------------------------------
    df_telemetry = df_telemetry.copy()
    df_telemetry["log_date"] = pd.to_datetime(df_telemetry["log_date"], errors="coerce")

    df_survey = df_survey.copy()
    df_survey["survey_date"] = pd.to_datetime(df_survey["survey_date"], errors="coerce")

    df_performance = df_performance.copy()
    df_performance["review_date"] = pd.to_datetime(df_performance["review_date"], errors="coerce")

    df_employee = df_employee.copy()
    df_employee["hire_date"] = pd.to_datetime(df_employee["hire_date"], errors="coerce")

    # -- 2. Telemetry Aggregation (rolling window) ----------------------------
    cutoff = df_telemetry["log_date"].max() - pd.Timedelta(days=lookback_days)
    recent_telemetry = df_telemetry[df_telemetry["log_date"] >= cutoff]

    telemetry_agg = recent_telemetry.groupby("emp_id").agg(
        avg_overtime_90d=("overtime_hours", "mean"),
        absence_trend_90d=("unplanned_absence_flag", "sum"),
        avg_hours_worked=("hours_worked", "mean"),
        avg_tasks_closed=("tasks_closed", "mean"),
        avg_commits=("commits_count", "mean"),
        late_login_pct=("late_login_flag", "mean"),
        total_workload_days=("log_date", "count"),
    ).reset_index()

    # Overtime intensity: flag if avg overtime > 2 hours consistently
    telemetry_agg["high_overtime_flag"] = (telemetry_agg["avg_overtime_90d"] > 2.0).astype(int)

    # Burnout risk: high overtime + high absences
    telemetry_agg["burnout_risk_score"] = (
        telemetry_agg["avg_overtime_90d"].clip(0, 5) / 5 * 50 +
        telemetry_agg["absence_trend_90d"].clip(0, 10) / 10 * 50
    ).round(2)

    logger.info(f"  Telemetry aggregated for {len(telemetry_agg)} employees")

    # -- 3. Survey: latest scores per employee --------------------------------
    latest_survey = (
        df_survey.sort_values("survey_date")
        .groupby("emp_id")
        .last()
        .reset_index()
    )[
        [
            "emp_id",
            "satisfaction_score",
            "work_life_balance_score",
            "manager_relationship_score",
            "career_growth_score",
            "compensation_fairness_score",
            "team_cohesion_score",
        ]
    ]

    # Composite engagement score (equal weights)
    score_cols = [
        "satisfaction_score", "work_life_balance_score",
        "manager_relationship_score", "career_growth_score",
        "compensation_fairness_score", "team_cohesion_score",
    ]
    latest_survey["composite_engagement_score"] = latest_survey[score_cols].mean(axis=1).round(2)

    logger.info(f"  Survey features built for {len(latest_survey)} employees")

    # -- 4. Performance: latest review per employee ---------------------------
    latest_perf = (
        df_performance.sort_values("review_date")
        .groupby("emp_id")
        .last()
        .reset_index()
    )[
        [
            "emp_id",
            "performance_rating",
            "projects_completed",
            "promotions_count",
            "training_hours",
            "peer_review_score",
            "manager_rating",
        ]
    ]

    # Promotion recency flag (promoted in last review cycle?)
    latest_perf["recently_promoted"] = (latest_perf["promotions_count"] > 0).astype(int)

    # Stagnation signal: no promotion + below-median performance
    perf_median = latest_perf["performance_rating"].median()
    latest_perf["stagnation_flag"] = (
        (latest_perf["promotions_count"] == 0) &
        (latest_perf["performance_rating"] < perf_median)
    ).astype(int)

    logger.info(f"  Performance features built for {len(latest_perf)} employees")

    # -- 5. Employee tenure ---------------------------------------------------
    reference_date = pd.Timestamp("today")
    df_employee["tenure_years"] = (
        (reference_date - df_employee["hire_date"]).dt.days / 365.25
    ).round(2)

    # Tenure risk bands: very new (<1yr) or mid-career (2-5yr) are higher risk
    df_employee["tenure_risk_band"] = pd.cut(
        df_employee["tenure_years"],
        bins=[-np.inf, 1, 3, 5, 10, np.inf],
        labels=["<1yr", "1-3yr", "3-5yr", "5-10yr", "10yr+"],
    )

    # -- 6. Merge all features -----------------------------------------------
    master_df = df_employee.merge(telemetry_agg, on="emp_id", how="left")
    master_df = master_df.merge(latest_survey, on="emp_id", how="left")
    master_df = master_df.merge(latest_perf, on="emp_id", how="left")

    # -- 7. Derived salary features -------------------------------------------
    # Salary percentile within department/level
    master_df["dept_level_salary_pct"] = master_df.groupby(
        ["department", "job_level"]
    )["base_salary"].rank(pct=True).round(3)

    # Underpaid flag: bottom 25% in department
    master_df["underpaid_flag"] = (master_df["dept_level_salary_pct"] < 0.25).astype(int)

    # -- 8. Fill missing values -----------------------------------------------
    numeric_cols = master_df.select_dtypes(include=[np.number]).columns
    master_df[numeric_cols] = master_df[numeric_cols].fillna(master_df[numeric_cols].median())

    logger.info(f"Feature set built: {len(master_df)} employees × {len(master_df.columns)} features")
    return master_df


def get_feature_columns() -> list[str]:
    """Returns the canonical list of model input features."""
    return [
        # Demographic
        "tenure_years",
        "distance_from_home",
        "base_salary",
        "dept_level_salary_pct",
        # Workload / Behavioral
        "avg_overtime_90d",
        "absence_trend_90d",
        "avg_hours_worked",
        "avg_tasks_closed",
        "avg_commits",
        "late_login_pct",
        "high_overtime_flag",
        "burnout_risk_score",
        # Survey / Engagement
        "satisfaction_score",
        "work_life_balance_score",
        "manager_relationship_score",
        "career_growth_score",
        "compensation_fairness_score",
        "composite_engagement_score",
        # Performance
        "performance_rating",
        "projects_completed",
        "promotions_count",
        "training_hours",
        "peer_review_score",
        "stagnation_flag",
        "underpaid_flag",
    ]


def build_feature_store(
    raw_data_dir: str = "./data/raw",
    output_path: str = PIPELINE_CONFIG.feature_store_path,
) -> pd.DataFrame:
    """
    Loads raw parquet files from the ingestion step, runs feature engineering,
    and saves the resulting feature store.
    """
    raw = Path(raw_data_dir)
    logger.info(f"Loading raw data from {raw}...")

    df_employee    = pd.read_parquet(raw / "employees.parquet")
    df_telemetry   = pd.read_parquet(raw / "telemetry.parquet")
    df_survey      = pd.read_parquet(raw / "surveys.parquet")
    df_performance = pd.read_parquet(raw / "performance.parquet")
    df_attrition   = pd.read_parquet(raw / "attrition.parquet")

    master_df = generate_workforce_features(
        df_employee, df_telemetry, df_survey, df_performance
    )

    # Merge target variable
    df_attrition["left_company_flag"] = df_attrition["left_company_flag"].astype(int)
    master_df = master_df.merge(
        df_attrition[["emp_id", "left_company_flag"]],
        on="emp_id",
        how="left",
    )
    master_df["left_company_flag"] = master_df["left_company_flag"].fillna(0).astype(int)

    # Save feature store
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    master_df.to_parquet(out, index=False)
    logger.info(f"Feature store saved: {out} ({len(master_df)} rows)")

    return master_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = build_feature_store()
    print(df.describe())
