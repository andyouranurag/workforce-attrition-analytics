"""
End-to-End Training Pipeline
Orchestrates: data loading → feature engineering → model training → evaluation → save
"""
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from etl.features import build_feature_store, get_feature_columns
from etl.transform import encode_categoricals
from models.model import train_attrition_model, train_burnout_detector, predict_risk_scores
from models.evaluate import (
    plot_roc_curve,
    plot_precision_recall,
    plot_shap_summary,
    generate_evaluation_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


def run_training_pipeline(
    raw_data_dir: str = "./data/raw",
    model_save_path: str = "./models/artifacts/attrition_model.json",
):
    """Full training pipeline from raw data to saved model + evaluation reports."""

    # 1. Build feature store
    logger.info("=" * 60)
    logger.info("STEP 1: Building feature store")
    logger.info("=" * 60)
    master_df = build_feature_store(raw_data_dir)

    # 2. Encode categoricals
    logger.info("=" * 60)
    logger.info("STEP 2: Encoding categorical features")
    logger.info("=" * 60)
    master_df = encode_categoricals(master_df)

    # 3. Prepare X, y
    feature_cols = [c for c in get_feature_columns() if c in master_df.columns]
    X = master_df[feature_cols].select_dtypes(include="number")
    y = master_df["left_company_flag"].astype(int)

    logger.info(f"Training set: {X.shape[0]} samples × {X.shape[1]} features")
    logger.info(f"Attrition rate: {y.mean():.2%}")

    # 4. Train attrition classifier
    logger.info("=" * 60)
    logger.info("STEP 3: Training XGBoost attrition classifier")
    logger.info("=" * 60)
    model, explainer, shap_values, metrics = train_attrition_model(
        X, y,
        model_save_path=model_save_path,
    )

    # 5. Train burnout detector
    logger.info("=" * 60)
    logger.info("STEP 4: Training Isolation Forest burnout detector")
    logger.info("=" * 60)
    burnout_cols = [
        "avg_overtime_90d", "absence_trend_90d", "avg_hours_worked",
        "late_login_pct", "burnout_risk_score",
    ]
    available_burnout = [c for c in burnout_cols if c in master_df.columns]
    if available_burnout:
        detector, scaler = train_burnout_detector(master_df[available_burnout + ["emp_id"]].set_index("emp_id"))

    # 6. Generate evaluation artifacts
    logger.info("=" * 60)
    logger.info("STEP 5: Generating evaluation reports")
    logger.info("=" * 60)
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    y_probs = model.predict_proba(X_test)[:, 1]

    plot_roc_curve(y_test, y_probs)
    plot_precision_recall(y_test, y_probs)
    plot_shap_summary(shap_values, X_test)
    generate_evaluation_report(metrics)

    # 7. Generate risk score snapshot
    logger.info("=" * 60)
    logger.info("STEP 6: Generating risk score snapshot")
    logger.info("=" * 60)
    risk_df = predict_risk_scores(model, X)
    risk_df["emp_id"] = master_df["emp_id"].values
    out_path = Path("./data/processed/risk_scores.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    risk_df.to_csv(out_path, index=False)
    logger.info(f"Risk scores saved: {out_path}")
    logger.info(f"High risk employees: {risk_df['high_risk_flag'].sum()} / {len(risk_df)}")

    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info(f"ROC-AUC: {metrics['roc_auc']} | Avg Precision: {metrics['average_precision']}")
    logger.info("=" * 60)

    return model, metrics


if __name__ == "__main__":
    run_training_pipeline()
