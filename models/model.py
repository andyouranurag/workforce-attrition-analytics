"""
Attrition Prediction Model
Dual ML pipeline:
  1. XGBoost classifier for individual flight risk prediction
  2. Isolation Forest anomaly detector for operational burnout
Includes SHAP explainability for HR-interpretable outputs.
"""
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Attrition Classifier (XGBoost)
# ---------------------------------------------------------------------------

def train_attrition_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    n_folds: int = 5,
    model_save_path: Optional[str] = None,
) -> tuple[xgb.XGBClassifier, shap.TreeExplainer, np.ndarray, dict]:
    """
    Trains an XGBoost classifier to predict employee attrition.

    Args:
        X:               Feature DataFrame (numeric only — encode categoricals first)
        y:               Binary target series (1=left, 0=retained)
        test_size:       Holdout fraction for final evaluation
        n_folds:         Stratified K-Fold cross-validation folds
        model_save_path: If set, saves the trained model as JSON

    Returns:
        model:       Trained XGBClassifier
        explainer:   SHAP TreeExplainer
        shap_values: SHAP values on the holdout test set
        metrics:     Dict with ROC-AUC, AP score, and classification report
    """
    logger.info(f"Training attrition model: {len(X)} samples, class balance: {y.mean():.2%} positive")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Handle class imbalance via scale_pos_weight
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_weight = neg / pos if pos > 0 else 1.0
    logger.info(f"  scale_pos_weight = {scale_weight:.2f} (neg={neg}, pos={pos})")

    # Cross-validation to tune early stopping
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_aucs = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        fold_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=scale_weight,
            random_state=42,
            eval_metric="auc",
            early_stopping_rounds=20,
            verbosity=0,
        )
        fold_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        val_probs = fold_model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_probs)
        cv_aucs.append(auc)
        logger.info(f"  Fold {fold}/{n_folds} — AUC: {auc:.4f}")

    logger.info(f"  CV AUC: {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")

    # Final model on full training set
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_weight,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    # Evaluation on holdout
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    roc_auc  = roc_auc_score(y_test, probs)
    avg_prec = average_precision_score(y_test, probs)
    cm       = confusion_matrix(y_test, preds).tolist()
    clf_report = classification_report(y_test, preds, output_dict=True)

    metrics = {
        "roc_auc": round(roc_auc, 4),
        "average_precision": round(avg_prec, 4),
        "cv_auc_mean": round(float(np.mean(cv_aucs)), 4),
        "cv_auc_std": round(float(np.std(cv_aucs)), 4),
        "confusion_matrix": cm,
        "classification_report": clf_report,
    }

    logger.info(f"  Holdout ROC-AUC:  {roc_auc:.4f}")
    logger.info(f"  Avg Precision:    {avg_prec:.4f}")
    print(classification_report(y_test, preds, target_names=["Retained", "Attrited"]))

    # SHAP explainability
    logger.info("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Save model
    if model_save_path:
        out = Path(model_save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(out))
        metrics_path = out.with_suffix(".metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info(f"  Model saved: {out}")
        logger.info(f"  Metrics saved: {metrics_path}")

    return model, explainer, shap_values, metrics


def predict_risk_scores(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    threshold: float = 0.75,
) -> pd.DataFrame:
    """
    Generate flight risk probability scores and risk tier for each employee.

    Risk Tiers:
        HIGH   — probability >= threshold  (immediate intervention)
        MEDIUM — 0.40 <= probability < threshold
        LOW    — probability < 0.40
    """
    probs = model.predict_proba(X)[:, 1]
    risk_tier = pd.cut(
        probs,
        bins=[-np.inf, 0.40, threshold, np.inf],
        labels=["LOW", "MEDIUM", "HIGH"],
    )
    return pd.DataFrame({
        "flight_risk_probability": probs.round(4),
        "risk_tier": risk_tier,
        "high_risk_flag": (probs >= threshold).astype(int),
    }, index=X.index)


# ---------------------------------------------------------------------------
# 2. Burnout Anomaly Detector (Isolation Forest)
# ---------------------------------------------------------------------------

def train_burnout_detector(
    df_telemetry_agg: pd.DataFrame,
    contamination: float = 0.08,
    model_save_path: Optional[str] = None,
) -> tuple[IsolationForest, StandardScaler]:
    """
    Trains an Isolation Forest to detect anomalous workload patterns
    indicative of operational burnout.

    Args:
        df_telemetry_agg: Per-employee aggregated telemetry features
        contamination:    Expected fraction of anomalies (default 8%)
        model_save_path:  If set, saves scaler params

    Returns:
        detector: Fitted IsolationForest
        scaler:   Fitted StandardScaler
    """
    BURNOUT_FEATURES = [
        "avg_overtime_90d",
        "absence_trend_90d",
        "avg_hours_worked",
        "late_login_pct",
        "burnout_risk_score",
    ]
    available = [f for f in BURNOUT_FEATURES if f in df_telemetry_agg.columns]
    X_burnout = df_telemetry_agg[available].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_burnout)

    detector = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    detector.fit(X_scaled)

    # Annotate predictions: -1 = anomaly, 1 = normal → remap to 1/0
    scores = detector.decision_function(X_scaled)
    preds  = detector.predict(X_scaled)  # -1 or 1

    burnout_flags = (preds == -1).astype(int)
    burnout_pct   = burnout_flags.mean() * 100
    logger.info(f"Burnout detector trained — {burnout_pct:.1f}% flagged as anomalies")

    return detector, scaler


def detect_burnout(
    detector: IsolationForest,
    scaler: StandardScaler,
    df_telemetry_agg: pd.DataFrame,
) -> pd.DataFrame:
    """Apply trained burnout detector and return anomaly scores per employee."""
    BURNOUT_FEATURES = [
        "avg_overtime_90d",
        "absence_trend_90d",
        "avg_hours_worked",
        "late_login_pct",
        "burnout_risk_score",
    ]
    available = [f for f in BURNOUT_FEATURES if f in df_telemetry_agg.columns]
    X = df_telemetry_agg[available].fillna(0)
    X_scaled = scaler.transform(X)

    anomaly_score = detector.decision_function(X_scaled)
    is_burnout    = (detector.predict(X_scaled) == -1).astype(int)

    return pd.DataFrame({
        "burnout_anomaly_score": anomaly_score.round(4),
        "burnout_flag": is_burnout,
    }, index=df_telemetry_agg.index)


# ---------------------------------------------------------------------------
# 3. SHAP Explanation Helpers
# ---------------------------------------------------------------------------

def get_top_risk_drivers(
    explainer: shap.TreeExplainer,
    X_row: pd.DataFrame,
    top_n: int = 5,
) -> list[dict]:
    """
    Returns the top N SHAP-based risk drivers for a single employee,
    formatted for HR dashboard display.

    Args:
        explainer: Fitted SHAP TreeExplainer
        X_row:     Single-row DataFrame for the employee
        top_n:     Number of top drivers to return

    Returns:
        List of dicts: [{"feature": ..., "value": ..., "impact": ...}, ...]
    """
    shap_vals = explainer.shap_values(X_row)[0]
    feature_names = X_row.columns.tolist()

    drivers = sorted(
        zip(feature_names, X_row.values[0], shap_vals),
        key=lambda x: abs(x[2]),
        reverse=True,
    )[:top_n]

    return [
        {
            "feature": feat,
            "value": round(float(val), 3),
            "shap_impact": round(float(shap_v), 4),
            "direction": "increases_risk" if shap_v > 0 else "reduces_risk",
        }
        for feat, val, shap_v in drivers
    ]


# ---------------------------------------------------------------------------
# 4. HR Intervention Recommender
# ---------------------------------------------------------------------------

INTERVENTION_RULES = [
    {
        "condition": lambda row: row.get("avg_overtime_90d", 0) > 3,
        "intervention": "Schedule workload review — sustained overtime detected (>3h/day avg over 90 days).",
        "priority": "HIGH",
    },
    {
        "condition": lambda row: row.get("absence_trend_90d", 0) > 5,
        "intervention": "Arrange wellness check-in — elevated unplanned absences in past 90 days.",
        "priority": "HIGH",
    },
    {
        "condition": lambda row: row.get("satisfaction_score", 5) < 2.5,
        "intervention": "Conduct 1-on-1 satisfaction review with HRBP — low engagement survey score.",
        "priority": "HIGH",
    },
    {
        "condition": lambda row: row.get("compensation_fairness_score", 5) < 2.5,
        "intervention": "Trigger compensation benchmark review — employee perceives pay as unfair.",
        "priority": "MEDIUM",
    },
    {
        "condition": lambda row: row.get("promotions_count", 1) == 0 and row.get("tenure_years", 0) > 3,
        "intervention": "Review career progression path — no promotion in 3+ years of tenure.",
        "priority": "MEDIUM",
    },
    {
        "condition": lambda row: row.get("career_growth_score", 5) < 2.5,
        "intervention": "Discuss growth opportunities — low career growth satisfaction.",
        "priority": "MEDIUM",
    },
    {
        "condition": lambda row: row.get("manager_relationship_score", 5) < 2,
        "intervention": "Escalate to HR Director — critical manager relationship breakdown detected.",
        "priority": "HIGH",
    },
]


def generate_interventions(employee_row: dict) -> list[dict]:
    """
    Generate actionable HR interventions for a flagged employee
    based on their feature profile.

    Args:
        employee_row: Dict of feature values for one employee

    Returns:
        List of triggered intervention recommendations
    """
    triggered = []
    for rule in INTERVENTION_RULES:
        try:
            if rule["condition"](employee_row):
                triggered.append({
                    "intervention": rule["intervention"],
                    "priority": rule["priority"],
                })
        except Exception:
            pass
    return triggered
