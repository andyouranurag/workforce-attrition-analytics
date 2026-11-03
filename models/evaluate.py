"""
Model Evaluation & Monitoring
Tracks model performance over time and generates evaluation reports.
"""
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
)

logger = logging.getLogger(__name__)


def plot_roc_curve(y_true, y_probs, save_path: str = "./reports/roc_curve.png"):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Attrition Model — ROC Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"ROC curve saved: {save_path}")


def plot_precision_recall(y_true, y_probs, save_path: str = "./reports/pr_curve.png"):
    """Plot and save Precision-Recall curve."""
    prec, rec, _ = precision_recall_curve(y_true, y_probs)
    ap = average_precision_score(y_true, y_probs)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec, prec, color="#ff7f0e", lw=2, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Attrition Model — Precision-Recall Curve")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"PR curve saved: {save_path}")


def plot_shap_summary(shap_values, X_test, save_path: str = "./reports/shap_summary.png"):
    """Generate SHAP summary bar plot."""
    import shap
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info(f"SHAP summary saved: {save_path}")


def generate_evaluation_report(metrics: dict, save_path: str = "./reports/evaluation_report.json"):
    """Append model evaluation metrics with timestamp to a JSON log."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    record = {
        "evaluated_at": datetime.utcnow().isoformat() + "Z",
        **metrics,
    }

    history = []
    if Path(save_path).exists():
        with open(save_path) as f:
            history = json.load(f)

    history.append(record)
    with open(save_path, "w") as f:
        json.dump(history, f, indent=2, default=str)

    logger.info(f"Evaluation report updated: {save_path}")
    return record
