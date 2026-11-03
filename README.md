# Predictive Workforce Attrition & Productivity Analytics

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange)](https://xgboost.readthedocs.io)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-336791?logo=postgresql)](https://postgresql.org)
[![Apache Superset](https://img.shields.io/badge/Apache_Superset-4.0+-E63946)](https://superset.apache.org)
[![Kilo Code](https://img.shields.io/badge/Kilo_Code-AI_Assisted-6C63FF)](https://kilocode.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**An end-to-end ML + BI pipeline to predict employee attrition and detect burnout — before it's too late.**

</div>

---

## 🎯 Overview

This system bridges **human resources** and **data science** to catch early behavioral cues of flight risk and workload burnout. Instead of reacting to resignation letters, it proactively surfaces intervention opportunities using a multi-source data pipeline.

```
[ HRIS / Payroll ] ──┐
                     │
[ Jira / GitHub ] ───┼──> [ ETL Pipeline ] ──> [ PostgreSQL ] ──> [ XGBoost + IsoForest ]
                     │                                                       │
[ LMS / Training ] ──┤                                                       ▼
                     │                                            [ Apache Superset BI ]
[ Badge / Access ] ──┘                                         (Executive / Manager / HR)
```

---

## ✨ Key Features

| Layer | Technology | Capability |
|-------|-----------|------------|
| **Ingestion** | Python / Pandas | Multi-source ETL: HRIS, Jira, GitHub, LMS, Badge |
| **Storage** | PostgreSQL 15+ | Normalized schema + analytical views |
| **Feature Eng.** | Pandas / NumPy | 90-day rolling windows, burnout scores, tenure bands |
| **ML — Attrition** | XGBoost + SHAP | Individual flight risk with explainable drivers |
| **ML — Burnout** | Isolation Forest | Anomaly detection on workload telemetry |
| **BI Dashboard** | Apache Superset | 3-tier role-based views (Executive / Manager / HR) |
| **AI Dev** | Kilo Code | Custom agent modes for analyst, pipeline, and ML roles |

---

## 🏗️ Architecture

### Data Schema

```
dim_employee          fact_performance        fact_workload_telemetry
──────────────        ────────────────        ───────────────────────
emp_id (PK)     ◄──  emp_id (FK)        ◄──  emp_id (FK)
hire_date             review_date              log_date
department            performance_rating        hours_worked
job_role              projects_completed        overtime_hours
base_salary           promotions_count          tasks_closed
manager_id            training_hours            unplanned_absence_flag

fact_engagement_survey     fact_attrition_labels
──────────────────────     ─────────────────────
emp_id (FK)           ◄──  emp_id (FK / PK)
survey_date                left_company_flag
satisfaction_score          exit_date
work_life_balance_score     voluntary_flag
manager_relationship_score
```

### ML Pipeline

```
Raw Data ──> Feature Engineering ──> XGBoost Classifier ──> Risk Scores
                                                         └──> SHAP Drivers
             Telemetry Aggregation ──> Isolation Forest  ──> Burnout Flags
                                                         └──> Anomaly Scores
```

### Dashboard Views

| View | Audience | Data Granularity |
|------|----------|-----------------|
| **Executive Overview** | C-Suite / VP HR | Company-wide, department aggregates |
| **Manager Portal** | Department Heads | Team risk tiers (no individual raw scores) |
| **HR Intervention Log** | HRBPs | Individual detail + intervention recommendations |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 15+
- Apache Superset (optional, for dashboard)

### Installation

```bash
# Clone the repo
git clone https://github.com/andyouranurag/workforce-attrition-analytics.git
cd workforce-attrition-analytics

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Database Setup

```bash
# Create PostgreSQL database
psql -U postgres -c "CREATE DATABASE workforce_analytics;"

# Run schema
psql -U postgres -d workforce_analytics -f database/schema.sql

# Generate and load seed data
python database/seed_data.py
```

### Generate Synthetic Data

```bash
python database/seed_data.py
# Outputs 500-employee synthetic dataset to database/seed/
```

### Run Ingestion

```bash
python -m etl.ingest
# Reads seed CSVs → saves parquet to data/raw/
```

### Train Models

```bash
python models/train_pipeline.py
# Full pipeline: features → XGBoost → IsoForest → eval reports → risk scores
# Outputs: models/artifacts/attrition_model.json
#          reports/roc_curve.png
#          reports/shap_summary.png
#          data/processed/risk_scores.csv
```

### Run Tests

```bash
pytest tests/ -v --cov=etl --cov=models
```

---

## 📁 Project Structure

```
workforce-attrition-analytics/
│
├── .kilocode/                  # Kilo Code AI configuration
│   ├── settings.json           # Agent modes: Attrition Analyst, Pipeline, ML Engineer
│   └── rules.md                # AI coding rules & project conventions
│
├── database/
│   ├── schema.sql              # Full PostgreSQL schema + analytical views
│   ├── seed_data.py            # Synthetic 500-employee data generator
│   ├── migrations/             # Versioned schema migrations
│   └── seed/                   # Generated CSV seed files (gitignored)
│
├── etl/
│   ├── config.py               # DB + source API configuration
│   ├── ingest.py               # Multi-source ingesters (HRIS, Jira, GitHub, Badge)
│   ├── features.py             # Feature engineering & rolling aggregations
│   └── transform.py            # Data cleaning & categorical encoding
│
├── models/
│   ├── model.py                # XGBoost attrition + Isolation Forest burnout
│   ├── evaluate.py             # ROC/PR curves, SHAP plots, eval reports
│   ├── train_pipeline.py       # End-to-end training orchestration
│   └── artifacts/              # Saved model binaries (gitignored)
│
├── dashboard/
│   ├── superset_config.py      # Apache Superset configuration
│   ├── charts/                 # Chart definitions (JSON exports)
│   └── dashboards/             # Dashboard layout exports
│
├── api/
│   ├── main.py                 # FastAPI REST endpoints
│   ├── schemas.py              # Pydantic request/response models
│   └── routers/                # Endpoint routers
│
├── tests/
│   ├── test_features.py        # Feature engineering unit tests
│   ├── test_model.py           # Model training + prediction tests
│   ├── test_ingest.py          # Ingestion pipeline tests
│   └── conftest.py             # Shared pytest fixtures
│
├── notebooks/
│   └── 01_exploratory_analysis.ipynb
│
├── reports/                    # Generated evaluation artifacts
├── data/
│   ├── raw/                    # Parquet files (gitignored)
│   └── processed/              # Feature store + risk scores (gitignored)
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 🔒 Privacy & Compliance

This system processes sensitive HR data. Built-in safeguards:

- **Anonymization**: emp_id only in ML outputs — names/emails never logged
- **Role-Based Access**: Dashboard tiers enforce minimum aggregation (team ≥ 5 employees)
- **Data Retention**: Telemetry 2yr, Survey 5yr, Attrition labels permanent
- **GDPR/CCPA**: Exit interview data restricted to HR Director role
- **Ethical AI**: Protected attributes (gender, marital status) excluded from final model features; SHAP fairness checks included

---

## 📊 Model Performance (Synthetic Data)

| Metric | Value |
|--------|-------|
| ROC-AUC | ~0.87 |
| Average Precision | ~0.72 |
| CV AUC (5-fold) | 0.85 ± 0.03 |
| High Risk Threshold | ≥ 75% probability |

---

## 🤖 Kilo Code Integration

This project is configured for [Kilo Code](https://kilocode.ai) with three specialized AI agent modes:

| Mode | ID | Focus |
|------|----|-------|
| Attrition Analyst | `attrition-analyst` | Full-stack HR data science |
| Pipeline Engineer | `data-pipeline` | ETL debugging + optimization |
| ML Engineer | `ml-engineer` | Model tuning + explainability |

Open in VS Code with Kilo Code extension → select mode from the mode switcher.

---

## 📄 License

MIT — see [LICENSE](LICENSE)

---

<div align="center">
Built with ❤️ using Python, XGBoost, PostgreSQL, Apache Superset & Kilo Code
</div>
