# Predictive Workforce Attrition & Productivity Analytics

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?logo=streamlit)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?logo=docker)](https://www.docker.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-336791?logo=postgresql)](https://postgresql.org)
[![Apache Superset](https://img.shields.io/badge/Apache_Superset-4.0+-E63946)](https://superset.apache.org)
[![Kilo Code](https://img.shields.io/badge/Kilo_Code-AI_Assisted-6C63FF)](https://kilocode.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**An end-to-end ML + BI pipeline to predict employee attrition, detect burnout anomalies, and recommend retention interventions — before it's too late.**

</div>

---

## 🎯 Overview

This system bridges **human resources** and **data science** to catch early behavioral cues of flight risk and workload burnout. Instead of reacting to resignation letters or discovering fatigue exit interviews, it proactively surfaces intervention opportunities using a multi-source data pipeline and dual machine learning models.

```
[ HRIS / Payroll ] ──┐
                     │
[ Jira / GitHub ] ───┼──> [ ETL Pipeline ] ──> [ PostgreSQL ] ──> [ XGBoost + IsoForest ]
                     │                                                       │
[ LMS / Training ] ──┤                                                       ▼
                     │                                            [ Interactive UI Layer ]
[ Badge / Access ] ──┘                                        (Streamlit / Superset / Swagger)
```

---

## ✨ Key Features

| Layer | Technology | Capability |
|-------|-----------|------------|
| **Ingestion** | Python / Pandas | Multi-source ETL: HRIS, Jira tickets, GitHub commits, LMS, and badge swipe data |
| **Storage** | PostgreSQL 15+ | Normalized 3NF schema + optimized analytical views (`vw_ml_feature_set`, `vw_department_health`) |
| **Feature Eng.** | Pandas / NumPy | 90-day rolling telemetry, burnout indicators, salary bands, and tenure calculations |
| **Dual ML Engine** | XGBoost + IsoForest | Supervised attrition flight risk classifier + unsupervised operational burnout detector |
| **Explainability** | SHAP TreeExplainer | Plain-language directional drivers explaining *why* an employee is flagged |
| **Interventions** | Rule Engine | Priority-tiered automated HR retention recommendations (`[HIGH]`, `[MEDIUM]`) |
| **BI Dashboards** | Streamlit + Superset | 3 role-based portals: **Executive Overview**, **Manager Portal**, and **HR Intervention Log** |
| **API Layer** | FastAPI + Uvicorn | Real-time `/predict/attrition`, batch prediction, and `/interventions` REST endpoints |
| **AI Assistant** | Kilo Code | Configured `.kilocode/` with 3 specialized modes (`attrition-analyst`, `data-pipeline`, `ml-engineer`) |

---

## 🏗️ Architecture & Data Flow

### Relational Schema

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

---

## 🖥️ Running & Testing the UIs

This project provides **4 distinct ways** to interact with and visualize the analytics:

### 1. 📊 Interactive BI Web Dashboard (Streamlit) — *Recommended*

The primary presentation layer featuring the **Executive Overview**, **Manager Team Portal**, **HR Intervention Log**, and a live **"What-If" Flight Risk Simulator**.

```powershell
python -m streamlit run dashboard/app.py
```
👉 Open your browser at: **`http://localhost:8501`**

- **🏢 Executive Overview:** Company-Wide Risk Gauge, KPI cards, and **Burnout Hotspot Heatmap** (Department vs. Month Overtime).
- **👥 Manager Portal:** Team retention risk donut chart and **Workload vs. Output Fatigue Scatter** (detecting falling task closures despite rising overtime).
- **📋 HR Intervention Log:** Filterable roster of flagged employees, primary SHAP risk drivers, and automated retention action plans.
- **🧪 Flight Risk Simulator:** Drag sliders (overtime, absences, satisfaction, tenure) to calculate flight risk in real time.

---

### 2. 🌐 FastAPI Interactive REST Console (Swagger UI)

Test real-time model inference, batch scoring, and automated intervention generation:

```powershell
python -m uvicorn api.main:app --reload --port 8080 --host 127.0.0.1
```
👉 Open your browser at: **`http://127.0.0.1:8080/docs`**

- `GET /health` — Verify model load status and runtime health.
- `POST /predict/attrition` — Submit single-employee telemetry and receive probability, risk tier (`HIGH`/`MEDIUM`/`LOW`), and top SHAP drivers.
- `POST /predict/batch` — Bulk predict up to 500 employees per call.
- `POST /interventions` — Retrieve recommended HR actions for an employee profile.

---

### 3. 📓 Jupyter Exploratory Analysis Notebook

Explore raw distributions, burnout correlation heatmaps, and statistical charts:

- **Inside VS Code:** Open [`notebooks/01_exploratory_analysis.ipynb`](notebooks/01_exploratory_analysis.ipynb), select your Python kernel, and click **Run All**.
- **In Browser:**
  ```powershell
  python -m jupyter notebook notebooks/01_exploratory_analysis.ipynb
  ```

---

### 4. 🐳 Docker & Docker Compose Setup

Run the entire ecosystem (Streamlit Dashboard + FastAPI REST API + PostgreSQL + Apache Superset) in isolated containers with zero manual environment configuration:

#### Start Everything:
```bash
docker compose up --build
```

#### Access Points:
| Service | URL | Default Credentials | Description |
|---|---|---|---|
| **Streamlit Dashboard** | `http://localhost:8501` | None | Interactive BI Analytics Portal |
| **FastAPI Swagger UI** | `http://localhost:8080/docs` | None | REST Prediction Engine & API Docs |
| **Apache Superset** | `http://localhost:8088` | `admin` / `admin` | Enterprise BI Dashboards |
| **PostgreSQL Database** | `localhost:5432` | `postgres` / `postgres` | Data Warehouse (`workforce_analytics`) |

#### Standalone Apache Superset (Alternative):
If you want to run Apache Superset on its own:
```bash
docker run -d --name superset -p 8088:8088 apache/superset
docker exec -it superset superset fab create-admin --username admin --firstname Admin --lastname User --email admin@fab.org --password admin
docker exec -it superset superset db upgrade
docker exec -it superset superset init
```
Connect Superset to PostgreSQL using URI:
`postgresql://postgres:password@host.docker.internal:5432/workforce_analytics`
Load chart queries directly from [`dashboard/superset_config.py`](dashboard/superset_config.py).

---

## 🚀 Local Quick Start (Without Docker)

### 1. Clone & Setup Environment
```bash
git clone https://github.com/andyouranurag/workforce-attrition-analytics.git
cd workforce-attrition-analytics

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # Linux / macOS

# Install all dependencies
pip install -r requirements.txt
```

### 2. Generate Synthetic HR Data
Produces a 500-employee realistic dataset across 8 departments:
```bash
python database/seed_data.py
```

### 3. Run Multi-Source Ingestion
Parses seed CSVs and populates parquet files in `data/raw/`:
```bash
python -m etl.ingest
```

### 4. Train Dual ML Pipeline
Executes feature engineering, 5-fold cross-validation, holdout evaluation, SHAP explainability, and saves model artifacts:
```bash
python models/train_pipeline.py
```
*Outputs:*
- `models/artifacts/attrition_model.json`
- `data/processed/risk_scores.csv`
- `reports/roc_curve.png`, `reports/pr_curve.png`, `reports/shap_summary.png`

### 5. Run the Test Suite
```bash
python -m pytest tests/ -v
```
*(Runs 32 unit and integration tests covering features, model predictions, interventions, and API controllers)*

---

## 📁 Project Structure

```
workforce-attrition-analytics/
│
├── .kilocode/                  # Kilo Code AI configuration
│   ├── settings.json           # Agent modes: Attrition Analyst, Pipeline, ML Engineer
│   └── rules.md                # Project guidelines, ethics & privacy rules
│
├── api/
│   ├── main.py                 # FastAPI REST application (optimized threadpool execution)
│   ├── schemas.py              # Pydantic V2 request & response models
│   └── __init__.py
│
├── dashboard/
│   ├── app.py                  # Streamlit BI Web Dashboard (Executive, Manager, HR, Simulator)
│   └── superset_config.py      # Apache Superset role-based dashboard definitions & SQL
│
├── database/
│   ├── schema.sql              # PostgreSQL DDL with indexes & analytical views
│   ├── seed_data.py            # Synthetic 500-employee realistic data generator
│   └── migrations/             # Schema migration scripts
│
├── etl/
│   ├── config.py               # Database and data source connection settings
│   ├── ingest.py               # Modular ingesters (HRIS, Jira, GitHub, LMS, Badge)
│   ├── features.py             # Feature store generator with 90-day rolling behavioral windows
│   └── transform.py            # Cleaning, clipping, and categorical one-hot encoding
│
├── models/
│   ├── model.py                # XGBoost classifier, Isolation Forest, and SHAP drivers
│   ├── evaluate.py             # Evaluation curve plots and JSON metrics logging
│   ├── train_pipeline.py       # End-to-end training orchestration script
│   └── artifacts/              # Serialized model binaries (gitignored)
│
├── notebooks/
│   └── 01_exploratory_analysis.ipynb # Interactive visual analysis notebook
│
├── reports/                    # Generated ROC, PR, and SHAP summary plots
├── tests/
│   ├── conftest.py             # Shared pytest fixtures
│   ├── test_api.py             # FastAPI endpoint integration tests
│   ├── test_features.py        # Feature engineering unit tests
│   └── test_model.py           # ML training, risk tiering, and intervention tests
│
├── Dockerfile                  # Container definition for API & Streamlit
├── docker-compose.yml          # Multi-container orchestration (App, DB, Superset)
├── requirements.txt            # Complete Python dependency specifications
├── .env.example                # Template environment variables
├── .gitignore
└── README.md
```

---

## 📊 Model Performance (Holdout Evaluation)

| Metric | Holdout Score | Notes |
|---|---|---|
| **ROC-AUC** | **0.9241** | Evaluated on stratified 20% holdout test set |
| **Average Precision** | **0.8730** | High precision on minority turnover class |
| **5-Fold CV AUC** | **0.9567 ± 0.0327** | Stable generalization across all validation folds |
| **F1-Score (Attrited)**| **0.86** | Recall: 0.89, Precision: 0.84 |
| **High Risk Cutoff** | **≥ 75% probability** | Triggers automated immediate HR intervention workflows |

---

## 🔒 Privacy & Compliance

- **Micro-Surveillance Prevention:** Manager portal shows team-level aggregations and risk tiers; raw probability decimals and individual surveillance flags are restricted.
- **Protected Attribute Safeguards:** Sensitive demographic attributes (e.g. gender, marital status) are excluded from the model feature store to prevent algorithmic bias.
- **Role-Based Access Control (RBAC):** Superset and dashboard configurations restrict individual intervention logs strictly to certified HRBPs and HR Directors.

---

## 🤖 Kilo Code Integration

This project is configured for [Kilo Code](https://kilocode.ai) with 3 specialized AI agent modes:

| Mode | ID | Primary Focus |
|---|---|---|
| **Attrition Analyst** | `attrition-analyst` | End-to-end HR analytics, feature engineering, and KPI layout design |
| **Pipeline Engineer** | `data-pipeline` | ETL data quality, schema drift checks, and PostgreSQL optimization |
| **ML Engineer** | `ml-engineer` | Model hyperparameter tuning, class balancing, and SHAP explainability |

Open this project in VS Code with the Kilo Code extension installed to switch between modes.

---

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
