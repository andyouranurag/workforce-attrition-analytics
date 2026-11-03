# Workforce Attrition Analytics — Kilo Code AI Rules
# These rules guide the AI assistant when working in this project.

## Project Context
This is a production-grade HR analytics system that predicts employee attrition
and detects burnout using machine learning. The codebase bridges data engineering,
ML science, and BI visualization.

## Code Style Rules

### Python
- Use type hints on all function signatures
- Write Google-style docstrings explaining the HR business context, not just the code
- Follow PEP 8; max line length 100 chars
- Use pathlib.Path instead of os.path for file operations
- Prefer pandas vectorized operations over loops for DataFrame transformations

### SQL
- Use explicit column lists in SELECT (no SELECT *)
- Always add COMMENT ON TABLE/VIEW for documentation
- Use CTEs (WITH) for complex analytical queries
- Add covering indexes on (emp_id, date) for all fact tables

### Machine Learning
- Always stratify train/test splits (attrition dataset is imbalanced ~18%)
- Document scale_pos_weight calculations explicitly
- Generate SHAP explanations alongside every prediction batch
- Log ROC-AUC, Average Precision, and CV fold results to ./reports/

## Privacy & Compliance Rules
- NEVER log or print individual employee PII (names, emails) in model outputs
- Anonymize before displaying — use emp_id only in ML outputs
- Risk scores in BI dashboard must be aggregated at team level (min 5 employees)
- Exit interview data is CONFIDENTIAL — restrict to HR Director role only
- Document data retention: telemetry = 2 years, survey = 5 years, attrition labels = permanent

## Data Quality Rules
- Validate emp_id foreign keys before any fact table insert
- Flag telemetry records where hours_worked > 14 as suspect (manual entry error)
- Reject survey records where satisfaction_score is outside [1, 5]
- Cross-check attrition exit_date against telemetry last_active_date (max 30-day gap)

## Testing Rules
- Every ETL transform function must have a pytest unit test in ./tests/
- ML model tests should use synthetic data (use database/seed/) not production data
- Run tests with: pytest tests/ -v --cov=etl --cov=models

## Dashboard Rules
- Executive view: aggregated, no individual data
- Manager view: team-level risk tiers only (HIGH/MEDIUM/LOW), no raw probabilities
- HR Intervention view: individual detail, restricted to HRBP role
- All dashboards must have a "Data as of" timestamp

## File Organization
```
workforce-attrition-analytics/
├── .kilocode/          # Kilo Code AI configuration
├── database/           # SQL schema + seed data
├── etl/                # Ingestion + feature engineering
├── models/             # ML models + evaluation
├── dashboard/          # BI configs + Superset exports
├── api/                # FastAPI REST endpoints
├── tests/              # pytest test suite
├── notebooks/          # Jupyter exploration notebooks
├── reports/            # Generated evaluation plots + metrics
├── data/
│   ├── raw/            # Parquet files from ingestion
│   └── processed/      # Feature store + risk scores
└── models/artifacts/   # Saved model binaries
```
