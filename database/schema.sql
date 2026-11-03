-- ============================================================
-- Predictive Workforce Attrition Analytics
-- PostgreSQL Schema Definition
-- ============================================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ------------------------------------------------------------
-- DIMENSION: Employee Master
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_employee (
    emp_id              VARCHAR(20) PRIMARY KEY,
    first_name          VARCHAR(100) NOT NULL,
    last_name           VARCHAR(100) NOT NULL,
    hire_date           DATE NOT NULL,
    department          VARCHAR(100) NOT NULL,
    job_role            VARCHAR(100) NOT NULL,
    job_level           VARCHAR(20) CHECK (job_level IN ('Junior','Mid','Senior','Lead','Manager','Director')),
    base_salary         NUMERIC(12,2) NOT NULL,
    manager_id          VARCHAR(20) REFERENCES dim_employee(emp_id),
    distance_from_home  NUMERIC(5,2),  -- km
    education_level     VARCHAR(50),
    employment_type     VARCHAR(30) CHECK (employment_type IN ('Full-Time','Part-Time','Contract')),
    gender              VARCHAR(20),
    marital_status      VARCHAR(20),
    location            VARCHAR(100),
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE dim_employee IS 'Slow-changing demographic and contractual attributes per employee';

-- ------------------------------------------------------------
-- FACT: Performance Records
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fact_performance (
    record_id           UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    emp_id              VARCHAR(20) NOT NULL REFERENCES dim_employee(emp_id),
    review_date         DATE NOT NULL,
    performance_rating  NUMERIC(3,1) CHECK (performance_rating BETWEEN 1 AND 5),
    projects_completed  INTEGER DEFAULT 0,
    promotions_count    INTEGER DEFAULT 0,
    training_hours      NUMERIC(6,2) DEFAULT 0,
    peer_review_score   NUMERIC(3,1) CHECK (peer_review_score BETWEEN 1 AND 5),
    manager_rating      NUMERIC(3,1) CHECK (manager_rating BETWEEN 1 AND 5),
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_fact_performance_emp_id ON fact_performance(emp_id);
CREATE INDEX idx_fact_performance_review_date ON fact_performance(review_date);

COMMENT ON TABLE fact_performance IS 'Historical performance scores and milestone markers per review cycle';

-- ------------------------------------------------------------
-- FACT: Workload Telemetry (Daily)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fact_workload_telemetry (
    log_id                  UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    emp_id                  VARCHAR(20) NOT NULL REFERENCES dim_employee(emp_id),
    log_date                DATE NOT NULL,
    hours_worked            NUMERIC(5,2) DEFAULT 0,
    overtime_hours          NUMERIC(5,2) DEFAULT 0,
    tasks_closed            INTEGER DEFAULT 0,
    tasks_opened            INTEGER DEFAULT 0,
    commits_count           INTEGER DEFAULT 0,  -- GitHub/GitLab activity
    tickets_resolved        INTEGER DEFAULT 0,  -- Jira
    unplanned_absence_flag  BOOLEAN DEFAULT FALSE,
    late_login_flag         BOOLEAN DEFAULT FALSE,
    badge_swipe_count       INTEGER DEFAULT 0,  -- Physical access
    created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_workload_emp_id ON fact_workload_telemetry(emp_id);
CREATE INDEX idx_workload_date ON fact_workload_telemetry(log_date);
CREATE INDEX idx_workload_emp_date ON fact_workload_telemetry(emp_id, log_date);

COMMENT ON TABLE fact_workload_telemetry IS 'Daily operational activity tracking from Jira, timesheets, access logs';

-- ------------------------------------------------------------
-- FACT: Engagement Surveys
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fact_engagement_survey (
    survey_id                   UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    emp_id                      VARCHAR(20) NOT NULL REFERENCES dim_employee(emp_id),
    survey_date                 DATE NOT NULL,
    satisfaction_score          NUMERIC(3,1) CHECK (satisfaction_score BETWEEN 1 AND 5),
    work_life_balance_score     NUMERIC(3,1) CHECK (work_life_balance_score BETWEEN 1 AND 5),
    manager_relationship_score  NUMERIC(3,1) CHECK (manager_relationship_score BETWEEN 1 AND 5),
    career_growth_score         NUMERIC(3,1) CHECK (career_growth_score BETWEEN 1 AND 5),
    compensation_fairness_score NUMERIC(3,1) CHECK (compensation_fairness_score BETWEEN 1 AND 5),
    team_cohesion_score         NUMERIC(3,1) CHECK (team_cohesion_score BETWEEN 1 AND 5),
    survey_type                 VARCHAR(30) DEFAULT 'pulse',  -- 'pulse' or 'annual'
    created_at                  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_survey_emp_id ON fact_engagement_survey(emp_id);
CREATE INDEX idx_survey_date ON fact_engagement_survey(survey_date);

COMMENT ON TABLE fact_engagement_survey IS 'Pulse and annual survey metrics on Likert scale (1-5)';

-- ------------------------------------------------------------
-- FACT: Attrition Labels (ML Target Variable)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fact_attrition_labels (
    emp_id          VARCHAR(20) PRIMARY KEY REFERENCES dim_employee(emp_id),
    left_company_flag   BOOLEAN NOT NULL DEFAULT FALSE,  -- 0=Active, 1=Left
    exit_date           DATE,
    voluntary_flag      BOOLEAN,  -- TRUE=Voluntary, FALSE=Involuntary
    exit_reason         VARCHAR(200),
    exit_interview_score NUMERIC(3,1),
    rehire_eligible     BOOLEAN DEFAULT TRUE,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE fact_attrition_labels IS 'Target variable records for ML model training — attrition outcomes';

-- ------------------------------------------------------------
-- ANALYTICAL VIEW: ML Feature Set
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW vw_ml_feature_set AS
SELECT
    e.emp_id,
    e.department,
    e.job_role,
    e.job_level,
    e.base_salary,
    e.distance_from_home,
    e.education_level,
    e.employment_type,
    e.gender,
    e.marital_status,
    -- Tenure in years
    ROUND(EXTRACT(EPOCH FROM (CURRENT_DATE - e.hire_date)) / (365.25 * 86400), 2) AS tenure_years,
    -- Latest performance metrics
    p.performance_rating,
    p.projects_completed,
    p.promotions_count,
    p.training_hours,
    p.peer_review_score,
    p.manager_rating,
    -- 90-day workload aggregates
    w.avg_overtime_90d,
    w.absence_trend_90d,
    w.avg_hours_worked_90d,
    w.avg_tasks_closed_90d,
    -- Latest survey scores
    s.satisfaction_score,
    s.work_life_balance_score,
    s.manager_relationship_score,
    s.career_growth_score,
    s.compensation_fairness_score,
    -- Target variable
    COALESCE(a.left_company_flag::INT, 0) AS attrition_label
FROM dim_employee e
LEFT JOIN LATERAL (
    SELECT * FROM fact_performance
    WHERE emp_id = e.emp_id
    ORDER BY review_date DESC
    LIMIT 1
) p ON TRUE
LEFT JOIN LATERAL (
    SELECT
        ROUND(AVG(overtime_hours)::NUMERIC, 2)          AS avg_overtime_90d,
        SUM(unplanned_absence_flag::INT)                AS absence_trend_90d,
        ROUND(AVG(hours_worked)::NUMERIC, 2)            AS avg_hours_worked_90d,
        ROUND(AVG(tasks_closed)::NUMERIC, 2)            AS avg_tasks_closed_90d
    FROM fact_workload_telemetry
    WHERE emp_id = e.emp_id
      AND log_date >= CURRENT_DATE - INTERVAL '90 days'
) w ON TRUE
LEFT JOIN LATERAL (
    SELECT * FROM fact_engagement_survey
    WHERE emp_id = e.emp_id
    ORDER BY survey_date DESC
    LIMIT 1
) s ON TRUE
LEFT JOIN fact_attrition_labels a ON a.emp_id = e.emp_id;

COMMENT ON VIEW vw_ml_feature_set IS 'Pre-joined ML feature set — join dim + facts for model training pipeline';

-- ------------------------------------------------------------
-- ANALYTICAL VIEW: Department Health Dashboard
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW vw_department_health AS
SELECT
    e.department,
    COUNT(DISTINCT e.emp_id)                            AS headcount,
    ROUND(AVG(w.avg_overtime_90d), 2)                  AS avg_overtime_dept,
    ROUND(AVG(s.satisfaction_score), 2)                AS avg_satisfaction,
    ROUND(AVG(s.work_life_balance_score), 2)           AS avg_wlb_score,
    SUM(CASE WHEN a.left_company_flag THEN 1 ELSE 0 END) AS total_attrition,
    ROUND(
        SUM(CASE WHEN a.left_company_flag THEN 1 ELSE 0 END)::NUMERIC /
        NULLIF(COUNT(DISTINCT e.emp_id), 0) * 100, 2
    )                                                   AS attrition_rate_pct
FROM dim_employee e
LEFT JOIN LATERAL (
    SELECT ROUND(AVG(overtime_hours)::NUMERIC, 2) AS avg_overtime_90d
    FROM fact_workload_telemetry
    WHERE emp_id = e.emp_id
      AND log_date >= CURRENT_DATE - INTERVAL '90 days'
) w ON TRUE
LEFT JOIN LATERAL (
    SELECT satisfaction_score, work_life_balance_score
    FROM fact_engagement_survey
    WHERE emp_id = e.emp_id
    ORDER BY survey_date DESC LIMIT 1
) s ON TRUE
LEFT JOIN fact_attrition_labels a ON a.emp_id = e.emp_id
GROUP BY e.department
ORDER BY attrition_rate_pct DESC NULLS LAST;

COMMENT ON VIEW vw_department_health IS 'Department-level health aggregation for BI dashboard Executive Overview';
