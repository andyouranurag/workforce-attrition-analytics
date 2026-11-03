"""
Apache Superset configuration for the Workforce Attrition Analytics dashboard.
"""

# ── Database connection (replace with your PostgreSQL credentials) ──────────
SQLALCHEMY_DATABASE_URI = "postgresql://postgres:password@localhost:5432/workforce_analytics"
SECRET_KEY = "your-superset-secret-key"

# ── Feature flags ─────────────────────────────────────────────────────────────
FEATURE_FLAGS = {
    "ENABLE_TEMPLATE_PROCESSING": True,
    "DASHBOARD_RBAC": True,        # Role-based access control per dashboard
    "THUMBNAILS": True,
    "LISTVIEWS_DEFAULT_CARD_VIEW": True,
}

# ── Role-Based Access Control mapping ────────────────────────────────────────
# Maps Superset roles to which dashboard views they can access
DASHBOARD_ROLES = {
    "Executive": ["executive_overview"],
    "Manager": ["executive_overview", "manager_portal"],
    "HRBP": ["executive_overview", "manager_portal", "hr_intervention_log"],
    "HR_Director": ["executive_overview", "manager_portal", "hr_intervention_log"],
    "Admin": ["executive_overview", "manager_portal", "hr_intervention_log"],
}

# ── Dashboard Definitions ────────────────────────────────────────────────────
DASHBOARDS = {
    "executive_overview": {
        "title": "Executive Workforce Overview",
        "description": "Company-wide attrition risk meter, burnout hotspots, trend analysis",
        "charts": [
            {
                "id": "attrition_risk_gauge",
                "type": "gauge_chart",
                "title": "Company-Wide Attrition Risk",
                "metric": "high_risk_pct",
                "query": """
                    SELECT
                        ROUND(
                            COUNT(*) FILTER (WHERE high_risk_flag = 1)::NUMERIC /
                            NULLIF(COUNT(*), 0) * 100, 1
                        ) AS high_risk_pct
                    FROM risk_scores
                """,
                "thresholds": {"green": 10, "yellow": 20, "red": 30},
            },
            {
                "id": "burnout_heatmap",
                "type": "heatmap",
                "title": "Burnout Hotspot Matrix (Dept × Avg Overtime)",
                "query": """
                    SELECT
                        e.department,
                        EXTRACT(MONTH FROM w.log_date) AS month,
                        ROUND(AVG(w.overtime_hours), 2) AS avg_overtime
                    FROM fact_workload_telemetry w
                    JOIN dim_employee e ON e.emp_id = w.emp_id
                    WHERE w.log_date >= CURRENT_DATE - INTERVAL '6 months'
                    GROUP BY e.department, EXTRACT(MONTH FROM w.log_date)
                    ORDER BY e.department, month
                """,
                "x_axis": "month",
                "y_axis": "department",
                "metric": "avg_overtime",
            },
            {
                "id": "attrition_trend",
                "type": "line_chart",
                "title": "Monthly Attrition Rate Trend",
                "query": """
                    SELECT
                        DATE_TRUNC('month', exit_date) AS month,
                        COUNT(*) AS attritions,
                        ROUND(COUNT(*)::NUMERIC / 500 * 100, 2) AS attrition_rate_pct
                    FROM fact_attrition_labels
                    WHERE left_company_flag = TRUE
                      AND exit_date >= CURRENT_DATE - INTERVAL '24 months'
                    GROUP BY 1
                    ORDER BY 1
                """,
            },
            {
                "id": "dept_health_table",
                "type": "table",
                "title": "Department Health Scorecard",
                "query": "SELECT * FROM vw_department_health",
                "sort": "attrition_rate_pct DESC",
            },
        ],
    },

    "manager_portal": {
        "title": "Manager Team Portal",
        "description": "Team risk tiers, workload vs output ratio, retention breakdown",
        "charts": [
            {
                "id": "team_risk_distribution",
                "type": "pie_chart",
                "title": "Team Risk Distribution",
                "query": """
                    SELECT
                        risk_tier,
                        COUNT(*) AS employee_count
                    FROM risk_scores rs
                    JOIN dim_employee e ON e.emp_id = rs.emp_id
                    WHERE e.manager_id = '{{current_user_emp_id}}'
                    GROUP BY risk_tier
                """,
                "note": "Uses Jinja template — replace {{current_user_emp_id}} with your Superset filter",
            },
            {
                "id": "workload_vs_output",
                "type": "scatter_chart",
                "title": "Workload vs Output Ratio (Fatigue Indicator)",
                "query": """
                    SELECT
                        e.emp_id,
                        ROUND(AVG(w.hours_worked), 2) AS avg_hours,
                        ROUND(AVG(w.tasks_closed), 2) AS avg_output,
                        ROUND(AVG(w.overtime_hours), 2) AS avg_overtime
                    FROM fact_workload_telemetry w
                    JOIN dim_employee e ON e.emp_id = w.emp_id
                    WHERE w.log_date >= CURRENT_DATE - INTERVAL '90 days'
                    GROUP BY e.emp_id
                """,
                "x_axis": "avg_hours",
                "y_axis": "avg_output",
                "size": "avg_overtime",
            },
        ],
    },

    "hr_intervention_log": {
        "title": "HR Intervention & Action Log",
        "description": "Individual risk drivers, recommended interventions, action tracking",
        "access": "HRBP_ONLY",
        "charts": [
            {
                "id": "high_risk_employees",
                "type": "table",
                "title": "High Flight Risk Employees",
                "query": """
                    SELECT
                        rs.emp_id,
                        e.department,
                        e.job_role,
                        ROUND(rs.flight_risk_probability * 100, 1) AS risk_pct,
                        rs.risk_tier,
                        s.satisfaction_score,
                        s.work_life_balance_score
                    FROM risk_scores rs
                    JOIN dim_employee e ON e.emp_id = rs.emp_id
                    LEFT JOIN LATERAL (
                        SELECT satisfaction_score, work_life_balance_score
                        FROM fact_engagement_survey
                        WHERE emp_id = e.emp_id
                        ORDER BY survey_date DESC LIMIT 1
                    ) s ON TRUE
                    WHERE rs.risk_tier = 'HIGH'
                    ORDER BY rs.flight_risk_probability DESC
                """,
                "privacy_note": "Restricted to HRBP role — do not share externally",
            },
        ],
    },
}
