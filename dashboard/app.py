"""
Workforce Attrition & Productivity Analytics — BI Dashboard
Built with Streamlit and Plotly.
Provides Executive Overview, Manager Portal, HR Action Log, and Interactive Simulator.
"""
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Workforce Attrition Analytics",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Data Loader
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    raw_dir = ROOT_DIR / "data" / "raw"
    seed_dir = ROOT_DIR / "database" / "seed"

    # 1. Load employees
    if (raw_dir / "employees.parquet").exists():
        df_emp = pd.read_parquet(raw_dir / "employees.parquet")
    elif (seed_dir / "dim_employee.csv").exists():
        df_emp = pd.read_csv(seed_dir / "dim_employee.csv")
    else:
        st.error("No employee data found. Please run: python database/seed_data.py")
        st.stop()

    # 2. Load telemetry
    if (raw_dir / "telemetry.parquet").exists():
        df_tel = pd.read_parquet(raw_dir / "telemetry.parquet")
    else:
        df_tel = pd.read_csv(seed_dir / "fact_workload_telemetry.csv")

    # 3. Load surveys
    if (raw_dir / "surveys.parquet").exists():
        df_surv = pd.read_parquet(raw_dir / "surveys.parquet")
    else:
        df_surv = pd.read_csv(seed_dir / "fact_engagement_survey.csv")

    # 4. Load attrition
    if (raw_dir / "attrition.parquet").exists():
        df_attr = pd.read_parquet(raw_dir / "attrition.parquet")
    else:
        df_attr = pd.read_csv(seed_dir / "fact_attrition_labels.csv")

    # 5. Load performance
    if (raw_dir / "performance.parquet").exists():
        df_perf = pd.read_parquet(raw_dir / "performance.parquet")
    else:
        df_perf = pd.read_csv(seed_dir / "fact_performance.csv")

    # Convert dates
    df_emp["hire_date"] = pd.to_datetime(df_emp["hire_date"])
    df_tel["log_date"] = pd.to_datetime(df_tel["log_date"])
    df_surv["survey_date"] = pd.to_datetime(df_surv["survey_date"])
    df_attr["left_company_flag"] = df_attr["left_company_flag"].astype(int)

    # Compute tenure
    ref_date = pd.Timestamp.now()
    df_emp["tenure_years"] = ((ref_date - df_emp["hire_date"]).dt.days / 365.25).round(1)

    # Telemetry 90-day rollups
    cutoff = df_tel["log_date"].max() - pd.Timedelta(days=90)
    recent_tel = df_tel[df_tel["log_date"] >= cutoff]

    tel_agg = recent_tel.groupby("emp_id").agg(
        avg_overtime_90d=("overtime_hours", "mean"),
        absence_trend_90d=("unplanned_absence_flag", "sum"),
        avg_hours_worked=("hours_worked", "mean"),
        avg_tasks_closed=("tasks_closed", "mean"),
    ).reset_index()

    # Latest survey
    latest_surv = df_surv.sort_values("survey_date").groupby("emp_id").last().reset_index()

    # Latest performance
    latest_perf = df_perf.sort_values("review_date").groupby("emp_id").last().reset_index()

    # Master join
    master = df_emp.merge(tel_agg, on="emp_id", how="left")
    master = master.merge(latest_surv, on="emp_id", how="left")
    master = master.merge(latest_perf, on="emp_id", how="left")
    master = master.merge(df_attr[["emp_id", "left_company_flag"]], on="emp_id", how="left")

    # Check for processed risk scores
    risk_file = ROOT_DIR / "data" / "processed" / "risk_scores.csv"
    if risk_file.exists():
        df_risk = pd.read_csv(risk_file)
        if "emp_id" in df_risk.columns:
            master = master.merge(df_risk[["emp_id", "flight_risk_probability", "risk_tier"]], on="emp_id", how="left")
        else:
            master["flight_risk_probability"] = df_risk["flight_risk_probability"].values
            master["risk_tier"] = df_risk["risk_tier"].values
    else:
        # Fallback heuristic calculation if model hasn't been executed yet
        overtime_comp = master["avg_overtime_90d"].fillna(1.0).clip(0, 5) / 5.0 * 0.35
        sat_comp = (5.0 - master["satisfaction_score"].fillna(3.5)).clip(0, 4) / 4.0 * 0.35
        absence_comp = master["absence_trend_90d"].fillna(1).clip(0, 6) / 6.0 * 0.20
        tenure_comp = (master["tenure_years"].between(1, 3)).astype(float) * 0.10

        probs = (overtime_comp + sat_comp + absence_comp + tenure_comp).clip(0.05, 0.95).round(3)
        master["flight_risk_probability"] = probs
        master["risk_tier"] = pd.cut(
            probs,
            bins=[-np.inf, 0.40, 0.75, np.inf],
            labels=["LOW", "MEDIUM", "HIGH"]
        )

    return master, df_tel


master_df, telemetry_df = load_data()

# -----------------------------------------------------------------------------
# Sidebar Navigation & Filters
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/group.png", width=64)
    st.title("Workforce BI")
    st.caption("Predictive Flight Risk & Productivity")

    selected_view = st.radio(
        "Navigation View",
        [
            "🏢 Executive Overview",
            "👥 Manager Team Portal",
            "📋 HR Intervention Log",
            "🧪 Flight Risk Simulator",
        ],
    )

    st.divider()
    st.subheader("Global Filters")
    dept_options = ["All Departments"] + sorted(master_df["department"].dropna().unique().tolist())
    selected_dept = st.selectbox("Department", dept_options)

    if selected_dept != "All Departments":
        filtered_df = master_df[master_df["department"] == selected_dept]
    else:
        filtered_df = master_df

    st.info(f"Showing **{len(filtered_df):,}** employees")


# -----------------------------------------------------------------------------
# VIEW 1: Executive Overview (Macro Health)
# -----------------------------------------------------------------------------
if selected_view == "🏢 Executive Overview":
    st.markdown('<div class="main-header">🏢 Executive Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Macro workforce health, flight risk saturation, and departmental burnout hotspots</div>', unsafe_allow_html=True)

    # Top KPI Metrics
    total_headcount = len(filtered_df)
    high_risk_count = (filtered_df["risk_tier"] == "HIGH").sum()
    high_risk_pct = (high_risk_count / total_headcount * 100) if total_headcount > 0 else 0
    historical_attrition_pct = (filtered_df["left_company_flag"].sum() / total_headcount * 100) if total_headcount > 0 else 0
    avg_overtime = filtered_df["avg_overtime_90d"].mean()
    avg_satisfaction = filtered_df["satisfaction_score"].mean()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Headcount", f"{total_headcount:,}")
    col2.metric(
        "High Flight Risk (>75%)",
        f"{high_risk_pct:.1f}%",
        f"{high_risk_count} employees",
        delta_color="inverse"
    )
    col3.metric("Historical Attrition", f"{historical_attrition_pct:.1f}%")
    col4.metric("Avg Daily Overtime", f"{avg_overtime:.2f} hrs", delta_color="inverse")
    col5.metric("Avg Satisfaction", f"{avg_satisfaction:.2f} / 5.0")

    st.divider()

    col_gauge, col_trend = st.columns([1, 2])

    with col_gauge:
        st.subheader("Company-Wide Risk Meter")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=high_risk_pct,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "High Flight Risk %", 'font': {'size': 18}},
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#E63946"},
                'steps': [
                    {'range': [0, 15], 'color': "#D8F3DC"},
                    {'range': [15, 30], 'color': "#FFF3B0"},
                    {'range': [30, 100], 'color': "#FFCCD5"},
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 25
                }
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_trend:
        st.subheader("Flight Risk Tier Distribution by Department")
        risk_by_dept = master_df.groupby(["department", "risk_tier"]).size().reset_index(name="count")
        fig_bar = px.bar(
            risk_by_dept,
            x="department",
            y="count",
            color="risk_tier",
            color_discrete_map={"LOW": "#2A9D8F", "MEDIUM": "#E9C46A", "HIGH": "#E76F51"},
            barmode="stack",
            height=300,
        )
        fig_bar.update_layout(margin=dict(l=20, r=20, t=20, b=20), legend_title_text="Risk Tier")
        st.plotly_chart(fig_bar, use_container_width=True)

    # Burnout Hotspot Matrix
    st.subheader("🔥 Burnout Hotspot Matrix (Department vs. Month Overtime)")
    tel_with_dept = telemetry_df.merge(master_df[["emp_id", "department"]], on="emp_id", how="left")
    tel_with_dept["month"] = tel_with_dept["log_date"].dt.strftime("%Y-%m")
    heatmap_data = tel_with_dept.groupby(["department", "month"])["overtime_hours"].mean().unstack(fill_value=0)

    # Keep last 6 months
    heatmap_data = heatmap_data.iloc[:, -6:]

    fig_heat = px.imshow(
        heatmap_data,
        labels=dict(x="Month", y="Department", color="Avg Overtime (hrs)"),
        color_continuous_scale="Reds",
        aspect="auto",
        text_auto=".1f",
    )
    fig_heat.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_heat, use_container_width=True)

    # Department Health Scorecard
    st.subheader("Department Health Scorecard")
    dept_health = master_df.groupby("department").agg(
        Headcount=("emp_id", "count"),
        High_Risk_Count=("risk_tier", lambda x: (x == "HIGH").sum()),
        Avg_Overtime=("avg_overtime_90d", "mean"),
        Avg_Satisfaction=("satisfaction_score", "mean"),
        Avg_Work_Life_Balance=("work_life_balance_score", "mean"),
    ).reset_index()
    dept_health["High_Risk_%"] = (dept_health["High_Risk_Count"] / dept_health["Headcount"] * 100).round(1)
    dept_health["Avg_Overtime"] = dept_health["Avg_Overtime"].round(2)
    dept_health["Avg_Satisfaction"] = dept_health["Avg_Satisfaction"].round(2)
    dept_health["Avg_Work_Life_Balance"] = dept_health["Avg_Work_Life_Balance"].round(2)

    st.dataframe(
        dept_health.sort_values(by="High_Risk_%", ascending=False),
        use_container_width=True,
        hide_index=True,
    )


# -----------------------------------------------------------------------------
# VIEW 2: Manager Team Portal (Team Insight)
# -----------------------------------------------------------------------------
elif selected_view == "👥 Manager Team Portal":
    st.markdown('<div class="main-header">👥 Manager Team Portal</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Team-level retention breakdown, individual flight risk tiers, and workload vs. output fatigue tracking</div>', unsafe_allow_html=True)

    col_m1, col_m2 = st.columns([1, 1])

    with col_m1:
        st.subheader("Team Retention Risk Breakdown")
        tier_counts = filtered_df["risk_tier"].value_counts().reset_index()
        tier_counts.columns = ["Tier", "Count"]

        fig_pie = px.pie(
            tier_counts,
            names="Tier",
            values="Count",
            color="Tier",
            color_discrete_map={"LOW": "#2A9D8F", "MEDIUM": "#E9C46A", "HIGH": "#E76F51"},
            hole=0.45,
            height=320,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_m2:
        st.subheader("⚡ Workload vs. Output Ratio (Fatigue Indicator)")
        st.caption("Tracks fatigue: when hours worked climb while task completion drops.")
        fig_scatter = px.scatter(
            filtered_df,
            x="avg_hours_worked",
            y="avg_tasks_closed",
            color="risk_tier",
            size="avg_overtime_90d",
            hover_data=["emp_id", "job_role", "base_salary"],
            color_discrete_map={"LOW": "#2A9D8F", "MEDIUM": "#E9C46A", "HIGH": "#E76F51"},
            labels={
                "avg_hours_worked": "Average Daily Hours Worked",
                "avg_tasks_closed": "Daily Tasks Closed",
                "avg_overtime_90d": "Overtime",
            },
            height=320,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()

    st.subheader("Team Member Roster & Health Overview")
    display_cols = [
        "emp_id", "department", "job_role", "job_level", "tenure_years",
        "avg_overtime_90d", "absence_trend_90d", "satisfaction_score",
        "flight_risk_probability", "risk_tier"
    ]
    st.dataframe(
        filtered_df[display_cols].sort_values(by="flight_risk_probability", ascending=False),
        use_container_width=True,
        hide_index=True,
    )


# -----------------------------------------------------------------------------
# VIEW 3: HR Intervention & Action Log
# -----------------------------------------------------------------------------
elif selected_view == "📋 HR Intervention Log":
    st.markdown('<div class="main-header">📋 HR Intervention & Action Log</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automated actionable triggers to catch burnout early and initiate proactive retention strategies</div>', unsafe_allow_html=True)

    from models.model import generate_interventions

    high_risk_employees = filtered_df[filtered_df["risk_tier"] == "HIGH"].sort_values(
        by="flight_risk_probability", ascending=False
    )

    st.write(f"Showing **{len(high_risk_employees)}** personnel flagged as High Flight Risk (> 75% probability):")

    for _, emp in high_risk_employees.head(15).iterrows():
        emp_dict = emp.to_dict()
        interventions = generate_interventions(emp_dict)

        with st.expander(f"🔴 **{emp['emp_id']}** — {emp['job_role']} ({emp['department']}) | Flight Risk: **{emp['flight_risk_probability']*100:.1f}%**"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Tenure", f"{emp['tenure_years']} yrs")
            c2.metric("Avg Overtime (90d)", f"{emp['avg_overtime_90d']:.1f} hrs/day")
            c3.metric("Absences (90d)", f"{int(emp['absence_trend_90d'])} days")
            c4.metric("Satisfaction Score", f"{emp['satisfaction_score']:.1f} / 5.0")

            st.markdown("##### 🚨 Primary Flight Risk Drivers (SHAP & Telemetry)")
            drivers = []
            if emp.get("avg_overtime_90d", 0) > 2.5:
                drivers.append(f"• **High Sustained Overtime**: Averages {emp['avg_overtime_90d']:.1f} overtime hours/day over 90 days.")
            if emp.get("absence_trend_90d", 0) > 3:
                drivers.append(f"• **Unplanned Absence Spike**: {int(emp['absence_trend_90d'])} unplanned absences in last quarter.")
            if emp.get("satisfaction_score", 5) < 3.0:
                drivers.append(f"• **Low Pulse Satisfaction**: Recorded {emp['satisfaction_score']:.1f}/5.0 in recent survey.")
            if emp.get("compensation_fairness_score", 5) < 2.8:
                drivers.append(f"• **Perceived Undercompensation**: Compensation fairness score of {emp['compensation_fairness_score']:.1f}/5.0.")
            if not drivers:
                drivers.append("• Multifactor behavioral shift detected across telemetry and performance.")

            for d in drivers:
                st.write(d)

            st.markdown("##### 💡 Recommended Interventions")
            if interventions:
                for act in interventions:
                    badge_color = "🔴" if act['priority'] == "HIGH" else "🟡"
                    st.write(f"{badge_color} **[{act['priority']}]** {act['intervention']}")
            else:
                st.write("• Conduct scheduled quarterly career progression check-in.")


# -----------------------------------------------------------------------------
# VIEW 4: Flight Risk Simulator ("What-If" Analysis)
# -----------------------------------------------------------------------------
elif selected_view == "🧪 Flight Risk Simulator":
    st.markdown('<div class="main-header">🧪 Live Flight Risk Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Adjust behavioral cues, workload intensity, and survey metrics to calculate flight risk in real time</div>', unsafe_allow_html=True)

    sim_col1, sim_col2 = st.columns([1, 1])

    with sim_col1:
        st.subheader("Employee Profile Inputs")
        sim_tenure = st.slider("Tenure (Years)", min_value=0.2, max_value=15.0, value=2.5, step=0.1)
        sim_overtime = st.slider("90-Day Avg Overtime (Hours/Day)", min_value=0.0, max_value=6.0, value=3.2, step=0.1)
        sim_absences = st.slider("Unplanned Absences (Last 90 Days)", min_value=0, max_value=15, value=4, step=1)
        sim_satisfaction = st.slider("Satisfaction Score (1 = Low, 5 = High)", min_value=1.0, max_value=5.0, value=2.2, step=0.1)
        sim_wlb = st.slider("Work-Life Balance Score", min_value=1.0, max_value=5.0, value=1.8, step=0.1)
        sim_comp = st.slider("Compensation Fairness Score", min_value=1.0, max_value=5.0, value=2.5, step=0.1)
        sim_promotions = st.selectbox("Promotions in Career", [0, 1, 2, 3], index=0)

    with sim_col2:
        st.subheader("Prediction Output")

        # Heuristic scoring formula matching model weights
        ot_w = (sim_overtime / 5.0) * 0.32
        sat_w = ((5.0 - sim_satisfaction) / 4.0) * 0.30
        wlb_w = ((5.0 - sim_wlb) / 4.0) * 0.18
        abs_w = (sim_absences / 10.0) * 0.12
        promo_w = 0.08 if (sim_promotions == 0 and sim_tenure > 2.0) else 0.0

        calculated_risk = min(0.98, max(0.04, ot_w + sat_w + wlb_w + abs_w + promo_w))
        tier = "HIGH" if calculated_risk >= 0.75 else ("MEDIUM" if calculated_risk >= 0.40 else "LOW")
        color = "#E63946" if tier == "HIGH" else ("#E9C46A" if tier == "MEDIUM" else "#2A9D8F")

        st.metric(
            "Predicted Flight Risk Probability",
            f"{calculated_risk * 100:.1f}%",
            f"Risk Tier: {tier}",
            delta_color="inverse" if tier == "HIGH" else "normal"
        )

        st.progress(calculated_risk)

        st.markdown(f"""
        <div style="background-color: {color}15; border-left: 4px solid {color}; padding: 12px; border-radius: 4px; margin-top: 15px;">
            <b style="color: {color};">Status: {tier} RISK</b><br>
            {'⚠️ Immediate HR intervention recommended to prevent imminent resignation.' if tier == 'HIGH' else 'Monitor workload telemetry and schedule regular 1-on-1s.'}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Real-Time Recommended Interventions")
        sim_emp_dict = {
            "avg_overtime_90d": sim_overtime,
            "absence_trend_90d": sim_absences,
            "satisfaction_score": sim_satisfaction,
            "work_life_balance_score": sim_wlb,
            "compensation_fairness_score": sim_comp,
            "promotions_count": sim_promotions,
            "tenure_years": sim_tenure,
        }
        from models.model import generate_interventions
        sim_actions = generate_interventions(sim_emp_dict)
        for act in sim_actions:
            st.write(f"• **[{act['priority']}]** {act['intervention']}")
