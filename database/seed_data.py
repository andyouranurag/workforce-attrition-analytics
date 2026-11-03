"""
Synthetic Data Generator for Workforce Attrition Analytics
Generates realistic HR data for development and testing.
"""
import random
import csv
import os
from datetime import date, timedelta
from pathlib import Path

random.seed(42)

OUTPUT_DIR = Path(__file__).parent / "seed"
OUTPUT_DIR.mkdir(exist_ok=True)

DEPARTMENTS = ["Engineering", "Sales", "HR", "Finance", "Marketing", "Operations", "Product", "Legal"]
JOB_ROLES = {
    "Engineering": ["Software Engineer", "Senior Engineer", "Tech Lead", "DevOps Engineer"],
    "Sales":       ["Sales Rep", "Account Executive", "Sales Manager", "BDR"],
    "HR":          ["HR Generalist", "Recruiter", "HRBP", "Compensation Analyst"],
    "Finance":     ["Financial Analyst", "Accountant", "FP&A Manager", "Controller"],
    "Marketing":   ["Marketing Specialist", "Content Writer", "SEO Analyst", "Campaign Manager"],
    "Operations":  ["Ops Analyst", "Project Manager", "Process Engineer", "Logistics Coordinator"],
    "Product":     ["Product Manager", "Product Analyst", "UX Designer", "Product Owner"],
    "Legal":       ["Legal Counsel", "Paralegal", "Compliance Officer", "Contract Specialist"],
}
JOB_LEVELS = ["Junior", "Mid", "Senior", "Lead", "Manager", "Director"]
EDUCATION = ["High School", "Associate", "Bachelor", "Master", "PhD"]
EMPLOYMENT_TYPES = ["Full-Time", "Part-Time", "Contract"]
GENDERS = ["Male", "Female", "Non-Binary", "Prefer not to say"]
MARITAL_STATUSES = ["Single", "Married", "Divorced", "Widowed"]
LOCATIONS = ["Bangalore", "Mumbai", "Delhi", "Hyderabad", "Chennai", "Pune", "Remote"]

EXIT_REASONS = [
    "Better opportunity", "Personal reasons", "Relocation", "Higher compensation",
    "Career change", "Work-life balance", "Manager conflict", "Company culture"
]

NUM_EMPLOYEES = 500
START_DATE = date(2019, 1, 1)
END_DATE = date(2024, 12, 31)


def random_date(start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def generate_employees():
    employees = []
    manager_pool = []
    for i in range(1, NUM_EMPLOYEES + 1):
        dept = random.choice(DEPARTMENTS)
        roles = JOB_ROLES[dept]
        role = random.choice(roles)
        level = random.choice(JOB_LEVELS)
        hire_date = random_date(START_DATE, END_DATE - timedelta(days=180))
        # Salary correlated with level
        base = {"Junior": 400000, "Mid": 600000, "Senior": 900000, "Lead": 1200000, "Manager": 1500000, "Director": 2000000}
        salary = base[level] + random.randint(-50000, 150000)
        emp_id = f"EMP{i:04d}"
        manager_id = random.choice(manager_pool) if manager_pool and random.random() > 0.1 else None
        if level in ["Manager", "Director", "Lead"]:
            manager_pool.append(emp_id)
        employees.append({
            "emp_id": emp_id,
            "first_name": f"Employee{i}",
            "last_name": f"LastName{i}",
            "hire_date": hire_date,
            "department": dept,
            "job_role": role,
            "job_level": level,
            "base_salary": salary,
            "manager_id": manager_id or "",
            "distance_from_home": round(random.uniform(1, 80), 2),
            "education_level": random.choice(EDUCATION),
            "employment_type": random.choices(EMPLOYMENT_TYPES, weights=[85, 10, 5])[0],
            "gender": random.choice(GENDERS),
            "marital_status": random.choice(MARITAL_STATUSES),
            "location": random.choice(LOCATIONS),
        })
    return employees


def generate_attrition_labels(employees):
    labels = []
    for emp in employees:
        # ~18% attrition rate (realistic for tech industry)
        left = random.random() < 0.18
        exit_date = None
        voluntary = None
        exit_reason = None
        if left:
            hire = emp["hire_date"]
            min_exit = hire + timedelta(days=90)
            exit_date = random_date(min_exit, END_DATE)
            voluntary = random.random() < 0.75  # 75% voluntary
            exit_reason = random.choice(EXIT_REASONS) if voluntary else "Performance/Termination"
        labels.append({
            "emp_id": emp["emp_id"],
            "left_company_flag": left,
            "exit_date": exit_date or "",
            "voluntary_flag": voluntary if voluntary is not None else "",
            "exit_reason": exit_reason or "",
        })
    return labels


def generate_performance(employees):
    records = []
    for emp in employees:
        hire = emp["hire_date"]
        # Generate 1-4 performance reviews per employee
        num_reviews = random.randint(1, 4)
        for j in range(num_reviews):
            review_date = hire + timedelta(days=365 * (j + 1))
            if review_date > END_DATE:
                break
            records.append({
                "emp_id": emp["emp_id"],
                "review_date": review_date,
                "performance_rating": round(random.gauss(3.5, 0.8), 1),
                "projects_completed": random.randint(1, 12),
                "promotions_count": random.randint(0, 2),
                "training_hours": round(random.uniform(0, 80), 1),
                "peer_review_score": round(random.gauss(3.5, 0.7), 1),
                "manager_rating": round(random.gauss(3.4, 0.9), 1),
            })
    return records


def generate_workload_telemetry(employees, attrition_map):
    records = []
    for emp in employees:
        hire = emp["hire_date"]
        left_flag = attrition_map[emp["emp_id"]]["left_company_flag"]
        exit_date = attrition_map[emp["emp_id"]]["exit_date"]
        end = exit_date if left_flag and exit_date else END_DATE
        # Sample ~60 random working days per employee (not all days for perf)
        num_days = min(60, (end - hire).days)
        if num_days <= 0:
            continue
        for _ in range(num_days):
            log_date = random_date(hire, end)
            overtime = max(0, round(random.gauss(1.5, 1.2), 2))
            records.append({
                "emp_id": emp["emp_id"],
                "log_date": log_date,
                "hours_worked": round(random.gauss(8, 1.5), 2),
                "overtime_hours": overtime,
                "tasks_closed": random.randint(0, 8),
                "tasks_opened": random.randint(0, 6),
                "commits_count": random.randint(0, 15),
                "tickets_resolved": random.randint(0, 5),
                "unplanned_absence_flag": random.random() < 0.05,
                "late_login_flag": random.random() < 0.08,
                "badge_swipe_count": random.randint(0, 4),
            })
    return records


def generate_surveys(employees):
    records = []
    for emp in employees:
        hire = emp["hire_date"]
        num_surveys = random.randint(1, 3)
        for j in range(num_surveys):
            survey_date = hire + timedelta(days=random.randint(30, 600) + j * 300)
            if survey_date > END_DATE:
                break
            records.append({
                "emp_id": emp["emp_id"],
                "survey_date": survey_date,
                "satisfaction_score": max(1, min(5, round(random.gauss(3.4, 0.9), 1))),
                "work_life_balance_score": max(1, min(5, round(random.gauss(3.2, 0.9), 1))),
                "manager_relationship_score": max(1, min(5, round(random.gauss(3.5, 0.8), 1))),
                "career_growth_score": max(1, min(5, round(random.gauss(3.0, 1.0), 1))),
                "compensation_fairness_score": max(1, min(5, round(random.gauss(2.9, 0.9), 1))),
                "team_cohesion_score": max(1, min(5, round(random.gauss(3.6, 0.7), 1))),
                "survey_type": random.choice(["pulse", "annual"]),
            })
    return records


def write_csv(filename, records, fieldnames):
    path = OUTPUT_DIR / filename
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"  Written: {path} ({len(records)} rows)")


def main():
    print("Generating synthetic HR data...")
    employees = generate_employees()
    attrition = generate_attrition_labels(employees)
    attrition_map = {a["emp_id"]: a for a in attrition}
    performance = generate_performance(employees)
    telemetry = generate_workload_telemetry(employees, attrition_map)
    surveys = generate_surveys(employees)

    write_csv("dim_employee.csv", employees, list(employees[0].keys()))
    write_csv("fact_attrition_labels.csv", attrition, list(attrition[0].keys()))
    write_csv("fact_performance.csv", performance, list(performance[0].keys()))
    write_csv("fact_workload_telemetry.csv", telemetry, list(telemetry[0].keys()))
    write_csv("fact_engagement_survey.csv", surveys, list(surveys[0].keys()))
    print(f"\nDone! Seed data written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
