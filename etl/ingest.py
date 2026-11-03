"""
Multi-Source Data Ingestion Pipeline
Handles ingestion from HRIS, Jira, GitHub, LMS, and Badge Access systems.
Supports both API-based and file-based (CSV) ingestion modes.
"""
import logging
import os
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

from .config import SOURCE_CONFIG, PIPELINE_CONFIG

logger = logging.getLogger(__name__)


class BaseIngester:
    """Abstract base class for all data source ingesters."""

    def __init__(self, source_name: str):
        self.source_name = source_name
        self.ingested_count = 0

    def ingest(self) -> pd.DataFrame:
        raise NotImplementedError

    def log_result(self, df: pd.DataFrame):
        self.ingested_count = len(df)
        logger.info(f"[{self.source_name}] Ingested {self.ingested_count} records.")


class CSVIngester(BaseIngester):
    """
    File-based ingester for development/testing — reads from CSV seed files.
    In production, replace with API-based ingesters below.
    """

    def __init__(self, source_name: str, file_path: str):
        super().__init__(source_name)
        self.file_path = Path(file_path)

    def ingest(self) -> pd.DataFrame:
        if not self.file_path.exists():
            logger.warning(f"[{self.source_name}] File not found: {self.file_path}")
            return pd.DataFrame()
        df = pd.read_csv(self.file_path, low_memory=False)
        self.log_result(df)
        return df


class HRISIngester(BaseIngester):
    """
    Ingests employee master data from HRIS/Payroll system.
    Falls back to CSV if API not configured.
    """

    def __init__(self, csv_fallback_path: Optional[str] = None):
        super().__init__("HRIS")
        self.csv_fallback = csv_fallback_path

    def ingest(self) -> pd.DataFrame:
        if SOURCE_CONFIG.hris_api_key:
            return self._from_api()
        elif self.csv_fallback:
            logger.info("[HRIS] API key not set — using CSV fallback.")
            return CSVIngester("HRIS_CSV", self.csv_fallback).ingest()
        return pd.DataFrame()

    def _from_api(self) -> pd.DataFrame:
        """
        Production implementation: call HRIS REST API.
        Replace with actual API client (e.g., Workday, SAP SuccessFactors).
        """
        try:
            import requests
            headers = {"Authorization": f"Bearer {SOURCE_CONFIG.hris_api_key}"}
            resp = requests.get(
                f"{SOURCE_CONFIG.hris_api_url}/employees",
                headers=headers,
                timeout=30,
            )
            resp.raise_for_status()
            df = pd.DataFrame(resp.json()["employees"])
            self.log_result(df)
            return df
        except Exception as e:
            logger.error(f"[HRIS] API call failed: {e}")
            return pd.DataFrame()


class JiraIngester(BaseIngester):
    """
    Ingests ticket/task activity from Jira to build workload telemetry.
    """

    def __init__(self, csv_fallback_path: Optional[str] = None):
        super().__init__("Jira")
        self.csv_fallback = csv_fallback_path

    def ingest(self, lookback_days: int = PIPELINE_CONFIG.lookback_days) -> pd.DataFrame:
        if SOURCE_CONFIG.jira_token:
            return self._from_api(lookback_days)
        elif self.csv_fallback:
            logger.info("[Jira] Token not set — using CSV fallback.")
            return CSVIngester("Jira_CSV", self.csv_fallback).ingest()
        return pd.DataFrame()

    def _from_api(self, lookback_days: int) -> pd.DataFrame:
        """
        Production: query Jira REST API for resolved issues per assignee.
        """
        try:
            from jira import JIRA
            jira = JIRA(
                server=SOURCE_CONFIG.jira_url,
                basic_auth=(SOURCE_CONFIG.jira_user, SOURCE_CONFIG.jira_token),
            )
            since = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            issues = jira.search_issues(
                f'resolved >= "{since}" ORDER BY resolved ASC',
                maxResults=5000,
                fields="assignee,resolutiondate,issuetype,priority",
            )
            rows = []
            for issue in issues:
                rows.append({
                    "assignee_email": getattr(issue.fields.assignee, "emailAddress", None),
                    "resolved_date": issue.fields.resolutiondate[:10] if issue.fields.resolutiondate else None,
                    "issue_type": issue.fields.issuetype.name,
                    "priority": issue.fields.priority.name if issue.fields.priority else None,
                })
            df = pd.DataFrame(rows)
            self.log_result(df)
            return df
        except Exception as e:
            logger.error(f"[Jira] API call failed: {e}")
            return pd.DataFrame()


class GitHubIngester(BaseIngester):
    """
    Ingests commit activity from GitHub to track developer productivity.
    """

    def __init__(self, csv_fallback_path: Optional[str] = None):
        super().__init__("GitHub")
        self.csv_fallback = csv_fallback_path

    def ingest(self, lookback_days: int = PIPELINE_CONFIG.lookback_days) -> pd.DataFrame:
        if SOURCE_CONFIG.github_token:
            return self._from_api(lookback_days)
        elif self.csv_fallback:
            logger.info("[GitHub] Token not set — using CSV fallback.")
            return CSVIngester("GitHub_CSV", self.csv_fallback).ingest()
        return pd.DataFrame()

    def _from_api(self, lookback_days: int) -> pd.DataFrame:
        try:
            from github import Github
            g = Github(SOURCE_CONFIG.github_token)
            org = g.get_organization(SOURCE_CONFIG.github_org)
            since = datetime.now() - timedelta(days=lookback_days)
            rows = []
            for repo in org.get_repos():
                for commit in repo.get_commits(since=since):
                    rows.append({
                        "author_email": commit.commit.author.email,
                        "commit_date": commit.commit.author.date.date().isoformat(),
                        "repo": repo.name,
                        "additions": commit.stats.additions,
                        "deletions": commit.stats.deletions,
                    })
            df = pd.DataFrame(rows)
            self.log_result(df)
            return df
        except Exception as e:
            logger.error(f"[GitHub] API call failed: {e}")
            return pd.DataFrame()


class BadgeAccessIngester(BaseIngester):
    """
    Ingests physical badge/access swipe data for attendance tracking.
    """

    def __init__(self, csv_fallback_path: Optional[str] = None):
        super().__init__("BadgeAccess")
        self.csv_fallback = csv_fallback_path

    def ingest(self) -> pd.DataFrame:
        if self.csv_fallback:
            return CSVIngester("Badge_CSV", self.csv_fallback).ingest()
        logger.warning("[BadgeAccess] No data source configured.")
        return pd.DataFrame()


class IngestionOrchestrator:
    """
    Orchestrates all ingesters and writes results to the raw data store.
    """

    def __init__(self, data_dir: str = SOURCE_CONFIG.data_dir):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.seed_dir = Path("./database/seed")

    def run(self) -> dict[str, pd.DataFrame]:
        """Run all ingesters and return a dict of DataFrames."""
        logger.info("Starting ingestion pipeline...")
        results = {}

        ingesters = {
            "employees": HRISIngester(self.seed_dir / "dim_employee.csv"),
            "telemetry": JiraIngester(self.seed_dir / "fact_workload_telemetry.csv"),
            "performance": CSVIngester("Performance", self.seed_dir / "fact_performance.csv"),
            "surveys": CSVIngester("Surveys", self.seed_dir / "fact_engagement_survey.csv"),
            "attrition": CSVIngester("Attrition", self.seed_dir / "fact_attrition_labels.csv"),
        }

        for name, ingester in ingesters.items():
            try:
                df = ingester.ingest()
                if not df.empty:
                    out_path = self.data_dir / f"{name}.parquet"
                    df.to_parquet(out_path, index=False)
                    logger.info(f"  Saved {name} -> {out_path}")
                results[name] = df
            except Exception as e:
                logger.error(f"  Failed to ingest {name}: {e}")
                results[name] = pd.DataFrame()

        logger.info("Ingestion pipeline complete.")
        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    orchestrator = IngestionOrchestrator()
    data = orchestrator.run()
    for name, df in data.items():
        print(f"{name}: {len(df)} rows")
