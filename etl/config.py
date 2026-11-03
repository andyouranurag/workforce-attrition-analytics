"""
ETL Configuration — data source connections and pipeline settings.
"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DatabaseConfig:
    host: str = os.getenv("PG_HOST", "localhost")
    port: int = int(os.getenv("PG_PORT", "5432"))
    database: str = os.getenv("PG_DATABASE", "workforce_analytics")
    user: str = os.getenv("PG_USER", "postgres")
    password: str = os.getenv("PG_PASSWORD", "")

    @property
    def connection_string(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


@dataclass
class SourceConfig:
    """Configuration for data source connectors."""
    # HRIS / Payroll system
    hris_api_url: str = os.getenv("HRIS_API_URL", "http://localhost:8080/api/hris")
    hris_api_key: str = os.getenv("HRIS_API_KEY", "")

    # Jira
    jira_url: str = os.getenv("JIRA_URL", "https://your-org.atlassian.net")
    jira_user: str = os.getenv("JIRA_USER", "")
    jira_token: str = os.getenv("JIRA_TOKEN", "")

    # GitHub / GitLab
    github_token: str = os.getenv("GITHUB_TOKEN", "")
    github_org: str = os.getenv("GITHUB_ORG", "")

    # LMS (Learning Management System)
    lms_api_url: str = os.getenv("LMS_API_URL", "")
    lms_api_key: str = os.getenv("LMS_API_KEY", "")

    # Data directory (for CSV/file-based ingestion)
    data_dir: str = os.getenv("DATA_DIR", "./data/raw")


@dataclass
class PipelineConfig:
    batch_size: int = 1000
    lookback_days: int = 90          # Rolling window for telemetry aggregation
    attrition_threshold: float = 0.75  # Probability above which we flag high risk
    model_path: str = "./models/attrition_model.json"
    feature_store_path: str = "./data/processed/feature_store.parquet"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


# Singleton configs
DB_CONFIG = DatabaseConfig()
SOURCE_CONFIG = SourceConfig()
PIPELINE_CONFIG = PipelineConfig()
