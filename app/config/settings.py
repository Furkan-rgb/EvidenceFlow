"""Environment-backed settings and stable local data paths."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings; model selection remains in ``config/models.yaml``."""

    model_config = SettingsConfigDict(env_file=".env")

    data_dir: Path = Field(Path("data"), alias="EVIDENCEFLOW_DATA_DIR")
    rules_config: Path = Field(
        Path("config/review_rules.yaml"), alias="EVIDENCEFLOW_RULES_CONFIG"
    )
    policies_dir: Path = Field(Path("policies"), alias="EVIDENCEFLOW_POLICIES_DIR")
    ollama_base_url: str | None = Field(None, alias="OLLAMA_BASE_URL")
    max_file_bytes: int = Field(10 * 1024 * 1024, alias="EVIDENCEFLOW_MAX_FILE_BYTES")
    max_bundle_bytes: int = Field(25 * 1024 * 1024, alias="EVIDENCEFLOW_MAX_BUNDLE_BYTES")
    max_pages: int = Field(50, alias="EVIDENCEFLOW_MAX_PAGES")
    mlflow_enabled: bool = Field(True, alias="EVIDENCEFLOW_MLFLOW_ENABLED")
    mlflow_log_content: bool = Field(False, alias="EVIDENCEFLOW_MLFLOW_LOG_CONTENT")
    log_sensitive_content: bool = Field(
        False, alias="EVIDENCEFLOW_LOG_SENSITIVE_CONTENT"
    )
    mlflow_tracking_uri: str = Field(
        "http://127.0.0.1:5001", alias="MLFLOW_TRACKING_URI"
    )
    mlflow_experiment_name: str = Field("evidenceflow", alias="MLFLOW_EXPERIMENT_NAME")

    @property
    def database_path(self) -> Path:
        return self.data_dir / "evidenceflow.db"

    @property
    def checkpoints_path(self) -> Path:
        return self.data_dir / "checkpoints.db"

    @property
    def policy_index_path(self) -> Path:
        return self.data_dir / "policy_index.db"

    @property
    def policy_manifest_path(self) -> Path:
        return self.data_dir / "policy_index_manifest.json"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def exports_dir(self) -> Path:
        return self.data_dir / "exports"

    def ensure_directories(self) -> None:
        for path in (self.data_dir, self.uploads_dir, self.exports_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
