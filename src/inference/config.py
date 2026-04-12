"""Service configuration via environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MLFLOW_DB_PATH = _REPO_ROOT / "mlflow.db"
_DEFAULT_TRACKING_URI = "sqlite:///" + _MLFLOW_DB_PATH.resolve().as_posix()


class Settings(BaseSettings):
    model_config = {"env_prefix": ""}

    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", _DEFAULT_TRACKING_URI)
    model_name: str = "tire-cliff-xgboost"
    model_alias: str = "production"

    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_group_id: str = "inference-service"
    lap_topic: str = "lap_completed"
    rc_topic: str = "race_control"

    feast_repo_path: str = str(_REPO_ROOT / "feature_repo")

    confidence_z: float = 1.645  # 90% CI (z-score)
    residual_mae: float = 3.0  # fallback MAE when model doesn't support quantiles

    host: str = "0.0.0.0"
    port: int = 8000


def get_settings() -> Settings:
    return Settings()
