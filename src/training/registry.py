"""MLflow Model Registry helpers.

Wraps registration, stage promotion, and production model loading.
Supports both the legacy Stages API (MLflow < 2.9) and the newer Aliases
API (MLflow >= 2.9).
"""

from __future__ import annotations

import os

import mlflow
from loguru import logger
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

_mlflow_version = tuple(int(x) for x in mlflow.__version__.split(".")[:2])
_USE_ALIASES = _mlflow_version >= (2, 9)


def _client() -> MlflowClient:
    return MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_model(
    run_id: str,
    model_name: str = "tire-cliff-xgboost",
    artifact_path: str = "model",
) -> int:
    """Register a logged model artifact into the MLflow Model Registry.

    Returns the new model version number.
    """
    model_uri = f"runs:/{run_id}/{artifact_path}"
    result = mlflow.register_model(model_uri, model_name)
    version = int(result.version)
    logger.info("Registered {} version {} from run {}", model_name, version, run_id)
    return version


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------


def promote_model(
    model_name: str,
    version: int,
    stage: str = "Staging",
) -> None:
    """Promote a model version to *Staging* or *Production*.

    Uses aliases on MLflow >= 2.9, falls back to the legacy stages API
    on older versions.
    """
    client = _client()

    if _USE_ALIASES:
        alias = stage.lower()
        client.set_registered_model_alias(model_name, alias, str(version))
        logger.info("Set alias '{}' on {} version {}", alias, model_name, version)
    else:
        client.transition_model_version_stage(
            name=model_name,
            version=str(version),
            stage=stage,
            archive_existing_versions=True,
        )
        logger.info("Transitioned {} version {} to {}", model_name, version, stage)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def get_production_model(model_name: str = "tire-cliff-xgboost"):
    """Load the current Production model for comparison or inference.

    Returns the deserialized model object, or *None* if no production model
    is registered.
    """
    client = _client()

    if _USE_ALIASES:
        try:
            mv = client.get_model_version_by_alias(model_name, "production")
            model_uri = f"models:/{model_name}@production"
        except mlflow.exceptions.MlflowException:
            logger.warning("No 'production' alias found for {}", model_name)
            return None
    else:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if not versions:
            logger.warning("No Production version found for {}", model_name)
            return None
        mv = versions[0]
        model_uri = f"models:/{model_name}/Production"

    logger.info("Loading {} version {} ({})", model_name, mv.version, model_uri)
    return mlflow.pyfunc.load_model(model_uri)
