"""Base training harness for tire-cliff prediction models.

Provides data loading (via Feast or cached Parquet), temporal train/val/test
splitting, feature preparation, and MLflow experiment logging.  Concrete model
scripts (XGBoost, LSTM, etc.) import and extend the utilities here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import LabelEncoder

from src.features.constants import ENGINEERED_FEATURE_COLUMNS, TRAINING_DATA_DIR
from src.training.metrics import evaluate_all

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT = "tire-cliff-prediction"

LABELED_DATASET_PATH = TRAINING_DATA_DIR / "labeled_dataset.parquet"

TRAIN_SEASONS = {"2021", "2022", "2023"}
VAL_SEASON = "2024"
VAL_MAX_ROUND = 12
TEST_SEASON = "2024"

FEATURE_COLUMNS = [c for c in ENGINEERED_FEATURE_COLUMNS]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_training_data(path: Path | None = None, rebuild: bool = False) -> pd.DataFrame:
    """Load the labeled dataset, rebuilding from Feast if needed.

    Parameters
    ----------
    path:
        Explicit parquet path.  Defaults to ``data/training/labeled_dataset.parquet``.
    rebuild:
        If True, always regenerate via ``build_labeled_dataset`` even when a
        cached file exists.
    """
    if path is None:
        path = LABELED_DATASET_PATH

    if path.exists() and not rebuild:
        logger.info("Loading cached labeled dataset from {}", path)
        return pd.read_parquet(path, engine="pyarrow")

    logger.info("Building labeled dataset via Feast")
    from src.features.dataset import build_labeled_dataset

    return build_labeled_dataset(output_path=path)


# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------


def _parse_session_id(session_id: str) -> tuple[str, int]:
    """Extract (season, round_num) from a session_id like '2024_5_Race'."""
    parts = session_id.split("_")
    return parts[0], int(parts[1])


def split_by_time(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataset by season and round for temporal evaluation.

    Returns (train, validation, test) DataFrames:
    - Train:      2021--2023 (all rounds)
    - Validation:  2024 rounds 1--12
    - Test:        2024 rounds 13+
    """
    parsed = df["session_id"].apply(_parse_session_id)
    seasons = parsed.str[0]
    rounds = parsed.str[1]

    train_mask = seasons.isin(TRAIN_SEASONS)
    val_mask = (seasons == VAL_SEASON) & (rounds <= VAL_MAX_ROUND)
    test_mask = (seasons == TEST_SEASON) & (rounds > VAL_MAX_ROUND)

    train_df = df.loc[train_mask].copy()
    val_df = df.loc[val_mask].copy()
    test_df = df.loc[test_mask].copy()

    logger.info(
        "Split sizes — train: {} | val: {} | test: {}",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

_label_encoder: LabelEncoder | None = None


def prepare_features(
    df: pd.DataFrame,
    fit_encoder: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Select model features and encode categoricals.

    Parameters
    ----------
    df:
        Labeled DataFrame containing feature and target columns.
    fit_encoder:
        If True, fit the compound label encoder on this split (use for train).
        If False, transform using an already-fitted encoder (use for val/test).

    Returns
    -------
    (X, y) where X is a 2-D float array and y is ``laps_to_cliff``.
    """
    global _label_encoder  # noqa: PLW0603

    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    X_df = df[available].copy()

    if "compound" in df.columns:
        if fit_encoder:
            _label_encoder = LabelEncoder()
            X_df["compound"] = _label_encoder.fit_transform(
                df["compound"].fillna("UNKNOWN").astype(str)
            )
        elif _label_encoder is not None:
            X_df["compound"] = _label_encoder.transform(
                df["compound"].fillna("UNKNOWN").astype(str)
            )

    X = X_df.to_numpy(dtype=np.float32, na_value=np.nan)
    y = df["laps_to_cliff"].to_numpy(dtype=np.float32)
    return X, y


def get_feature_names(df: pd.DataFrame) -> list[str]:
    """Return the feature column names that ``prepare_features`` would use."""
    names = [c for c in FEATURE_COLUMNS if c in df.columns]
    if "compound" in df.columns:
        names.append("compound")
    return names


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------


def log_to_mlflow(
    model: Any,
    params: dict[str, Any],
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    model_type: str = "sklearn",
    artifacts: dict[str, str] | None = None,
    tags: dict[str, str] | None = None,
) -> str:
    """Log a completed training run to MLflow.

    Returns the MLflow run ID.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run() as run:
        mlflow.log_params(params)

        for prefix, metrics in [
            ("train", train_metrics),
            ("val", val_metrics),
            ("test", test_metrics),
        ]:
            for name, value in metrics.items():
                mlflow.log_metric(f"{prefix}_{name}", value)

        if model_type == "sklearn":
            mlflow.sklearn.log_model(model, artifact_path="model")
        elif model_type == "xgboost":
            mlflow.xgboost.log_model(model, artifact_path="model")
        elif model_type == "lightgbm":
            mlflow.lightgbm.log_model(model, artifact_path="model")

        if artifacts:
            for name, path in artifacts.items():
                mlflow.log_artifact(path, artifact_path=name)

        if tags:
            mlflow.set_tags(tags)

        run_id = run.info.run_id
        logger.info("Logged MLflow run {}", run_id)
        return run_id


# ---------------------------------------------------------------------------
# Experiment orchestrator
# ---------------------------------------------------------------------------


def run_experiment(
    model_fn: Callable[..., Any],
    params: dict[str, Any],
    experiment_name: str | None = None,
    model_type: str = "sklearn",
    data_path: Path | None = None,
    tags: dict[str, str] | None = None,
) -> str:
    """End-to-end: load data, split, train, evaluate, log to MLflow.

    Parameters
    ----------
    model_fn:
        Callable that accepts ``(X_train, y_train, params)`` and returns a
        fitted model with a ``.predict(X)`` method.
    params:
        Hyperparameters forwarded to *model_fn* and logged to MLflow.
    experiment_name:
        Override the default MLflow experiment name.
    model_type:
        One of ``"sklearn"``, ``"xgboost"``, ``"lightgbm"`` — controls which
        MLflow model logger is used.
    data_path:
        Optional override for the labeled dataset parquet.
    tags:
        Extra MLflow tags (e.g. model family, author).

    Returns
    -------
    The MLflow run ID.
    """
    if experiment_name:
        global MLFLOW_EXPERIMENT  # noqa: PLW0603
        MLFLOW_EXPERIMENT = experiment_name

    df = load_training_data(path=data_path)
    train_df, val_df, test_df = split_by_time(df)

    X_train, y_train = prepare_features(train_df, fit_encoder=True)
    X_val, y_val = prepare_features(val_df)
    X_test, y_test = prepare_features(test_df)

    logger.info("Training model with params: {}", params)
    model = model_fn(X_train, y_train, params)

    train_metrics = evaluate_all(y_train, model.predict(X_train), train_df)
    val_metrics = evaluate_all(y_val, model.predict(X_val), val_df)
    test_metrics = evaluate_all(y_test, model.predict(X_test), test_df)

    logger.info("Train metrics: {}", train_metrics)
    logger.info("Val metrics:   {}", val_metrics)
    logger.info("Test metrics:  {}", test_metrics)

    split_params = {
        **params,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
    }

    run_id = log_to_mlflow(
        model=model,
        params=split_params,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        model_type=model_type,
        tags=tags,
    )
    return run_id
