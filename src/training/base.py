"""Base training harness for tire-cliff prediction models.

Provides data loading (via Feast or cached Parquet), temporal train/val/test
splitting, feature preparation, and MLflow experiment logging.  Concrete model
scripts (XGBoost, LSTM, etc.) import and extend the utilities here.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from mlflow.tracking import MlflowClient
from sklearn.preprocessing import LabelEncoder

from src.features.constants import ENGINEERED_FEATURE_COLUMNS, TRAINING_DATA_DIR
from src.training.metrics import evaluate_all

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default: SQLite backend + local artifact root (no server). File-only ``file://.../mlruns``
# tracking is deprecated in recent MLflow; see env override below.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_MLFLOW_DB_PATH = _REPO_ROOT / "mlflow.db"
_MLARTIFACTS_DIR = _REPO_ROOT / "mlartifacts"
_DEFAULT_TRACKING_URI = "sqlite:///" + _MLFLOW_DB_PATH.resolve().as_posix()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", _DEFAULT_TRACKING_URI)
MLFLOW_EXPERIMENT = "tire-cliff-prediction"

LABELED_DATASET_PATH = TRAINING_DATA_DIR / "labeled_dataset.parquet"

TRAIN_SEASONS = {"2021", "2022", "2023"}
VAL_SEASON = "2024"
VAL_MAX_ROUND = 12
TEST_SEASON = "2024"

FEATURE_COLUMNS = [c for c in ENGINEERED_FEATURE_COLUMNS]


# ---------------------------------------------------------------------------
# ExperimentResult
# ---------------------------------------------------------------------------


@dataclass
class ExperimentResult:
    """All outputs from a single training run, for downstream analysis."""

    run_id: str
    model: Any
    feature_names: list[str]
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    test_df: pd.DataFrame
    train_metrics: dict[str, float] = field(default_factory=dict)
    val_metrics: dict[str, float] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)


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


def split_by_time_adaptive(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train/val/test by chronological ``(season, round)`` thirds of distinct race weekends.

    Use when :func:`split_by_time` matches nothing (e.g. labeled data is only 2025). This keeps
    a temporal ordering (early season → train, later → val/test) but is **not** the README
    benchmark split (2021–23 / 2024).
    """
    if df.empty or "session_id" not in df.columns:
        empty = df.iloc[[]].copy()
        return empty, empty, empty

    parsed = df["session_id"].apply(_parse_session_id)
    s = parsed.str[0]
    r = parsed.str[1]

    unique_pairs = pd.DataFrame({"_s": s, "_r": r}).drop_duplicates().sort_values(["_s", "_r"])
    keys: list[tuple[str, int]] = list(
        zip(unique_pairs["_s"].tolist(), unique_pairs["_r"].tolist(), strict=True)
    )
    n = len(keys)
    if n == 0:
        empty = df.iloc[[]].copy()
        return empty, empty, empty

    chunk_idxs = np.array_split(np.arange(n), 3)
    train_keys = {keys[i] for i in chunk_idxs[0]}
    val_keys = {keys[i] for i in chunk_idxs[1]}
    test_keys = {keys[i] for i in chunk_idxs[2]}

    row_tuples = list(zip(s.tolist(), r.tolist(), strict=True))
    train_mask = [t in train_keys for t in row_tuples]
    val_mask = [t in val_keys for t in row_tuples]
    test_mask = [t in test_keys for t in row_tuples]

    train_df = df.loc[train_mask].copy()
    val_df = df.loc[val_mask].copy()
    test_df = df.loc[test_mask].copy()

    logger.info(
        "Adaptive temporal split — train: {} | val: {} | test: {} ({} distinct weekends)",
        len(train_df),
        len(val_df),
        len(test_df),
        n,
    )
    return train_df, val_df, test_df


def temporal_split_diagnostics(df: pd.DataFrame) -> str:
    """Explain empty temporal splits (wrong seasons, empty file, bad session_id)."""
    n = len(df)
    if n == 0:
        return (
            f"Labeled dataset has 0 rows at `{LABELED_DATASET_PATH.resolve()}`. "
            "Run ingestion + `python -m src.features.dataset` (or "
            "`python -m src.training.baseline --rebuild-dataset` after Feast prep)."
        )
    if "session_id" not in df.columns:
        return "Column `session_id` is missing; engineering should set ids like `2024_5_Race`."

    sid = df["session_id"].astype(str)
    parts = sid.str.split("_", n=2, expand=True)
    if parts.shape[1] < 2:
        sample = sid.iloc[0] if len(sid) else ""
        return (
            f"{n} rows, but `session_id` is not `YYYY_<round>_...` (need at least two `_` segments). "
            f"Example: {sample!r}"
        )
    season_vc = parts[0].value_counts().head(10)
    seasons_str = ", ".join(f"{k!r}:{v}" for k, v in season_vc.items())
    return (
        f"{n} rows; leading `session_id` season token counts: {seasons_str}. "
        f"Train uses seasons {sorted(TRAIN_SEASONS)}; val/test use {VAL_SEASON} with round "
        f"≤{VAL_MAX_ROUND} (val) or >{VAL_MAX_ROUND} (test)."
    )


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

_label_encoder: LabelEncoder | None = None
# circuit_tire_limitation is a string in Feast / reference CSV ("rear", "front"); map to int codes.
_circuit_tire_categories: tuple[str, ...] | None = None


def _encode_circuit_tire_limitation_column(X_df: pd.DataFrame, fit_encoder: bool) -> None:
    """In-place: replace string ``circuit_tire_limitation`` with float codes for XGBoost."""
    global _circuit_tire_categories  # noqa: PLW0603

    if "circuit_tire_limitation" not in X_df.columns:
        return
    s = X_df["circuit_tire_limitation"].fillna("__nan__").astype(str)
    if fit_encoder:
        _circuit_tire_categories = tuple(sorted(s.unique()))
    if not _circuit_tire_categories:
        X_df["circuit_tire_limitation"] = np.float32(-1.0)
        return
    codes = pd.Categorical(s, categories=list(_circuit_tire_categories), ordered=False).codes
    X_df["circuit_tire_limitation"] = codes.astype(np.float32)


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
        If True, fit encoders on this split (``compound``, ``circuit_tire_limitation``).
        If False, apply encoders fitted on the training split (use for val/test).

    Returns
    -------
    (X, y) where X is a 2-D float array and y is ``laps_to_cliff``.
    """
    global _label_encoder  # noqa: PLW0603

    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    X_df = df[available].copy()

    _encode_circuit_tire_limitation_column(X_df, fit_encoder=fit_encoder)

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
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    if client.get_experiment_by_name(MLFLOW_EXPERIMENT) is None:
        client.create_experiment(
            MLFLOW_EXPERIMENT,
            artifact_location=_MLARTIFACTS_DIR.resolve().as_uri(),
        )
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run() as run:
        mlflow.log_params(params)

        for prefix, metrics in [
            ("train", train_metrics),
            ("val", val_metrics),
            ("test", test_metrics),
        ]:
            for name, value in metrics.items():
                if isinstance(value, (int, float)) and not math.isfinite(value):
                    continue
                mlflow.log_metric(f"{prefix}_{name}", value)

        if model_type == "sklearn":
            mlflow.sklearn.log_model(model, name="model")
        elif model_type == "xgboost":
            mlflow.xgboost.log_model(model, name="model")
        elif model_type == "lightgbm":
            mlflow.lightgbm.log_model(model, name="model")
        elif model_type == "pytorch":
            mlflow.pytorch.log_model(model, name="model")

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
    rebuild_dataset: bool = False,
    show_training_progress: bool = True,
    allow_adaptive_split: bool = True,
) -> ExperimentResult:
    """End-to-end: load data, split, train, evaluate, log to MLflow.

    Parameters
    ----------
    model_fn:
        Callable that accepts ``(X_train, y_train, params)`` and returns a
        fitted model with a ``.predict(X)`` method. Implementations may also
        accept optional keyword arguments ``eval_set`` and ``show_progress``.
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
    rebuild_dataset:
        If True, rebuild the labeled dataset via Feast before training.
    show_training_progress:
        Passed to *model_fn* as ``show_progress`` when supported (e.g. tqdm bar).
    allow_adaptive_split:
        If True and the canonical split yields no training rows but *df* is non-empty,
        fall back to :func:`split_by_time_adaptive`.

    Returns
    -------
    An :class:`ExperimentResult` containing the run ID, fitted model,
    feature names, split arrays, and evaluation metrics.
    """
    if experiment_name:
        global MLFLOW_EXPERIMENT  # noqa: PLW0603
        MLFLOW_EXPERIMENT = experiment_name

    split_mode = "canonical"
    df = load_training_data(path=data_path, rebuild=rebuild_dataset)
    train_df, val_df, test_df = split_by_time(df)

    if train_df.empty and not df.empty and allow_adaptive_split:
        logger.warning(
            "Canonical split (train 2021–2023, val/test 2024 by round) matched no rows; "
            "using adaptive chronological split. For the paper-style split, ingest 2021–2024."
        )
        train_df, val_df, test_df = split_by_time_adaptive(df)
        split_mode = "adaptive"

    if train_df.empty:
        raise ValueError(
            "Cannot train: temporal split left no training rows.\n"
            + temporal_split_diagnostics(df)
        )
    if val_df.empty:
        if split_mode == "canonical":
            logger.warning(
                "Validation split is empty (no 2024 rounds 1–{} in `session_id`); "
                "training will run without an eval_set.",
                VAL_MAX_ROUND,
            )
        else:
            logger.warning(
                "Validation split is empty under adaptive split; training without eval_set."
            )

    feature_names = get_feature_names(train_df)

    X_train, y_train = prepare_features(train_df, fit_encoder=True)
    X_val, y_val = prepare_features(val_df)
    X_test, y_test = prepare_features(test_df)

    logger.info("Training model with params: {}", params)
    eval_set = (X_val, y_val) if len(val_df) else None
    try:
        model = model_fn(
            X_train,
            y_train,
            params,
            eval_set=eval_set,
            show_progress=show_training_progress,
        )
    except TypeError as exc:
        err = str(exc)
        if "unexpected keyword argument" in err and any(
            kw in err for kw in ("eval_set", "show_progress")
        ):
            model = model_fn(X_train, y_train, params)
        else:
            raise

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

    merged_tags = {**(tags or {}), "temporal_split": split_mode}
    run_id = log_to_mlflow(
        model=model,
        params=split_params,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        model_type=model_type,
        tags=merged_tags,
    )
    return ExperimentResult(
        run_id=run_id,
        model=model,
        feature_names=feature_names,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        test_df=test_df,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )
