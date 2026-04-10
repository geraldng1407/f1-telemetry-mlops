"""Optuna hyperparameter tuning for XGBoost tire-cliff prediction.

Wraps the existing training harness in an Optuna objective, searching over
n_estimators, max_depth, learning_rate, min_child_weight, subsample, and
colsample_bytree.  Each trial is logged as a nested MLflow run.
"""

from __future__ import annotations

from typing import Any

import mlflow
import numpy as np
import optuna
from loguru import logger
from sklearn.metrics import mean_absolute_error

from src.training.base import (
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    get_feature_names,
    load_training_data,
    prepare_features,
    split_by_time,
    split_by_time_adaptive,
    temporal_split_diagnostics,
)
from src.training.baseline import xgb_model_fn

# Fixed params that are not tuned
_FIXED_PARAMS: dict[str, Any] = {
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
}


def _build_params(trial: optuna.Trial) -> dict[str, Any]:
    """Sample hyperparameters for a single Optuna trial."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        **_FIXED_PARAMS,
    }


def run_tuning_study(
    n_trials: int = 50,
    experiment_name: str | None = None,
    rebuild_dataset: bool = False,
    allow_adaptive_split: bool = True,
) -> dict[str, Any]:
    """Run an Optuna study optimising val MAE and return the best params.

    Each trial trains XGBoost, evaluates on the validation split, and logs
    params + val_mae as a nested MLflow run under a parent "tuning" run.

    Returns
    -------
    A merged dict of the best sampled hyperparameters and ``_FIXED_PARAMS``,
    ready to pass to :func:`run_experiment`.
    """
    df = load_training_data(rebuild=rebuild_dataset)
    train_df, val_df, test_df = split_by_time(df)

    if train_df.empty and not df.empty and allow_adaptive_split:
        logger.warning("Canonical split empty; falling back to adaptive split for tuning.")
        train_df, val_df, test_df = split_by_time_adaptive(df)

    if train_df.empty:
        raise ValueError(
            "Cannot tune: temporal split left no training rows.\n"
            + temporal_split_diagnostics(df)
        )
    if val_df.empty:
        raise ValueError(
            "Cannot tune: validation split is empty (need val set to optimise against).\n"
            + temporal_split_diagnostics(df)
        )

    X_train, y_train = prepare_features(train_df, fit_encoder=True)
    X_val, y_val = prepare_features(val_df)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name or MLFLOW_EXPERIMENT)

    def objective(trial: optuna.Trial) -> float:
        params = _build_params(trial)

        with mlflow.start_run(nested=True, run_name=f"optuna-trial-{trial.number}"):
            mlflow.log_params(params)
            mlflow.set_tag("optuna_trial", trial.number)

            model = xgb_model_fn(
                X_train,
                y_train,
                params,
                eval_set=(X_val, y_val),
                show_progress=False,
            )
            preds = model.predict(X_val)
            val_mae = float(mean_absolute_error(y_val, preds))

            mlflow.log_metric("val_mae", val_mae)

        return val_mae

    study = optuna.create_study(direction="minimize", study_name="xgb-tuning")

    with mlflow.start_run(run_name="optuna-tuning-parent"):
        mlflow.set_tag("tuning", "optuna")
        mlflow.log_param("n_trials", n_trials)

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        mlflow.log_metric("best_val_mae", study.best_value)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})

    best_params: dict[str, Any] = {**study.best_params, **_FIXED_PARAMS}

    logger.info("Optuna best val MAE: {:.4f}", study.best_value)
    logger.info("Best params: {}", best_params)
    return best_params
