"""Baseline XGBoost training for tire-cliff prediction.

Uses ``run_experiment`` (Feast-backed dataset when rebuilt, temporal split,
MLflow logging). Requires ``pip install -e ".[training]"`` and a running
By default uses SQLite at ``./mlflow.db`` and artifacts under ``./mlartifacts``
(no server). Set ``MLFLOW_TRACKING_URI`` (e.g. ``http://localhost:5000``) for Docker MLflow.

Extended flags:
  --tune          Run Optuna hyperparameter search before final training
  --n-trials N    Number of Optuna trials (default 50)
  --shap          Compute and log SHAP explanations after training
  --promote       Promote registered model to MLflow Staging
"""

from __future__ import annotations

import argparse
import inspect
import sys
from typing import Any

from loguru import logger

from src.training.base import ExperimentResult, run_experiment

DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
}

_SPLIT_META_KEYS = frozenset({"train_size", "val_size", "test_size"})


def xgb_model_fn(
    X_train: Any,
    y_train: Any,
    params: dict[str, Any],
    eval_set: tuple[Any, Any] | None = None,
    show_progress: bool = True,
) -> Any:
    import xgboost as xgb

    model_kwargs = {k: v for k, v in params.items() if k not in _SPLIT_META_KEYS}
    model = xgb.XGBRegressor(**model_kwargs)

    fit_kwargs: dict[str, Any] = {"verbose": False}
    if eval_set is not None:
        X_val, y_val = eval_set
        fit_kwargs["eval_set"] = [(X_val, y_val)]

    supports_fit_callbacks = "callbacks" in inspect.signature(model.fit).parameters

    tqdm_cb = None
    if show_progress:
        try:
            from tqdm.auto import tqdm
        except ImportError:
            tqdm = None

        if tqdm is not None and supports_fit_callbacks:
            n_est = int(model_kwargs.get("n_estimators", 100))

            class _TqdmTrainCallback(xgb.callback.TrainingCallback):
                def __init__(self) -> None:
                    self._pbar = tqdm(
                        total=n_est,
                        desc="XGBoost",
                        unit="tree",
                        leave=True,
                    )

                def after_iteration(self, model: Any, epoch: int, evals_log: dict) -> bool:
                    self._pbar.update(1)
                    if evals_log:
                        parts = []
                        for name, metrics in evals_log.items():
                            for mname, history in metrics.items():
                                if history:
                                    parts.append(f"{name}:{mname}={history[-1]:.4f}")
                        if parts:
                            self._pbar.set_postfix_str(", ".join(parts)[:80])
                    return False

                def close(self) -> None:
                    self._pbar.close()

            tqdm_cb = _TqdmTrainCallback()
            fit_kwargs.setdefault("callbacks", []).append(tqdm_cb)
        elif fit_kwargs.get("eval_set"):
            fit_kwargs["verbose"] = True
        else:
            logger.info(
                "Training XGBoost with n_estimators={} (no val split: no per-round progress)",
                model_kwargs.get("n_estimators", "?"),
            )

    try:
        model.fit(X_train, y_train, **fit_kwargs)
    finally:
        if tqdm_cb is not None:
            tqdm_cb.close()

    return model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train baseline XGBoost model for laps_to_cliff (MLflow experiment logging).",
    )
    parser.add_argument(
        "--rebuild-dataset",
        action="store_true",
        help="Regenerate labeled dataset via Feast get_historical_features() before training.",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="MLflow experiment name (default: tire-cliff-prediction).",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Register the logged model in the MLflow Model Registry after the run.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Disable tqdm progress bar during boosting.",
    )
    parser.add_argument(
        "--no-adaptive-split",
        action="store_true",
        help="Do not fall back to chronological split when 2021-2024 canonical split has no train rows.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run Optuna hyperparameter search before the final training run.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials (default: 50). Only used with --tune.",
    )
    parser.add_argument(
        "--shap",
        action="store_true",
        help="Compute SHAP explanations after training and log as MLflow artifacts.",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Promote the registered model to MLflow Staging (implies --register).",
    )
    args = parser.parse_args(argv)

    if args.promote:
        args.register = True

    try:
        import xgboost  # noqa: F401
    except ImportError:
        logger.error('Missing optional deps; run: pip install -e ".[training]"')
        return 1

    # --- 1. Optuna tuning (optional) ---
    params = dict(DEFAULT_XGB_PARAMS)
    if args.tune:
        from src.training.tuning import run_tuning_study

        logger.info("Starting Optuna tuning with {} trials", args.n_trials)
        params = run_tuning_study(
            n_trials=args.n_trials,
            experiment_name=args.experiment,
            rebuild_dataset=args.rebuild_dataset,
            allow_adaptive_split=not args.no_adaptive_split,
        )
        logger.info("Tuning complete; best params: {}", params)

    # --- 2. Final training run ---
    tags = {"model_family": "xgboost", "script": "baseline"}
    if args.tune:
        tags["tuning"] = "optuna"
        tags["n_trials"] = str(args.n_trials)

    result: ExperimentResult = run_experiment(
        xgb_model_fn,
        params,
        experiment_name=args.experiment,
        model_type="xgboost",
        tags=tags,
        rebuild_dataset=args.rebuild_dataset,
        show_training_progress=not args.quiet,
        allow_adaptive_split=not args.no_adaptive_split,
    )
    logger.info("MLflow run_id={}", result.run_id)

    # --- 3. Feature importance (always — cheap) ---
    from src.training.explainability import compute_feature_importance

    compute_feature_importance(
        result.model,
        result.feature_names,
        run_id=result.run_id,
    )

    # --- 4. SHAP explanations (optional) ---
    if args.shap:
        from src.training.explainability import compute_shap_explanations

        logger.info("Computing SHAP explanations on test set ({} samples)", len(result.X_test))
        compute_shap_explanations(
            result.model,
            result.X_test,
            result.feature_names,
            run_id=result.run_id,
        )

    # --- 5. Register & promote ---
    if args.register:
        from src.training.registry import register_model

        version = register_model(result.run_id)
        logger.info("Registered model version {}", version)

        if args.promote:
            from src.training.registry import promote_model

            promote_model("tire-cliff-xgboost", version, stage="Staging")
            logger.info("Promoted model version {} to Staging", version)

    return 0


if __name__ == "__main__":
    sys.exit(main())
