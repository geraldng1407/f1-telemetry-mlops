"""CLI for sequential model training (LSTM / TFT / ensemble).

Mirrors ``baseline.py`` but uses windowed sequences and PyTorch training loops.

Usage examples::

    python -m src.training.sequential                       # LSTM defaults
    python -m src.training.sequential --model tft           # TFT
    python -m src.training.sequential --compare             # train both + compare
    python -m src.training.sequential --ensemble            # + ensemble with XGBoost
    python -m src.training.sequential --register --promote  # MLflow registry
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np
from loguru import logger

from src.training.base import (
    ExperimentResult,
    get_feature_names,
    load_training_data,
    log_to_mlflow,
    prepare_features,
    split_by_time,
    split_by_time_adaptive,
    temporal_split_diagnostics,
)
from src.training.metrics import evaluate_all
from src.training.sequence_data import create_dataloaders, create_sequences

# ---------------------------------------------------------------------------
# LSTM path
# ---------------------------------------------------------------------------


def _train_lstm_pipeline(
    train_df: Any,
    val_df: Any,
    test_df: Any,
    params: dict[str, Any],
    tags: dict[str, str],
) -> ExperimentResult:
    """Full LSTM pipeline: build sequences, train, evaluate, log."""
    from src.training.lstm import DEFAULT_LSTM_PARAMS, LSTMPredictor, train_lstm

    p = {**DEFAULT_LSTM_PARAMS, **params}
    seq_len = p["seq_len"]
    batch_size = p["batch_size"]

    dl = create_dataloaders(train_df, val_df, test_df, seq_len=seq_len, batch_size=batch_size)
    n_features = dl["n_features"]
    if n_features == 0:
        raise ValueError("No features found after sequence construction")

    model, history = train_lstm(
        dl["train_loader"], dl["val_loader"], n_features, params=p,
    )
    predictor = LSTMPredictor(model)

    X_test_seq = dl["X_test_seq"]
    y_test = dl["y_test"]
    test_preds = predictor.predict(X_test_seq)

    # Evaluate using the test_df rows aligned via row indices
    test_row_idx = dl["test_row_indices"]
    test_eval_df = test_df.loc[test_row_idx].reset_index(drop=True)
    test_metrics = evaluate_all(y_test, test_preds, test_eval_df)

    # Train / val preds for logging
    X_train_seq, y_train_raw, _ = create_sequences(train_df, seq_len, dl["feature_cols"])
    train_preds = predictor.predict(X_train_seq)
    train_metrics = evaluate_all(y_train_raw, train_preds, train_df)

    val_metrics: dict[str, float] = {}
    if len(val_df):
        X_val_seq, y_val_raw, _ = create_sequences(val_df, seq_len, dl["feature_cols"])
        val_preds = predictor.predict(X_val_seq)
        val_metrics = evaluate_all(y_val_raw, val_preds, val_df)

    log_params = {
        k: v
        for k, v in p.items()
        if isinstance(v, (int, float, str, bool))
    }
    log_params["train_size"] = len(train_df)
    log_params["val_size"] = len(val_df)
    log_params["test_size"] = len(test_df)
    log_params["n_features"] = n_features

    run_id = log_to_mlflow(
        model=model,
        params=log_params,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        model_type="pytorch",
        tags=tags,
    )

    logger.info("LSTM test metrics: {}", test_metrics)

    return ExperimentResult(
        run_id=run_id,
        model=predictor,
        feature_names=dl["feature_cols"],
        X_train=X_train_seq,
        y_train=y_train_raw,
        X_val=np.empty(0),
        y_val=np.empty(0),
        X_test=X_test_seq,
        y_test=y_test,
        test_df=test_eval_df,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )


# ---------------------------------------------------------------------------
# TFT path
# ---------------------------------------------------------------------------


def _train_tft_pipeline(
    train_df: Any,
    val_df: Any,
    test_df: Any,
    params: dict[str, Any],
    tags: dict[str, str],
) -> ExperimentResult:
    """Full TFT pipeline: build TFT datasets, train, evaluate, log."""
    from src.training.tft import (
        DEFAULT_TFT_PARAMS,
        TFTPredictor,
        build_tft_datasets,
        log_tft_attention,
        prepare_tft_dataframe,
        train_tft,
    )

    p = {**DEFAULT_TFT_PARAMS, **params}
    seq_len = p["seq_len"]
    batch_size = p["batch_size"]

    datasets = build_tft_datasets(train_df, val_df, seq_len=seq_len, batch_size=batch_size)
    model, trainer = train_tft(datasets, params=p)

    predictor = TFTPredictor(model, datasets["training"])

    test_preds = predictor.predict(test_df, seq_len=seq_len)
    n_preds = len(test_preds)
    test_sub = test_df.iloc[-n_preds:].reset_index(drop=True)
    y_test = test_sub["laps_to_cliff"].to_numpy(dtype=np.float32)
    test_metrics = evaluate_all(y_test, test_preds, test_sub)

    train_metrics: dict[str, float] = {}
    val_metrics: dict[str, float] = {}

    log_params = {
        k: v for k, v in p.items() if isinstance(v, (int, float, str, bool))
    }
    log_params["train_size"] = len(train_df)
    log_params["val_size"] = len(val_df)
    log_params["test_size"] = len(test_df)

    run_id = log_to_mlflow(
        model=model,
        params=log_params,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        model_type="pytorch",
        tags=tags,
    )

    # Log attention interpretability
    try:
        log_tft_attention(model, datasets["val_loader"], run_id=run_id)
    except Exception:
        logger.warning("Could not extract TFT attention (non-critical)")

    logger.info("TFT test metrics: {}", test_metrics)

    return ExperimentResult(
        run_id=run_id,
        model=predictor,
        feature_names=datasets.get("known_reals", []) + datasets.get("unknown_reals", []),
        X_train=np.empty(0),
        y_train=np.empty(0),
        X_val=np.empty(0),
        y_val=np.empty(0),
        X_test=np.empty(0),
        y_test=y_test,
        test_df=test_sub,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------


def _print_comparison(results: dict[str, dict[str, float]]) -> None:
    """Print a side-by-side metric comparison table."""
    all_metrics = sorted({m for r in results.values() for m in r})
    header = f"{'Metric':<22}" + "".join(f"{name:>14}" for name in results)
    logger.info(header)
    logger.info("-" * len(header))
    for metric in all_metrics:
        row = f"{metric:<22}"
        for name in results:
            val = results[name].get(metric, float("nan"))
            row += f"{val:>14.4f}"
        logger.info(row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train sequential models (LSTM / TFT) for laps_to_cliff.",
    )
    parser.add_argument(
        "--model",
        choices=["lstm", "tft"],
        default="lstm",
        help="Which sequential model to train (default: lstm).",
    )
    parser.add_argument("--seq-len", type=int, default=10, help="Sequence window length.")
    parser.add_argument("--hidden-size", type=int, default=64, help="LSTM hidden size.")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Train both LSTM and TFT, compare on test set.",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Build weighted ensemble with XGBoost baseline.",
    )
    parser.add_argument(
        "--rebuild-dataset",
        action="store_true",
        help="Regenerate labeled dataset before training.",
    )
    parser.add_argument(
        "--no-adaptive-split",
        action="store_true",
        help="Disable fallback to adaptive temporal split.",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Register model in MLflow Model Registry.",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Promote to MLflow Staging (implies --register).",
    )
    args = parser.parse_args(argv)

    if args.promote:
        args.register = True

    try:
        import torch  # noqa: F401
    except ImportError:
        logger.error('PyTorch not installed; run: pip install -e ".[sequential]"')
        return 1

    # ---- Load + split ----
    df = load_training_data(rebuild=args.rebuild_dataset)
    train_df, val_df, test_df = split_by_time(df)

    if train_df.empty and not df.empty and not args.no_adaptive_split:
        logger.warning("Canonical split empty; using adaptive split.")
        train_df, val_df, test_df = split_by_time_adaptive(df)

    if train_df.empty:
        raise ValueError(
            "Cannot train: no training rows.\n" + temporal_split_diagnostics(df)
        )

    # ---- Build params ----
    user_params: dict[str, Any] = {"seq_len": args.seq_len, "batch_size": args.batch_size}
    if args.hidden_size:
        user_params["hidden_size"] = args.hidden_size
    if args.epochs:
        user_params["epochs"] = args.epochs
        user_params["max_epochs"] = args.epochs
    if args.lr:
        user_params["lr"] = args.lr

    results: dict[str, ExperimentResult] = {}

    # ---- Train requested model(s) ----
    models_to_train = ["lstm", "tft"] if args.compare else [args.model]

    for model_name in models_to_train:
        tags = {"model_family": model_name, "script": "sequential"}
        logger.info("=" * 60)
        logger.info("Training {} model", model_name.upper())
        logger.info("=" * 60)

        if model_name == "lstm":
            result = _train_lstm_pipeline(train_df, val_df, test_df, user_params, tags)
        else:
            try:
                import pytorch_forecasting  # noqa: F401
            except ImportError:
                logger.error(
                    "pytorch-forecasting not installed; "
                    'run: pip install -e ".[sequential]"'
                )
                return 1
            result = _train_tft_pipeline(train_df, val_df, test_df, user_params, tags)

        results[model_name] = result

    # ---- Comparison table ----
    if len(results) > 1:
        logger.info("\n=== Model Comparison (test set) ===")
        _print_comparison({k: v.test_metrics for k, v in results.items()})

    # ---- Ensemble ----
    if args.ensemble:
        _run_ensemble(train_df, val_df, test_df, results, user_params)

    # ---- Register / promote best sequential ----
    best_model_name = min(results, key=lambda k: results[k].test_metrics.get("mae", float("inf")))
    best_result = results[best_model_name]

    if args.register:
        from src.training.registry import register_model

        reg_name = f"tire-cliff-{best_model_name}"
        version = register_model(best_result.run_id, model_name=reg_name)
        logger.info("Registered {} version {}", reg_name, version)

        if args.promote:
            from src.training.registry import promote_model

            promote_model(reg_name, version, stage="Staging")
            logger.info("Promoted {} version {} to Staging", reg_name, version)

    return 0


def _run_ensemble(
    train_df: Any,
    val_df: Any,
    test_df: Any,
    seq_results: dict[str, ExperimentResult],
    params: dict[str, Any],
) -> None:
    """Build an XGBoost + sequential ensemble and log results."""
    from src.training.baseline import DEFAULT_XGB_PARAMS, xgb_model_fn
    from src.training.base import run_experiment
    from src.training.ensemble import build_ensemble
    from src.training.sequence_data import create_sequences

    # Train XGBoost for ensemble (or reuse if available)
    logger.info("Training XGBoost for ensemble comparison")
    xgb_result = run_experiment(
        xgb_model_fn,
        dict(DEFAULT_XGB_PARAMS),
        model_type="xgboost",
        tags={"model_family": "xgboost", "script": "sequential-ensemble"},
    )

    # Pick the best sequential model
    best_seq_name = min(
        seq_results,
        key=lambda k: seq_results[k].test_metrics.get("mae", float("inf")),
    )
    best_seq = seq_results[best_seq_name]

    seq_len = params.get("seq_len", 10)

    # Val predictions from XGBoost (flat)
    xgb_val_preds = xgb_result.model.predict(xgb_result.X_val)

    # Val predictions from sequential model
    feature_cols = best_seq.feature_names
    X_val_seq, y_val_seq, _ = create_sequences(val_df, seq_len, feature_cols)
    seq_val_preds = best_seq.model.predict(X_val_seq)

    # Align lengths (val sets may differ slightly)
    min_len = min(len(xgb_val_preds), len(seq_val_preds), len(xgb_result.y_val))
    if min_len == 0:
        logger.warning("Cannot build ensemble: empty validation set")
        return

    ensemble_pred, alpha = build_ensemble(
        xgb_val_preds[:min_len],
        seq_val_preds[:min_len],
        xgb_result.y_val[:min_len],
        xgb_result.model,
        best_seq.model,
    )

    # Test predictions
    xgb_test_preds = xgb_result.model.predict(xgb_result.X_test)
    X_test_seq, y_test_seq, test_idx = create_sequences(test_df, seq_len, feature_cols)
    seq_test_preds = best_seq.model.predict(X_test_seq)

    min_test = min(len(xgb_test_preds), len(seq_test_preds))
    ens_test_preds = alpha * xgb_test_preds[:min_test] + (1 - alpha) * seq_test_preds[:min_test]

    # Align test_df for evaluation
    test_eval = test_df.iloc[:min_test].reset_index(drop=True)
    y_test_ens = test_eval["laps_to_cliff"].to_numpy(dtype=np.float32)
    ens_metrics = evaluate_all(y_test_ens, ens_test_preds, test_eval)

    log_to_mlflow(
        model=ensemble_pred,
        params={"alpha": alpha, "seq_model": best_seq_name, "ensemble": True},
        train_metrics={},
        val_metrics={},
        test_metrics=ens_metrics,
        model_type="sklearn",
        tags={"model_family": "ensemble", "script": "sequential"},
    )

    logger.info("\n=== Ensemble Results ===")
    comparison = {
        "XGBoost": xgb_result.test_metrics,
        best_seq_name.upper(): best_seq.test_metrics,
        f"Ensemble(a={alpha:.2f})": ens_metrics,
    }
    _print_comparison(comparison)


if __name__ == "__main__":
    sys.exit(main())
