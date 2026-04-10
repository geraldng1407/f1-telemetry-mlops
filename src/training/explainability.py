"""Feature importance and SHAP explanations for tire-cliff models.

Provides two main functions:

* :func:`compute_feature_importance` -- native XGBoost feature importance
  (gain / weight / cover), saved as CSV + bar chart PNG.
* :func:`compute_shap_explanations` -- TreeExplainer SHAP values, saved as
  beeswarm summary plot, per-sample waterfall plots, and a Parquet artifact.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from loguru import logger

from src.training.base import MLFLOW_TRACKING_URI

# ---------------------------------------------------------------------------
# Feature importance (XGBoost native)
# ---------------------------------------------------------------------------


def compute_feature_importance(
    model: Any,
    feature_names: list[str],
    run_id: str | None = None,
) -> pd.DataFrame:
    """Extract and log XGBoost native feature importance.

    Returns a DataFrame with columns ``feature``, ``gain``, ``weight``,
    ``cover``, sorted by gain descending.  When *run_id* is provided the
    CSV and a bar-chart PNG are logged as MLflow artifacts.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    booster = model.get_booster()

    rows = []
    for importance_type in ("gain", "weight", "cover"):
        scores = booster.get_score(importance_type=importance_type)
        for fname, score in scores.items():
            rows.append({"feature": fname, "type": importance_type, "score": score})

    raw_df = pd.DataFrame(rows)
    if raw_df.empty:
        logger.warning("No feature importance scores returned by the booster.")
        return pd.DataFrame(columns=["feature", "gain", "weight", "cover"])

    pivot = raw_df.pivot(index="feature", columns="type", values="score").fillna(0.0)
    for col in ("gain", "weight", "cover"):
        if col not in pivot.columns:
            pivot[col] = 0.0

    # XGBoost uses internal names f0, f1, ... — map back to readable names
    fname_map = {f"f{i}": name for i, name in enumerate(feature_names)}
    pivot.index = pivot.index.map(lambda x: fname_map.get(x, x))
    pivot = pivot.sort_values("gain", ascending=False).reset_index()
    pivot.columns.name = None

    logger.info("Top-5 features by gain:\n{}", pivot.head(5).to_string(index=False))

    if run_id:
        _log_importance_artifacts(pivot, run_id)

    return pivot


def _log_importance_artifacts(importance_df: pd.DataFrame, run_id: str) -> None:
    """Save CSV + bar chart to MLflow."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "feature_importance.csv"
        importance_df.to_csv(csv_path, index=False)

        top_n = importance_df.head(15)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top_n["feature"][::-1], top_n["gain"][::-1])
        ax.set_xlabel("Gain")
        ax.set_title("Feature Importance (Top 15 by Gain)")
        fig.tight_layout()
        chart_path = Path(tmpdir) / "feature_importance.png"
        fig.savefig(chart_path, dpi=150)
        plt.close(fig)

        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(str(csv_path), artifact_path="explainability")
            mlflow.log_artifact(str(chart_path), artifact_path="explainability")

    logger.info("Logged feature importance artifacts to run {}", run_id)


# ---------------------------------------------------------------------------
# SHAP explanations
# ---------------------------------------------------------------------------


def compute_shap_explanations(
    model: Any,
    X: np.ndarray,
    feature_names: list[str],
    run_id: str | None = None,
    max_display: int = 20,
    sample_indices: list[int] | None = None,
) -> np.ndarray:
    """Compute SHAP values and generate explanation plots.

    Parameters
    ----------
    model:
        A fitted XGBoost model (sklearn API).
    X:
        Feature matrix to explain (typically test split).
    feature_names:
        Column names corresponding to X's columns.
    run_id:
        If provided, log plots and SHAP values to this MLflow run.
    max_display:
        Number of features to show in the summary plot.
    sample_indices:
        Row indices into *X* for which to generate waterfall plots.
        Defaults to first 3 rows.

    Returns
    -------
    SHAP values array with shape ``(n_samples, n_features)``.
    """
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    logger.info(
        "Computed SHAP values for {} samples x {} features",
        shap_values.shape[0],
        shap_values.shape[1],
    )

    if run_id:
        _log_shap_artifacts(
            explainer,
            shap_values,
            X,
            feature_names,
            run_id,
            max_display=max_display,
            sample_indices=sample_indices,
        )

    return shap_values


def _log_shap_artifacts(
    explainer: Any,
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    run_id: str,
    max_display: int = 20,
    sample_indices: list[int] | None = None,
) -> None:
    """Generate and log SHAP plots + values Parquet to MLflow."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    if sample_indices is None:
        sample_indices = list(range(min(3, len(X))))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Beeswarm summary plot
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=feature_names,
            max_display=max_display,
            show=False,
        )
        summary_path = tmpdir_path / "shap_summary.png"
        plt.savefig(summary_path, dpi=150, bbox_inches="tight")
        plt.close("all")

        # Per-sample waterfall plots
        explanation = shap.Explanation(
            values=shap_values,
            base_values=np.full(len(X), explainer.expected_value),
            data=X,
            feature_names=feature_names,
        )

        for idx in sample_indices:
            if idx >= len(X):
                continue
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(explanation[idx], max_display=max_display, show=False)
            waterfall_path = tmpdir_path / f"shap_waterfall_sample_{idx}.png"
            plt.savefig(waterfall_path, dpi=150, bbox_inches="tight")
            plt.close("all")

        # SHAP values as Parquet
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        parquet_path = tmpdir_path / "shap_values.parquet"
        shap_df.to_parquet(parquet_path, engine="pyarrow")

        with mlflow.start_run(run_id=run_id):
            for artifact_file in tmpdir_path.iterdir():
                mlflow.log_artifact(str(artifact_file), artifact_path="explainability")

    logger.info(
        "Logged SHAP artifacts ({} waterfall plots) to run {}",
        len(sample_indices),
        run_id,
    )
