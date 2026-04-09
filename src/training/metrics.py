"""Evaluation metrics for tire-cliff prediction.

Three metrics aligned with the project README:
  1. MAE of predicted ``laps_to_cliff`` vs actual
  2. Precision@K — fraction of predictions within K laps of actual
  3. Strategy Accuracy — would the model's pit recommendation beat the
     team's actual strategy?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.features.target import STINT_GROUP_KEYS


# ---------------------------------------------------------------------------
# 1. MAE
# ---------------------------------------------------------------------------


def mae_laps_to_cliff(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    is_censored: np.ndarray | None = None,
) -> float:
    """Mean absolute error, optionally excluding censored stints.

    Parameters
    ----------
    y_true, y_pred:
        Ground-truth and predicted ``laps_to_cliff``.
    is_censored:
        Binary array; if provided, only uncensored rows (``== 0``) are scored.
    """
    if len(y_true) == 0:
        return float("nan")
    mask = _uncensored_mask(is_censored, len(y_true))
    if not np.any(mask):
        return float("nan")
    return float(mean_absolute_error(y_true[mask], y_pred[mask]))


# ---------------------------------------------------------------------------
# 2. Precision@K
# ---------------------------------------------------------------------------


def precision_at_k(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int = 3,
    is_censored: np.ndarray | None = None,
) -> float:
    """Fraction of predictions within *k* laps of the actual cliff.

    Answers: "Is the predicted cliff lap within *k* laps of the actual?"
    Only scored on uncensored stints when *is_censored* is supplied.
    """
    mask = _uncensored_mask(is_censored, len(y_true))
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return 0.0
    return float(np.mean(np.abs(yt - yp) <= k))


# ---------------------------------------------------------------------------
# 3. Strategy Accuracy
# ---------------------------------------------------------------------------


def strategy_accuracy(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    early_margin: int = 2,
) -> float:
    """Fraction of stints where the model's pit recommendation is optimal.

    For each stint the "recommended pit lap" is the first observation where
    the model predicts the cliff is imminent: ``current_lap + predicted_laps_to_cliff``.
    A recommendation is considered correct when it falls in the window
    ``[cliff_lap - early_margin, cliff_lap]`` — close enough to avoid degraded
    laps without pitting unnecessarily early.

    Only evaluated on uncensored stints (those with a known cliff).
    """
    work = df[STINT_GROUP_KEYS + ["stint_lap_number", "cliff_lap", "is_censored"]].copy()
    work["predicted_laps_to_cliff"] = y_pred
    work["recommended_pit_lap"] = work["stint_lap_number"] + work["predicted_laps_to_cliff"]

    uncensored = work[work["is_censored"] == 0]
    if uncensored.empty:
        return 0.0

    # Per-stint: take the minimum recommended pit lap (earliest signal)
    stint_recs = (
        uncensored.groupby(STINT_GROUP_KEYS)
        .agg(
            rec_pit=("recommended_pit_lap", "min"),
            cliff=("cliff_lap", "first"),
        )
    )

    lower = stint_recs["cliff"] - early_margin
    upper = stint_recs["cliff"]
    correct = (stint_recs["rec_pit"] >= lower) & (stint_recs["rec_pit"] <= upper)
    return float(correct.mean())


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def evaluate_all(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    df: pd.DataFrame,
    k: int = 3,
) -> dict[str, float]:
    """Compute all three metrics and return as a flat dict."""
    is_censored = df["is_censored"].to_numpy() if "is_censored" in df.columns else None

    return {
        "mae": mae_laps_to_cliff(y_true, y_pred, is_censored),
        "precision_at_3": precision_at_k(y_true, y_pred, k=k, is_censored=is_censored),
        "strategy_accuracy": strategy_accuracy(df, y_pred),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uncensored_mask(is_censored: np.ndarray | None, n: int) -> np.ndarray:
    """Return a boolean mask selecting uncensored observations."""
    if is_censored is None:
        return np.ones(n, dtype=bool)
    return np.asarray(is_censored) == 0
