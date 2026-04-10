"""Weighted ensemble of XGBoost + sequential model predictions.

The two model families operate on different feature representations (flat
tabular vs. windowed sequences), so the ensemble combines their scalar
predictions post-inference rather than at the feature level.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger
from sklearn.metrics import mean_absolute_error


# ---------------------------------------------------------------------------
# Ensemble weight optimisation
# ---------------------------------------------------------------------------


def optimize_ensemble_weight(
    xgb_preds: np.ndarray,
    seq_preds: np.ndarray,
    y_true: np.ndarray,
    steps: int = 11,
) -> float:
    """Grid-search for the best blending weight on validation data.

    The ensemble prediction is ``alpha * xgb + (1 - alpha) * seq``.
    Returns the alpha that minimises MAE.
    """
    best_alpha = 0.5
    best_mae = float("inf")

    for alpha in np.linspace(0.0, 1.0, steps):
        blended = alpha * xgb_preds + (1.0 - alpha) * seq_preds
        mae = mean_absolute_error(y_true, blended)
        if mae < best_mae:
            best_mae = mae
            best_alpha = float(alpha)

    logger.info("Optimal ensemble alpha={:.2f}  (val MAE {:.4f})", best_alpha, best_mae)
    return best_alpha


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


class EnsemblePredictor:
    """Blends predictions from an XGBoost model and a sequential model.

    Both underlying models must expose a ``.predict(X)`` method.  The caller
    is responsible for providing the correct feature format to each model
    (flat array for XGBoost, 3-D array for LSTM).
    """

    def __init__(
        self,
        xgb_model: Any,
        seq_model: Any,
        alpha: float = 0.5,
    ) -> None:
        self.xgb_model = xgb_model
        self.seq_model = seq_model
        self.alpha = alpha

    def predict(
        self,
        X_flat: np.ndarray,
        X_seq: np.ndarray,
    ) -> np.ndarray:
        """Return ``alpha * xgb(X_flat) + (1 - alpha) * seq(X_seq)``."""
        xgb_preds = np.asarray(self.xgb_model.predict(X_flat), dtype=np.float64)
        seq_preds = np.asarray(self.seq_model.predict(X_seq), dtype=np.float64)
        return self.alpha * xgb_preds + (1.0 - self.alpha) * seq_preds


# ---------------------------------------------------------------------------
# High-level builder
# ---------------------------------------------------------------------------


def build_ensemble(
    xgb_val_preds: np.ndarray,
    seq_val_preds: np.ndarray,
    y_val: np.ndarray,
    xgb_model: Any,
    seq_model: Any,
) -> tuple[EnsemblePredictor, float]:
    """Find optimal weight on validation set and return an ``EnsemblePredictor``.

    Returns ``(predictor, alpha)``.
    """
    alpha = optimize_ensemble_weight(xgb_val_preds, seq_val_preds, y_val)
    predictor = EnsemblePredictor(xgb_model, seq_model, alpha=alpha)
    return predictor, alpha
