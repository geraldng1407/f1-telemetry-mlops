"""MLflow model loading, warm-up, and inference with confidence intervals."""

from __future__ import annotations

from typing import Any

import mlflow
import numpy as np
import pandas as pd
import structlog

from src.features.constants import ENGINEERED_FEATURE_COLUMNS
from src.inference.config import Settings
from src.inference.models import PredictionResult

logger = structlog.get_logger(__name__)

# Known compound categories matching the LabelEncoder fitted during training.
# Order matters: LabelEncoder sorts classes alphabetically.
_COMPOUND_CATEGORIES = ["HARD", "INTERMEDIATE", "MEDIUM", "SOFT", "UNKNOWN", "WET"]

# Known circuit_tire_limitation categories (alphabetically sorted).
_TIRE_LIMITATION_CATEGORIES = ["__nan__", "front", "rear"]


def _encode_compound(value: str | None) -> float:
    v = (value or "UNKNOWN").upper()
    try:
        return float(_COMPOUND_CATEGORIES.index(v))
    except ValueError:
        return float(_COMPOUND_CATEGORIES.index("UNKNOWN"))


def _encode_tire_limitation(value: Any) -> float:
    s = str(value) if value is not None and not _is_nan(value) else "__nan__"
    try:
        return float(_TIRE_LIMITATION_CATEGORIES.index(s))
    except ValueError:
        return -1.0


def _is_nan(v: Any) -> bool:
    try:
        return v != v  # NaN != NaN
    except (TypeError, ValueError):
        return False


class ModelRunner:
    """Wraps an MLflow PyFunc model for real-time inference."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Any = None

    def load(self) -> None:
        mlflow.set_tracking_uri(self._settings.mlflow_tracking_uri)
        model_uri = f"models:/{self._settings.model_name}@{self._settings.model_alias}"
        logger.info("loading_model", uri=model_uri)
        self._model = mlflow.pyfunc.load_model(model_uri)
        logger.info("model_loaded", uri=model_uri)

    def warm_up(self) -> None:
        """Run a dummy prediction to trigger any lazy initialisation."""
        n_features = len(ENGINEERED_FEATURE_COLUMNS) + 1  # +1 for compound
        dummy = pd.DataFrame(
            np.zeros((1, n_features), dtype=np.float32),
            columns=list(ENGINEERED_FEATURE_COLUMNS) + ["compound"],
        )
        self._model.predict(dummy)
        logger.info("model_warmed_up")

    def predict(
        self,
        features: dict[str, Any],
        driver_number: str,
        lap_number: int,
        stint_number: int,
    ) -> PredictionResult:
        """Assemble the feature vector, run inference, and return a result with CI."""
        row: dict[str, float] = {}
        for col in ENGINEERED_FEATURE_COLUMNS:
            val = features.get(col, float("nan"))
            if col == "circuit_tire_limitation":
                row[col] = _encode_tire_limitation(val)
            else:
                row[col] = float(val) if val is not None else float("nan")

        row["compound"] = _encode_compound(features.get("compound"))

        df = pd.DataFrame([row])
        pred = float(self._model.predict(df)[0])

        # Confidence interval: symmetric interval using configured MAE * z-score
        half_width = self._settings.residual_mae * self._settings.confidence_z
        lower = max(pred - half_width, 0.0)
        upper = pred + half_width

        return PredictionResult(
            driver_number=driver_number,
            lap_number=lap_number,
            stint_number=stint_number,
            compound=features.get("compound"),
            tire_age=(
                int(features["tire_age_laps"])
                if not _is_nan(features.get("tire_age_laps"))
                else None
            ),
            predicted_laps_to_cliff=round(pred, 2),
            confidence_lower=round(lower, 2),
            confidence_upper=round(upper, 2),
        )
