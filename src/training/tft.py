"""Temporal Fusion Transformer for tire-cliff prediction.

Uses ``pytorch_forecasting.TemporalFusionTransformer`` which natively handles
static covariates (circuit, compound) alongside temporal inputs, and exposes
interpretable attention weights.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.features.constants import ENGINEERED_FEATURE_COLUMNS
from src.features.target import STINT_GROUP_KEYS
from src.training.sequence_data import DEFAULT_SEQ_LEN, encode_categoricals

# ---------------------------------------------------------------------------
# Default hyper-parameters
# ---------------------------------------------------------------------------

DEFAULT_TFT_PARAMS: dict[str, Any] = {
    "hidden_size": 32,
    "attention_head_size": 1,
    "dropout": 0.1,
    "hidden_continuous_size": 16,
    "lr": 1e-3,
    "max_epochs": 50,
    "batch_size": 64,
    "seq_len": 10,
    "gradient_clip_val": 0.5,
    "patience": 8,
}

# Static categoricals that TFT handles as group-level embeddings.
_STATIC_CATEGORICALS = [
    "compound",
    "circuit_tire_limitation",
    "circuit_street_circuit",
]

# Time-varying known reals (available at prediction time).
_TIME_VARYING_KNOWN = [
    "stint_lap_number",
    "tire_age_laps",
]

# Time-varying unknown reals (only available up to the current time step).
_TIME_VARYING_UNKNOWN = [
    c
    for c in ENGINEERED_FEATURE_COLUMNS
    if c not in _STATIC_CATEGORICALS and c not in _TIME_VARYING_KNOWN
]


# ---------------------------------------------------------------------------
# DataFrame preparation
# ---------------------------------------------------------------------------


def prepare_tft_dataframe(
    df: pd.DataFrame,
    fit_encoder: bool = False,
) -> pd.DataFrame:
    """Prepare a DataFrame for ``TimeSeriesDataSet``.

    Adds a ``time_idx`` column (per-stint sequential index) and a
    ``stint_id`` group column, and encodes categoricals to strings
    suitable for pytorch-forecasting's categorical embeddings.
    """
    df = encode_categoricals(df, fit=fit_encoder)
    df = df.copy()

    df["stint_id"] = (
        df["session_id"].astype(str)
        + "_"
        + df["driver_number"].astype(str)
        + "_"
        + df["stint_number"].astype(str)
    )

    df = df.sort_values(STINT_GROUP_KEYS + ["stint_lap_number"])

    df["time_idx"] = df.groupby("stint_id").cumcount()

    for col in _STATIC_CATEGORICALS:
        if col in df.columns:
            df[col] = df[col].fillna(-1).astype(int).astype(str)

    numeric_cols = _TIME_VARYING_KNOWN + _TIME_VARYING_UNKNOWN + ["laps_to_cliff"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)

    return df


# ---------------------------------------------------------------------------
# Dataset / DataLoader construction
# ---------------------------------------------------------------------------


def build_tft_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    seq_len: int = DEFAULT_SEQ_LEN,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Build ``TimeSeriesDataSet`` objects and DataLoaders for train and val.

    Returns dict with ``training``, ``validation``, ``train_loader``,
    ``val_loader``, and resolved column lists.
    """
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data.encoders import NaNLabelEncoder

    train_prep = prepare_tft_dataframe(train_df, fit_encoder=True)
    val_prep = prepare_tft_dataframe(val_df, fit_encoder=False)

    available_static_cats = [c for c in _STATIC_CATEGORICALS if c in train_prep.columns]
    available_known = [c for c in _TIME_VARYING_KNOWN if c in train_prep.columns]
    available_unknown = [c for c in _TIME_VARYING_UNKNOWN if c in train_prep.columns]

    # Val/test splits can contain stints and compound/circuit combos never seen in train;
    # map unknown categories to the reserved NaN bucket instead of raising KeyError.
    categorical_encoders: dict[str, NaNLabelEncoder] = {
        "stint_id": NaNLabelEncoder(add_nan=True),
    }
    for col in available_static_cats:
        categorical_encoders[col] = NaNLabelEncoder(add_nan=True)

    training = TimeSeriesDataSet(
        train_prep,
        time_idx="time_idx",
        target="laps_to_cliff",
        group_ids=["stint_id"],
        max_encoder_length=seq_len,
        max_prediction_length=1,
        static_categoricals=available_static_cats,
        time_varying_known_reals=available_known,
        time_varying_unknown_reals=available_unknown,
        allow_missing_timesteps=True,
        categorical_encoders=categorical_encoders,
    )

    validation = TimeSeriesDataSet.from_dataset(training, val_prep, stop_randomization=True)

    train_loader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    return {
        "training": training,
        "validation": validation,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "static_categoricals": available_static_cats,
        "known_reals": available_known,
        "unknown_reals": available_unknown,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_tft(
    datasets: dict[str, Any],
    params: dict[str, Any] | None = None,
) -> tuple[Any, Any]:
    """Train a TFT model via PyTorch Lightning.

    Returns ``(model, trainer)``.
    """
    import pytorch_lightning as pl
    from pytorch_forecasting import TemporalFusionTransformer
    from pytorch_forecasting.metrics import QuantileLoss
    from pytorch_lightning.callbacks import EarlyStopping

    p = {**DEFAULT_TFT_PARAMS, **(params or {})}

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=p["patience"],
        mode="min",
        verbose=True,
    )

    trainer = pl.Trainer(
        max_epochs=p["max_epochs"],
        gradient_clip_val=p["gradient_clip_val"],
        callbacks=[early_stop],
        enable_progress_bar=True,
        accelerator="auto",
        log_every_n_steps=10,
    )

    model = TemporalFusionTransformer.from_dataset(
        datasets["training"],
        hidden_size=p["hidden_size"],
        attention_head_size=p["attention_head_size"],
        dropout=p["dropout"],
        hidden_continuous_size=p["hidden_continuous_size"],
        learning_rate=p["lr"],
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    logger.info("TFT model params: {:,}", sum(p.numel() for p in model.parameters()))

    trainer.fit(
        model,
        train_dataloaders=datasets["train_loader"],
        val_dataloaders=datasets["val_loader"],
    )

    best_model_path = trainer.checkpoint_callback.best_model_path  # type: ignore[union-attr]
    if best_model_path:
        model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    return model, trainer


# ---------------------------------------------------------------------------
# Predictor wrapper
# ---------------------------------------------------------------------------


class TFTPredictor:
    """Wraps a trained TFT with a ``.predict()`` returning numpy predictions."""

    def __init__(self, model: Any, training_dataset: Any) -> None:
        self.model = model
        self.training_dataset = training_dataset

    def predict(self, df: pd.DataFrame, seq_len: int = DEFAULT_SEQ_LEN) -> np.ndarray:
        """Generate predictions for every row in *df*.

        Builds a ``TimeSeriesDataSet`` from the trained dataset's parameters,
        runs prediction, and returns a flat numpy array aligned with *df*
        (one prediction per row that has enough history).
        """
        from pytorch_forecasting import TimeSeriesDataSet

        prep = prepare_tft_dataframe(df, fit_encoder=False)
        dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset, prep, stop_randomization=True,
        )
        dl = dataset.to_dataloader(train=False, batch_size=256, num_workers=0)
        raw_preds = self.model.predict(dl)

        preds_np = raw_preds.cpu().numpy() if hasattr(raw_preds, "cpu") else np.asarray(raw_preds)
        if preds_np.ndim > 1:
            preds_np = preds_np.mean(axis=-1) if preds_np.shape[-1] > 1 else preds_np.squeeze(-1)
        return preds_np.ravel()


# ---------------------------------------------------------------------------
# Interpretability
# ---------------------------------------------------------------------------


def log_tft_attention(
    model: Any,
    val_loader: Any,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Extract and optionally log TFT attention / variable importance.

    Returns the interpretation dict from pytorch-forecasting.
    """
    interpretation = model.interpret_output(
        model.predict(val_loader, return_x=True, mode="raw"),
        reduction="sum",
    )

    if run_id is not None:
        try:
            import mlflow

            with tempfile.TemporaryDirectory() as tmpdir:
                for key, val in interpretation.items():
                    if hasattr(val, "cpu"):
                        arr = val.cpu().numpy()
                    else:
                        arr = np.asarray(val)
                    path = Path(tmpdir) / f"tft_{key}.npy"
                    np.save(path, arr)
                    mlflow.log_artifact(str(path), artifact_path="tft_attention")
        except Exception:
            logger.warning("Could not log TFT attention artifacts to MLflow")

    return interpretation
