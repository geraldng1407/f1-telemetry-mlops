"""Sequence data preparation for LSTM / TFT tire-cliff prediction.

Converts the flat labeled dataset into sliding-window sequences grouped by
stint, suitable for PyTorch sequential models.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.features.constants import ENGINEERED_FEATURE_COLUMNS
from src.features.target import STINT_GROUP_KEYS

DEFAULT_SEQ_LEN = 10

# Columns that are strings in raw data and need numeric encoding for the model.
_CATEGORICAL_ENCODE_COLS = ("circuit_tire_limitation", "compound")


# ---------------------------------------------------------------------------
# Categorical encoding (shared with base.py's logic, kept standalone so
# sequence_data doesn't depend on module-level state in base)
# ---------------------------------------------------------------------------

_fitted_categories: dict[str, list[str]] = {}


def encode_categoricals(
    df: pd.DataFrame,
    fit: bool = False,
) -> pd.DataFrame:
    """Encode string categoricals to integer codes (in-place on a copy).

    When *fit* is True the category mapping is stored; subsequent calls with
    ``fit=False`` reuse it.
    """
    df = df.copy()
    for col in _CATEGORICAL_ENCODE_COLS:
        if col not in df.columns:
            continue
        s = df[col].fillna("__nan__").astype(str)
        if fit:
            _fitted_categories[col] = sorted(s.unique())
        cats = _fitted_categories.get(col)
        if not cats:
            df[col] = np.float32(-1.0)
        else:
            codes = pd.Categorical(s, categories=cats, ordered=False).codes
            df[col] = codes.astype(np.float32)
    return df


# ---------------------------------------------------------------------------
# Feature column resolution
# ---------------------------------------------------------------------------


def get_sequence_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return the ordered list of numeric feature columns available in *df*."""
    cols = [c for c in ENGINEERED_FEATURE_COLUMNS if c in df.columns]
    if "compound" in df.columns and "compound" not in cols:
        cols.append("compound")
    return cols


# ---------------------------------------------------------------------------
# Sliding-window sequence builder
# ---------------------------------------------------------------------------


def create_sequences(
    df: pd.DataFrame,
    seq_len: int = DEFAULT_SEQ_LEN,
    feature_cols: list[str] | None = None,
    fit_encoder: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sliding-window sequences from a flat labeled DataFrame.

    Parameters
    ----------
    df:
        Labeled DataFrame with stint-level data (must contain
        ``stint_lap_number``, ``laps_to_cliff``, and the stint group keys).
    seq_len:
        Number of past laps (including the current one) in each window.
    feature_cols:
        Explicit feature column list.  Defaults to all available engineered
        feature columns.
    fit_encoder:
        If True, fit categorical encoders on this data (use for train split).

    Returns
    -------
    X_seq : np.ndarray, shape ``(n_samples, seq_len, n_features)``
    y : np.ndarray, shape ``(n_samples,)``
    row_indices : np.ndarray, shape ``(n_samples,)``
        Index into the original *df* for each sample (for aligning predictions
        back with the DataFrame for metric evaluation).
    """
    df_enc = encode_categoricals(df, fit=fit_encoder)

    if feature_cols is None:
        feature_cols = get_sequence_feature_cols(df_enc)

    n_features = len(feature_cols)
    sequences: list[np.ndarray] = []
    targets: list[float] = []
    indices: list[int] = []

    for _keys, stint_df in df_enc.groupby(STINT_GROUP_KEYS, sort=False):
        stint_df = stint_df.sort_values("stint_lap_number")
        feat_matrix = stint_df[feature_cols].to_numpy(dtype=np.float32, na_value=0.0)
        y_vals = stint_df["laps_to_cliff"].to_numpy(dtype=np.float32)
        idx_vals = stint_df.index.to_numpy()

        n_laps = len(stint_df)
        for i in range(n_laps):
            start = i - seq_len + 1
            if start < 0:
                pad_len = -start
                window = np.zeros((seq_len, n_features), dtype=np.float32)
                window[pad_len:] = feat_matrix[: i + 1]
            else:
                window = feat_matrix[start : i + 1]
            sequences.append(window)
            targets.append(y_vals[i])
            indices.append(idx_vals[i])

    if not sequences:
        return (
            np.empty((0, seq_len, n_features), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    X_seq = np.stack(sequences, axis=0)
    y = np.array(targets, dtype=np.float32)
    row_indices = np.array(indices, dtype=np.int64)

    logger.info(
        "Created {} sequences (seq_len={}, n_features={}) from {} stints",
        len(y),
        seq_len,
        n_features,
        df_enc.groupby(STINT_GROUP_KEYS, sort=False).ngroups,
    )
    return X_seq, y, row_indices


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

try:
    import torch
    from torch.utils.data import DataLoader, Dataset, TensorDataset  # noqa: F401

    class StintSequenceDataset(Dataset):  # type: ignore[type-arg]
        """PyTorch Dataset wrapping pre-built sequence arrays."""

        def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
            self.X = torch.as_tensor(X, dtype=torch.float32)
            self.y = torch.as_tensor(y, dtype=torch.float32)

        def __len__(self) -> int:
            return len(self.y)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            return self.X[idx], self.y[idx]

    def create_dataloaders(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        seq_len: int = DEFAULT_SEQ_LEN,
        batch_size: int = 64,
        feature_cols: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build DataLoaders and metadata from the three temporal splits.

        Returns a dict with keys: ``train_loader``, ``val_loader``,
        ``test_loader``, ``test_row_indices``, ``n_features``,
        ``feature_cols``, ``X_test_seq``, ``y_test``.
        """
        X_train, y_train, _ = create_sequences(
            train_df, seq_len, feature_cols, fit_encoder=True,
        )
        resolved_cols = feature_cols or get_sequence_feature_cols(
            encode_categoricals(train_df, fit=False),
        )
        X_val, y_val, _ = create_sequences(val_df, seq_len, resolved_cols)
        X_test, y_test, test_idx = create_sequences(test_df, seq_len, resolved_cols)

        train_loader = DataLoader(
            StintSequenceDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            StintSequenceDataset(X_val, y_val),
            batch_size=batch_size,
            shuffle=False,
        )
        test_loader = DataLoader(
            StintSequenceDataset(X_test, y_test),
            batch_size=batch_size,
            shuffle=False,
        )

        return {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
            "test_row_indices": test_idx,
            "n_features": X_train.shape[2] if X_train.ndim == 3 else 0,
            "feature_cols": resolved_cols,
            "X_test_seq": X_test,
            "y_test": y_test,
        }

except ImportError:
    pass
