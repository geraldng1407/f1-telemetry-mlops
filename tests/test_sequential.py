"""Tests for sequential models: sequence data, LSTM, ensemble, and TFT."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stint_df(n_stints: int = 3, laps_per_stint: int = 12) -> pd.DataFrame:
    """Create a synthetic stint DataFrame with minimal columns."""
    rows: list[dict] = []
    rng = np.random.default_rng(42)
    for s in range(n_stints):
        for lap in range(1, laps_per_stint + 1):
            rows.append(
                {
                    "session_id": f"2022_{s + 1}_Race",
                    "driver_number": 55,
                    "stint_number": 1,
                    "stint_lap_number": float(lap),
                    "tire_age_laps": float(lap),
                    "rolling_mean_laptime_5": 90.0 + rng.normal(0, 0.5),
                    "rolling_var_laptime_5": rng.exponential(0.1),
                    "fuel_corrected_laptime": 90.0 + rng.normal(0, 0.3),
                    "track_temp_c": 35.0 + rng.normal(0, 2),
                    "laps_to_cliff": float(max(1, laps_per_stint - lap)),
                    "is_censored": 0,
                    "cliff_lap": float(laps_per_stint),
                    "laps_remaining_in_stint": float(laps_per_stint - lap),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sequence data tests
# ---------------------------------------------------------------------------


class TestCreateSequences:
    def test_output_shape(self):
        from src.training.sequence_data import create_sequences

        df = _make_stint_df(n_stints=2, laps_per_stint=8)
        seq_len = 5
        X, y, idx = create_sequences(df, seq_len=seq_len, fit_encoder=True)

        assert X.ndim == 3
        assert X.shape[0] == len(df)  # one sample per lap
        assert X.shape[1] == seq_len
        assert y.shape == (len(df),)
        assert idx.shape == (len(df),)

    def test_padding_early_laps(self):
        from src.training.sequence_data import create_sequences

        df = _make_stint_df(n_stints=1, laps_per_stint=6)
        seq_len = 4
        X, y, _ = create_sequences(df, seq_len=seq_len, fit_encoder=True)

        # First lap should have 3 zero-padded rows + 1 real row
        assert np.allclose(X[0, :3, :], 0.0)
        assert not np.allclose(X[0, 3, :], 0.0)

        # Second lap: 2 padded + 2 real
        assert np.allclose(X[1, :2, :], 0.0)
        assert not np.allclose(X[1, 2, :], 0.0)

    def test_row_indices_map_back(self):
        from src.training.sequence_data import create_sequences

        df = _make_stint_df(n_stints=2, laps_per_stint=5)
        _, _, idx = create_sequences(df, seq_len=3, fit_encoder=True)

        # Every index should be valid
        assert all(i in df.index for i in idx)

    def test_empty_df(self):
        from src.training.sequence_data import create_sequences

        df = pd.DataFrame(columns=["session_id", "driver_number", "stint_number",
                                    "stint_lap_number", "laps_to_cliff"])
        X, y, idx = create_sequences(df, seq_len=5, fit_encoder=True)
        assert X.shape[0] == 0
        assert X.shape[1] == 5
        assert y.shape == (0,)


# ---------------------------------------------------------------------------
# LSTM tests
# ---------------------------------------------------------------------------


class TestLSTM:
    @pytest.fixture(autouse=True)
    def _skip_no_torch(self):
        pytest.importorskip("torch")

    def test_model_forward_shape(self):
        import torch
        from src.training.lstm import LSTMCliffModel

        model = LSTMCliffModel(n_features=6, hidden_size=16, num_layers=2, dropout=0.0)
        x = torch.randn(4, 5, 6)
        out = model(x)
        assert out.shape == (4,)

    def test_training_loop_loss_decreases(self):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from src.training.lstm import train_lstm

        rng = np.random.default_rng(42)
        n, seq_len, n_feat = 200, 5, 4
        X = rng.standard_normal((n, seq_len, n_feat)).astype(np.float32)
        # Deterministic target from the last timestep — easy for the LSTM to fit.
        y = X[:, -1, :].sum(axis=1).astype(np.float32)

        ds = TensorDataset(torch.as_tensor(X), torch.as_tensor(y))
        loader = DataLoader(ds, batch_size=32, shuffle=True)

        model, history = train_lstm(
            loader, loader, n_features=n_feat,
            params={
                "epochs": 25,
                "patience": 100,
                "hidden_size": 32,
                "num_layers": 1,
                "dropout": 0.0,
                "lr": 0.05,
            },
        )
        assert len(history.train_loss) >= 2
        assert history.train_loss[-1] < history.train_loss[0]

    def test_predictor_predict_shape(self):
        from src.training.lstm import LSTMCliffModel, LSTMPredictor

        model = LSTMCliffModel(n_features=4, hidden_size=8, num_layers=1, dropout=0.0)
        predictor = LSTMPredictor(model)
        X = np.random.default_rng(0).random((10, 5, 4)).astype(np.float32)
        preds = predictor.predict(X)
        assert preds.shape == (10,)
        assert preds.dtype == np.float32 or preds.dtype == np.float64


# ---------------------------------------------------------------------------
# Ensemble tests
# ---------------------------------------------------------------------------


class TestEnsemble:
    def test_optimize_weight_in_range(self):
        from src.training.ensemble import optimize_ensemble_weight

        rng = np.random.default_rng(7)
        y = rng.random(50) * 20
        xgb_preds = y + rng.normal(0, 2, 50)
        seq_preds = y + rng.normal(0, 3, 50)

        alpha = optimize_ensemble_weight(xgb_preds, seq_preds, y)
        assert 0.0 <= alpha <= 1.0

    def test_predictor_weighted_average(self):
        from src.training.ensemble import EnsemblePredictor

        class DummyModel:
            def __init__(self, offset: float):
                self.offset = offset

            def predict(self, X: np.ndarray) -> np.ndarray:
                return np.full(len(X), self.offset)

        ens = EnsemblePredictor(DummyModel(10.0), DummyModel(20.0), alpha=0.6)
        X_dummy = np.zeros((5, 3))
        preds = ens.predict(X_dummy, X_dummy)
        expected = 0.6 * 10.0 + 0.4 * 20.0
        np.testing.assert_allclose(preds, expected)

    def test_build_ensemble(self):
        from src.training.ensemble import build_ensemble

        class DummyModel:
            def predict(self, X: np.ndarray) -> np.ndarray:
                return np.ones(len(X))

        rng = np.random.default_rng(1)
        y = rng.random(20) * 10
        xgb_p = y + rng.normal(0, 1, 20)
        seq_p = y + rng.normal(0, 1, 20)

        predictor, alpha = build_ensemble(xgb_p, seq_p, y, DummyModel(), DummyModel())
        assert 0.0 <= alpha <= 1.0
        assert hasattr(predictor, "predict")


# ---------------------------------------------------------------------------
# StintSequenceDataset tests
# ---------------------------------------------------------------------------


class TestStintSequenceDataset:
    @pytest.fixture(autouse=True)
    def _skip_no_torch(self):
        pytest.importorskip("torch")

    def test_dataset_length_and_getitem(self):
        from src.training.sequence_data import StintSequenceDataset

        X = np.random.default_rng(0).random((20, 5, 4)).astype(np.float32)
        y = np.random.default_rng(1).random(20).astype(np.float32)
        ds = StintSequenceDataset(X, y)
        assert len(ds) == 20
        x_item, y_item = ds[0]
        assert x_item.shape == (5, 4)
        assert y_item.shape == ()


# ---------------------------------------------------------------------------
# TFT tests (guarded — heavy dependency)
# ---------------------------------------------------------------------------


class TestTFT:
    @pytest.fixture(autouse=True)
    def _skip_no_tft(self):
        pytest.importorskip("pytorch_forecasting")

    def test_prepare_tft_dataframe_adds_columns(self):
        from src.training.tft import prepare_tft_dataframe

        df = _make_stint_df(n_stints=2, laps_per_stint=6)
        prep = prepare_tft_dataframe(df, fit_encoder=True)
        assert "stint_id" in prep.columns
        assert "time_idx" in prep.columns
        assert prep["time_idx"].min() == 0

    def test_build_tft_datasets_creates_loaders(self):
        from src.training.tft import build_tft_datasets

        df = _make_stint_df(n_stints=4, laps_per_stint=15)
        train_df = df.iloc[: len(df) // 2].copy()
        val_df = df.iloc[len(df) // 2 :].copy()

        result = build_tft_datasets(train_df, val_df, seq_len=5, batch_size=8)
        assert "train_loader" in result
        assert "val_loader" in result
        assert "training" in result
