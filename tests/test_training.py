"""Tests for training splits, metrics, and baseline helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.training import metrics
from src.training import base as base_mod
from src.training.base import prepare_features, split_by_time, split_by_time_adaptive
from src.training.baseline import xgb_model_fn


def test_split_by_time_masks():
    df = pd.DataFrame(
        {
            "session_id": [
                "2022_5_Race",
                "2024_5_Race",
                "2024_15_Race",
            ],
            "laps_to_cliff": [1.0, 2.0, 3.0],
        }
    )
    train, val, test = split_by_time(df)
    assert len(train) == 1 and train["session_id"].iloc[0] == "2022_5_Race"
    assert len(val) == 1 and val["session_id"].iloc[0] == "2024_5_Race"
    assert len(test) == 1 and test["session_id"].iloc[0] == "2024_15_Race"


def test_evaluate_all_keys_and_mae():
    y_true = np.array([10.0, 5.0, 8.0], dtype=np.float32)
    y_pred = np.array([11.0, 5.0, 10.0], dtype=np.float32)
    df = pd.DataFrame(
        {
            "session_id": ["2024_1_Race"] * 3,
            "driver_number": [55] * 3,
            "stint_number": [2] * 3,
            "is_censored": [0, 0, 0],
            "stint_lap_number": [5, 6, 7],
            "cliff_lap": [15, 11, 15],
        }
    )
    out = metrics.evaluate_all(y_true, y_pred, df, k=3)
    assert set(out.keys()) == {"mae", "precision_at_3", "strategy_accuracy"}
    assert out["mae"] == pytest.approx(1.0)  # mean(|1|, |0|, |2|)


def test_xgb_model_fn_strips_split_meta():
    pytest.importorskip("xgboost")
    X = np.random.default_rng(0).random((20, 4)).astype(np.float32)
    y = np.random.default_rng(1).random(20).astype(np.float32)
    params = {
        "train_size": 999,
        "val_size": 999,
        "test_size": 999,
        "n_estimators": 5,
        "max_depth": 2,
    }
    model = xgb_model_fn(X, y, params, show_progress=False)
    assert hasattr(model, "predict")


def test_precision_at_k_within_three_laps():
    y_true = np.array([10.0, 10.0])
    y_pred = np.array([12.0, 14.0])  # errors 2 and 4
    assert metrics.precision_at_k(y_true, y_pred, k=3) == 0.5


def test_mae_empty_or_all_censored_returns_nan():
    y_true = np.array([1.0, 2.0], dtype=np.float32)
    y_pred = np.array([1.0, 2.0], dtype=np.float32)
    censored = np.array([1, 1], dtype=np.int32)
    assert np.isnan(metrics.mae_laps_to_cliff(y_true, y_pred, censored))
    assert np.isnan(metrics.mae_laps_to_cliff(np.array([]), np.array([]), None))


def test_run_experiment_raises_on_empty_train(monkeypatch):
    """Only 2024 late-season rows → empty train under default temporal split."""
    df = pd.DataFrame({"session_id": ["2024_20_Race", "2024_20_Race"]})
    monkeypatch.setattr(
        base_mod,
        "load_training_data",
        lambda path=None, rebuild=False: df.copy(),
    )

    def _stub_model_fn(X_train, y_train, params, **kwargs):
        raise AssertionError("should not train on empty split")

    with pytest.raises(ValueError, match="no training rows"):
        base_mod.run_experiment(_stub_model_fn, {}, allow_adaptive_split=False)


def test_run_experiment_raises_on_zero_row_dataset(monkeypatch):
    empty = pd.DataFrame(columns=["session_id", "laps_to_cliff"])
    monkeypatch.setattr(
        base_mod,
        "load_training_data",
        lambda path=None, rebuild=False: empty.copy(),
    )

    def _stub_model_fn(X_train, y_train, params, **kwargs):
        raise AssertionError("should not train")

    with pytest.raises(ValueError, match="0 rows"):
        base_mod.run_experiment(_stub_model_fn, {})


def test_prepare_features_encodes_circuit_tire_limitation():
    base_mod._circuit_tire_categories = None
    base_mod._label_encoder = None
    df = pd.DataFrame(
        {
            "stint_lap_number": [1.0, 2.0],
            "circuit_tire_limitation": ["rear", "front"],
            "laps_to_cliff": [5.0, 6.0],
        }
    )
    X, y = prepare_features(df, fit_encoder=True)
    assert X.shape == (2, 2)
    assert X.dtype == np.float32
    assert np.issubdtype(y.dtype, np.floating)
    # Encoded column should be integer-like floats, not NaN from failed coercion
    assert np.isfinite(X[:, 1]).all()


def test_split_by_time_adaptive_partitions_timeline():
    rows = [{"session_id": f"2025_{r}_Race", "x": 1} for r in range(1, 7)]
    df = pd.DataFrame(rows)
    tr, va, te = split_by_time_adaptive(df)
    assert len(tr) + len(va) + len(te) == len(df)
    assert len(tr) >= 1 and len(va) >= 1 and len(te) >= 1
