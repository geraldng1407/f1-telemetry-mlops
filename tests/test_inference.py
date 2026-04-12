"""Tests for the inference service — feature computation, race state, and endpoints."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

from src.inference.feature_computer import (
    _rolling_mean,
    _rolling_var,
    compute_features,
)
from src.inference.models import (
    DriverStateResponse,
    PredictionResult,
    RaceStateResponse,
)
from src.inference.race_state import (
    DriverState,
    LapRecord,
    RaceStateManager,
    SessionState,
)
from src.streaming.models import (
    LapCompletedEvent,
    RaceControlEvent,
    RaceControlEventType,
)

# ---------------------------------------------------------------------------
# Rolling stat helpers
# ---------------------------------------------------------------------------


class TestRollingStats:
    def test_rolling_mean_full_window(self):
        assert _rolling_mean([1.0, 2.0, 3.0, 4.0, 5.0], 3) == pytest.approx(4.0)

    def test_rolling_mean_partial_window(self):
        assert _rolling_mean([10.0], 3) == pytest.approx(10.0)

    def test_rolling_mean_empty(self):
        assert math.isnan(_rolling_mean([], 3))

    def test_rolling_var_full_window(self):
        # var([3,4,5]) with ddof=1 = 1.0
        assert _rolling_var([1.0, 2.0, 3.0, 4.0, 5.0], 3) == pytest.approx(1.0)

    def test_rolling_var_single_value(self):
        assert math.isnan(_rolling_var([5.0], 3))

    def test_rolling_var_two_values(self):
        # var([4,5]) with ddof=1 = 0.5
        assert _rolling_var([4.0, 5.0], 3) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------


_SECTORS = [28.0, 32.0, 30.0]


def _lap(
    num: int,
    time: float = 90.0,
    sectors: list[float | None] | None = None,
    age: int = 1,
    temp: float = 45.0,
    gap: float | None = None,
) -> dict:
    return {
        "lap_number": num,
        "lap_time": time,
        "sector_times": sectors or list(_SECTORS),
        "tire_age": age,
        "track_temp": temp,
        "gap_ahead": gap,
    }


def _make_driver(laps: list[dict]) -> DriverState:
    ds = DriverState(stint_number=1, compound="SOFT")
    for lap in laps:
        ds.laps.append(LapRecord(**lap))
    return ds


def _make_session(
    session_id: str = "2024_12_Race",
    total_laps: int = 10,
) -> SessionState:
    ss = SessionState(session_id=session_id)
    ss.total_laps_seen = total_laps
    return ss


class TestComputeFeatures:
    def test_stint_lap_number(self):
        driver = _make_driver([
            _lap(1, 90.0),
            _lap(2, 91.0, [28.5, 32.5, 30.0], age=2, gap=1.2),
        ])
        session = _make_session()
        feats = compute_features(driver, session)
        assert feats["stint_lap_number"] == 2

    def test_rolling_stats_computed(self):
        driver = _make_driver([
            _lap(i, 90.0 + i, age=i) for i in range(1, 6)
        ])
        session = _make_session()
        feats = compute_features(driver, session)
        assert not math.isnan(feats["rolling_mean_laptime_3"])
        assert not math.isnan(feats["rolling_mean_laptime_5"])
        assert not math.isnan(feats["rolling_var_laptime_3"])

    def test_sector_deltas(self):
        driver = _make_driver([
            _lap(1, 90.0),
            _lap(2, 91.0, [29.0, 33.0, 30.5], age=2),
        ])
        session = _make_session()
        feats = compute_features(driver, session)
        assert feats["sector1_delta_from_best"] == pytest.approx(1.0)
        assert feats["sector2_delta_from_best"] == pytest.approx(1.0)
        assert feats["sector3_delta_from_best"] == pytest.approx(0.5)

    def test_fuel_corrected_laptime(self):
        driver = _make_driver([_lap(1, 90.0)])
        session = _make_session()
        feats = compute_features(driver, session)
        assert feats["fuel_corrected_laptime"] == pytest.approx(90.06)

    def test_dirty_air_flag(self):
        driver = _make_driver([_lap(1, 90.0, gap=0.8)])
        driver.dirty_air_cumulative = 1
        session = _make_session()
        feats = compute_features(driver, session)
        assert feats["is_dirty_air"] == 1
        assert feats["dirty_air_cumulative_laps"] == 1

    def test_no_dirty_air_when_gap_large(self):
        driver = _make_driver([_lap(1, 90.0, gap=3.0)])
        session = _make_session()
        feats = compute_features(driver, session)
        assert feats["is_dirty_air"] == 0

    def test_circuit_features_from_cache(self):
        driver = _make_driver([_lap(1, 90.0)])
        session = _make_session()
        circuit = {
            "circuit_high_speed_corners": 5,
            "circuit_abrasiveness": 3,
        }
        feats = compute_features(
            driver, session, circuit_features=circuit,
        )
        assert feats["circuit_high_speed_corners"] == 5
        assert feats["circuit_abrasiveness"] == 3

    def test_empty_driver_returns_nan(self):
        driver = DriverState(stint_number=1, compound="SOFT")
        session = _make_session()
        feats = compute_features(driver, session)
        assert math.isnan(feats["stint_lap_number"])


# ---------------------------------------------------------------------------
# Race state management
# ---------------------------------------------------------------------------


class TestRaceStateManager:
    @pytest.fixture()
    def manager(self) -> RaceStateManager:
        return RaceStateManager()

    @pytest.mark.asyncio
    async def test_update_lap_creates_driver(self, manager: RaceStateManager):
        event = LapCompletedEvent(
            session_id="2024_12_Race",
            driver_number="1",
            lap_number=1,
            lap_time=90.0,
            compound="SOFT",
            tire_age=1,
        )
        driver = await manager.update_lap(event)
        assert driver.stint_number == 1
        assert driver.compound == "SOFT"
        assert len(driver.laps) == 1

    @pytest.mark.asyncio
    async def test_stint_change_on_compound_change(self, manager: RaceStateManager):
        e1 = LapCompletedEvent(
            session_id="2024_12_Race", driver_number="1", lap_number=1,
            compound="SOFT", tire_age=10,
        )
        e2 = LapCompletedEvent(
            session_id="2024_12_Race", driver_number="1", lap_number=2,
            compound="HARD", tire_age=1,
        )
        await manager.update_lap(e1)
        driver = await manager.update_lap(e2)
        assert driver.stint_number == 2
        assert driver.compound == "HARD"
        assert len(driver.laps) == 1  # laps reset on new stint

    @pytest.mark.asyncio
    async def test_stint_change_on_tire_age_reset(self, manager: RaceStateManager):
        e1 = LapCompletedEvent(
            session_id="2024_12_Race", driver_number="1", lap_number=10,
            compound="SOFT", tire_age=10,
        )
        e2 = LapCompletedEvent(
            session_id="2024_12_Race", driver_number="1", lap_number=11,
            compound="SOFT", tire_age=1,
        )
        await manager.update_lap(e1)
        driver = await manager.update_lap(e2)
        assert driver.stint_number == 2

    @pytest.mark.asyncio
    async def test_race_control_update(self, manager: RaceStateManager):
        event = RaceControlEvent(
            session_id="2024_12_Race", lap_number=5,
            event_type=RaceControlEventType.SAFETY_CAR, timestamp=450.0,
        )
        await manager.update_race_control(event)
        session = manager.get_session_state("2024_12_Race")
        assert session is not None
        assert session.race_control_status == RaceControlEventType.SAFETY_CAR

    @pytest.mark.asyncio
    async def test_sc_lap_flagged(self, manager: RaceStateManager):
        rc = RaceControlEvent(
            session_id="2024_12_Race", lap_number=5,
            event_type=RaceControlEventType.SAFETY_CAR, timestamp=450.0,
        )
        await manager.update_race_control(rc)

        lap = LapCompletedEvent(
            session_id="2024_12_Race", driver_number="1", lap_number=6,
            lap_time=95.0, compound="SOFT", tire_age=6,
        )
        driver = await manager.update_lap(lap)
        assert driver.laps[-1].is_sc_lap is True

    def test_build_race_state_response_empty(self, manager: RaceStateManager):
        resp = manager.build_race_state_response("unknown")
        assert resp.session_id == "unknown"
        assert resp.drivers == []

    @pytest.mark.asyncio
    async def test_build_race_state_with_drivers(self, manager: RaceStateManager):
        e = LapCompletedEvent(
            session_id="2024_12_Race", driver_number="44", lap_number=3,
            lap_time=91.0, compound="MEDIUM", tire_age=3,
        )
        await manager.update_lap(e)
        resp = manager.build_race_state_response("2024_12_Race")
        assert len(resp.drivers) == 1
        assert resp.drivers[0].driver_number == "44"

    @pytest.mark.asyncio
    async def test_websocket_subscribe_and_broadcast(self, manager: RaceStateManager):
        q = manager.subscribe("2024_12_Race")
        await manager.broadcast("2024_12_Race", '{"test": true}')
        msg = q.get_nowait()
        assert msg == '{"test": true}'

    @pytest.mark.asyncio
    async def test_websocket_unsubscribe(self, manager: RaceStateManager):
        q = manager.subscribe("2024_12_Race")
        manager.unsubscribe("2024_12_Race", q)
        await manager.broadcast("2024_12_Race", '{"test": true}')
        assert q.empty()


# ---------------------------------------------------------------------------
# Response model tests
# ---------------------------------------------------------------------------


class TestResponseModels:
    def test_prediction_result_roundtrip(self):
        p = PredictionResult(
            driver_number="1",
            lap_number=10,
            stint_number=2,
            compound="HARD",
            tire_age=5,
            predicted_laps_to_cliff=12.5,
            confidence_lower=7.6,
            confidence_upper=17.4,
        )
        restored = PredictionResult.model_validate_json(p.model_dump_json())
        assert restored == p

    def test_race_state_response_roundtrip(self):
        resp = RaceStateResponse(
            session_id="2024_12_Race",
            race_control_status="CLEAR",
            drivers=[
                DriverStateResponse(
                    driver_number="1", stint_number=1, compound="SOFT",
                    tire_age=5, lap_number=5,
                ),
            ],
        )
        restored = RaceStateResponse.model_validate_json(resp.model_dump_json())
        assert restored.session_id == "2024_12_Race"
        assert len(restored.drivers) == 1


# ---------------------------------------------------------------------------
# Endpoint tests (using HTTPX test client)
# ---------------------------------------------------------------------------


class TestEndpoints:
    @pytest.fixture()
    def mock_runner(self):
        runner = MagicMock()
        runner.predict.return_value = PredictionResult(
            driver_number="1",
            lap_number=1,
            stint_number=1,
            compound="SOFT",
            tire_age=1,
            predicted_laps_to_cliff=15.0,
            confidence_lower=10.1,
            confidence_upper=19.9,
        )
        return runner

    @pytest.fixture()
    def client(self, mock_runner):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from src.inference.endpoints import init_routes, router
        from src.inference.race_state import RaceStateManager

        state = RaceStateManager()
        circuit_cache: dict = {}
        init_routes(state, mock_runner, circuit_cache)

        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_predict_endpoint(self, client, mock_runner):
        payload = {
            "session_id": "2024_12_Race",
            "driver_number": "1",
            "lap_number": 1,
            "lap_time": 90.0,
            "sector_times": [28.0, 32.0, 30.0],
            "compound": "SOFT",
            "tire_age": 1,
            "track_temp": 45.0,
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["driver_number"] == "1"
        assert "predicted_laps_to_cliff" in data
        assert "confidence_lower" in data
        assert "confidence_upper" in data

    def test_race_state_endpoint_empty(self, client):
        resp = client.get("/race/unknown_session/state")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "unknown_session"
        assert data["drivers"] == []

    def test_race_state_after_predict(self, client, mock_runner):
        payload = {
            "session_id": "2024_12_Race",
            "driver_number": "44",
            "lap_number": 3,
            "lap_time": 91.0,
            "compound": "MEDIUM",
            "tire_age": 3,
        }
        client.post("/predict", json=payload)

        resp = client.get("/race/2024_12_Race/state")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["drivers"]) == 1
        assert data["drivers"][0]["driver_number"] == "44"
