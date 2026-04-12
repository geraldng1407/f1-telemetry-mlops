"""REST and WebSocket route definitions."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.inference.feature_computer import compute_features
from src.inference.model_runner import ModelRunner
from src.inference.models import PredictionResult, RaceStateResponse
from src.inference.race_state import RaceStateManager
from src.streaming.models import LapCompletedEvent

logger = structlog.get_logger(__name__)

router = APIRouter()

# These are injected at app startup via app.state and retrieved through
# request.app.state inside each handler.  The module-level references are
# set by ``init_routes``.
_state: RaceStateManager | None = None
_runner: ModelRunner | None = None
_circuit_cache: dict | None = None


def init_routes(
    state: RaceStateManager,
    runner: ModelRunner,
    circuit_cache: dict,
) -> None:
    global _state, _runner, _circuit_cache  # noqa: PLW0603
    _state = state
    _runner = runner
    _circuit_cache = circuit_cache


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------


@router.post("/predict", response_model=PredictionResult)
async def predict(event: LapCompletedEvent) -> PredictionResult:
    """Given a single lap event, return the predicted laps_to_cliff."""
    assert _state is not None and _runner is not None

    driver = await _state.update_lap(event)
    session = _state.get_session_state(event.session_id)
    assert session is not None

    circuit_feats = (_circuit_cache or {}).get(event.session_id)
    features = compute_features(driver, session, circuit_feats)

    prediction = _runner.predict(
        features,
        driver_number=event.driver_number,
        lap_number=event.lap_number,
        stint_number=driver.stint_number,
    )
    await _state.store_prediction(event.session_id, event.driver_number, prediction)
    return prediction


# ---------------------------------------------------------------------------
# GET /race/{session_id}/state
# ---------------------------------------------------------------------------


@router.get("/race/{session_id}/state", response_model=RaceStateResponse)
async def race_state(session_id: str) -> RaceStateResponse:
    """Return the current race state for all drivers in a session."""
    assert _state is not None
    return _state.build_race_state_response(session_id)


# ---------------------------------------------------------------------------
# WebSocket /ws/race/{session_id}
# ---------------------------------------------------------------------------


@router.websocket("/ws/race/{session_id}")
async def ws_race(websocket: WebSocket, session_id: str) -> None:
    """Push predictions to connected clients in real time."""
    assert _state is not None
    await websocket.accept()

    snapshot = _state.build_race_state_response(session_id)
    await websocket.send_text(snapshot.model_dump_json())

    q = _state.subscribe(session_id)
    try:
        while True:
            msg = await q.get()
            await websocket.send_text(msg)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        _state.unsubscribe(session_id, q)
