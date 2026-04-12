"""In-memory race state management with WebSocket fan-out."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from src.inference.models import DriverStateResponse, PredictionResult, RaceStateResponse
from src.streaming.models import LapCompletedEvent, RaceControlEvent, RaceControlEventType


@dataclass
class LapRecord:
    """Minimal per-lap data kept in the stint history."""

    lap_number: int
    lap_time: float | None
    sector_times: list[float | None]
    tire_age: int | None
    track_temp: float | None
    gap_ahead: float | None
    is_sc_lap: bool = False


@dataclass
class DriverState:
    stint_number: int = 1
    compound: str | None = None
    laps: list[LapRecord] = field(default_factory=list)
    latest_prediction: PredictionResult | None = None
    dirty_air_cumulative: int = 0

    @property
    def stint_lap_number(self) -> int:
        return len(self.laps)

    @property
    def tire_age(self) -> int | None:
        if self.laps:
            return self.laps[-1].tire_age
        return None

    @property
    def lap_number(self) -> int:
        if self.laps:
            return self.laps[-1].lap_number
        return 0


@dataclass
class SessionState:
    session_id: str
    drivers: dict[str, DriverState] = field(default_factory=dict)
    race_control_status: RaceControlEventType = RaceControlEventType.CLEAR
    total_laps_seen: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    ws_subscribers: list[asyncio.Queue[str]] = field(default_factory=list)


class RaceStateManager:
    """Manages per-session, per-driver race state and WebSocket fan-out."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def _get_session(self, session_id: str) -> SessionState:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState(session_id=session_id)
        return self._sessions[session_id]

    def get_session_state(self, session_id: str) -> SessionState | None:
        return self._sessions.get(session_id)

    async def update_lap(self, event: LapCompletedEvent) -> DriverState:
        """Ingest a lap event and return the updated driver state.

        Detects stint changes when compound changes or tire_age resets to a
        value lower than the previous lap's tire_age.
        """
        session = self._get_session(event.session_id)
        async with session.lock:
            session.total_laps_seen += 1
            driver = session.drivers.get(event.driver_number)

            if driver is None:
                driver = DriverState(stint_number=1, compound=event.compound)
                session.drivers[event.driver_number] = driver
            else:
                is_new_stint = False
                if event.compound and event.compound != driver.compound:
                    is_new_stint = True
                elif (
                    event.tire_age is not None
                    and driver.tire_age is not None
                    and event.tire_age < driver.tire_age
                ):
                    is_new_stint = True

                if is_new_stint:
                    driver.stint_number += 1
                    driver.compound = event.compound
                    driver.laps = []
                    driver.dirty_air_cumulative = 0
                    driver.latest_prediction = None

            is_sc = session.race_control_status in (
                RaceControlEventType.SAFETY_CAR,
                RaceControlEventType.VSC,
            )

            lap = LapRecord(
                lap_number=event.lap_number,
                lap_time=event.lap_time,
                sector_times=list(event.sector_times),
                tire_age=event.tire_age,
                track_temp=event.track_temp,
                gap_ahead=event.gap_ahead,
                is_sc_lap=is_sc,
            )
            driver.laps.append(lap)

            from src.features.constants import DIRTY_AIR_THRESHOLD_S

            if event.gap_ahead is not None and event.gap_ahead < DIRTY_AIR_THRESHOLD_S:
                driver.dirty_air_cumulative += 1

            return driver

    async def update_race_control(self, event: RaceControlEvent) -> None:
        session = self._get_session(event.session_id)
        async with session.lock:
            session.race_control_status = event.event_type

    async def store_prediction(
        self, session_id: str, driver_number: str, prediction: PredictionResult
    ) -> None:
        session = self._get_session(session_id)
        async with session.lock:
            driver = session.drivers.get(driver_number)
            if driver is not None:
                driver.latest_prediction = prediction

    def build_race_state_response(self, session_id: str) -> RaceStateResponse:
        session = self._sessions.get(session_id)
        if session is None:
            return RaceStateResponse(session_id=session_id)

        drivers = []
        for dn, ds in session.drivers.items():
            drivers.append(
                DriverStateResponse(
                    driver_number=dn,
                    stint_number=ds.stint_number,
                    compound=ds.compound,
                    tire_age=ds.tire_age,
                    lap_number=ds.lap_number,
                    latest_prediction=ds.latest_prediction,
                )
            )
        return RaceStateResponse(
            session_id=session_id,
            race_control_status=session.race_control_status.value,
            drivers=drivers,
        )

    # ------------------------------------------------------------------
    # WebSocket fan-out
    # ------------------------------------------------------------------

    def subscribe(self, session_id: str) -> asyncio.Queue[str]:
        session = self._get_session(session_id)
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=64)
        session.ws_subscribers.append(q)
        return q

    def unsubscribe(self, session_id: str, q: asyncio.Queue[str]) -> None:
        session = self._sessions.get(session_id)
        if session is not None:
            try:
                session.ws_subscribers.remove(q)
            except ValueError:
                pass

    async def broadcast(self, session_id: str, message: str) -> None:
        session = self._sessions.get(session_id)
        if session is None:
            return
        for q in list(session.ws_subscribers):
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                pass
