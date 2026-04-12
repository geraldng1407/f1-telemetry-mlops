"""Background Kafka consumer that processes lap/RC events and triggers inference."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog
from confluent_kafka import Consumer, KafkaError, KafkaException

from src.inference.config import Settings
from src.inference.feature_computer import compute_features
from src.inference.model_runner import ModelRunner
from src.inference.race_state import RaceStateManager
from src.streaming.models import LapCompletedEvent, RaceControlEvent

logger = structlog.get_logger(__name__)


class InferenceConsumer:
    """Consumes Kafka lap and race-control events, runs inference, and
    broadcasts predictions via WebSocket fan-out.
    """

    def __init__(
        self,
        settings: Settings,
        state: RaceStateManager,
        runner: ModelRunner,
        circuit_cache: dict[str, dict[str, Any]],
    ) -> None:
        self._settings = settings
        self._state = state
        self._runner = runner
        self._circuit_cache = circuit_cache
        self._running = False
        self._consumer: Consumer | None = None

    async def start(self) -> None:
        self._running = True
        self._consumer = Consumer(
            {
                "bootstrap.servers": self._settings.kafka_bootstrap_servers,
                "group.id": self._settings.kafka_group_id,
                "auto.offset.reset": "latest",
                "enable.auto.commit": True,
            }
        )
        self._consumer.subscribe(
            [self._settings.lap_topic, self._settings.rc_topic]
        )
        logger.info(
            "kafka_consumer_started",
            topics=[self._settings.lap_topic, self._settings.rc_topic],
        )

        try:
            while self._running:
                await self._poll_once()
        finally:
            self._consumer.close()
            logger.info("kafka_consumer_stopped")

    async def _poll_once(self) -> None:
        loop = asyncio.get_running_loop()
        msg = await loop.run_in_executor(None, self._consumer.poll, 1.0)
        if msg is None:
            return

        err = msg.error()
        if err is not None:
            if err.code() == KafkaError._PARTITION_EOF:
                return
            logger.error("kafka_error", error=str(err))
            raise KafkaException(err)

        topic = msg.topic()
        try:
            payload = json.loads(msg.value().decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.warning("malformed_message", topic=topic)
            return

        if topic == self._settings.lap_topic:
            await self._handle_lap(payload)
        elif topic == self._settings.rc_topic:
            await self._handle_rc(payload)

    async def _handle_lap(self, payload: dict[str, Any]) -> None:
        event = LapCompletedEvent.model_validate(payload)
        driver = await self._state.update_lap(event)

        session = self._state.get_session_state(event.session_id)
        if session is None:
            return

        circuit_feats = self._circuit_cache.get(event.session_id)
        features = compute_features(driver, session, circuit_feats)

        prediction = self._runner.predict(
            features,
            driver_number=event.driver_number,
            lap_number=event.lap_number,
            stint_number=driver.stint_number,
        )
        await self._state.store_prediction(
            event.session_id, event.driver_number, prediction
        )

        await self._state.broadcast(
            event.session_id, prediction.model_dump_json()
        )
        logger.debug(
            "prediction_made",
            session=event.session_id,
            driver=event.driver_number,
            lap=event.lap_number,
            laps_to_cliff=prediction.predicted_laps_to_cliff,
        )

    async def _handle_rc(self, payload: dict[str, Any]) -> None:
        event = RaceControlEvent.model_validate(payload)
        await self._state.update_race_control(event)
        logger.info(
            "race_control_update",
            session=event.session_id,
            event_type=event.event_type.value,
            lap=event.lap_number,
        )

    def stop(self) -> None:
        self._running = False
