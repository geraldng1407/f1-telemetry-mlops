from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import fastf1
from dagster import ConfigurableResource
from loguru import logger

from src.ingestion.constants import FASTF1_CACHE_DIR

if TYPE_CHECKING:
    from fastf1.core import Session


class FastF1Resource(ConfigurableResource):
    """Dagster resource wrapping FastF1 session loading with caching."""

    cache_dir: str = str(FASTF1_CACHE_DIR)
    force_reload: bool = False

    def setup_for_execution(self, context) -> None:
        cache_path = Path(self.cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(cache_path))

    def load_session(
        self, year: int, round_num: int, session_type: str
    ) -> Session:
        logger.info(
            "Loading session: {} round {} – {}",
            year,
            round_num,
            session_type,
        )
        session = fastf1.get_session(year, round_num, session_type)
        session.load(laps=True, telemetry=True, weather=True)
        return session

    def get_session_metadata(
        self, year: int, round_num: int, session_type: str
    ) -> Session:
        """Get a session object without loading heavy data (laps/telemetry/weather).

        Useful for accessing session.event metadata only.
        """
        session = fastf1.get_session(year, round_num, session_type)
        return session

    def get_event_schedule(self, year: int) -> fastf1.events.EventSchedule:
        return fastf1.get_event_schedule(year, include_testing=False)
