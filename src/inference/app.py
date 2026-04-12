"""FastAPI application factory with lifespan, Prometheus metrics, and structured logging."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from src.inference.config import Settings, get_settings
from src.inference.consumer import InferenceConsumer
from src.inference.endpoints import init_routes, router
from src.inference.model_runner import ModelRunner
from src.inference.race_state import RaceStateManager

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(0),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    state = RaceStateManager()
    runner = ModelRunner(settings)
    circuit_cache: dict[str, dict[str, Any]] = {}

    # 1. Load model and warm up
    runner.load()
    runner.warm_up()

    # 2. Initialise Feast online store (best-effort; circuit features are
    #    fetched lazily per-session when the first lap arrives)
    try:
        from feast import FeatureStore

        feast_store = FeatureStore(repo_path=settings.feast_repo_path)
        app.state.feast_store = feast_store
        logger.info("feast_store_initialised", repo=settings.feast_repo_path)
    except Exception:
        logger.warning("feast_store_unavailable", exc_info=True)
        app.state.feast_store = None

    # 3. Wire up routes
    init_routes(state, runner, circuit_cache)

    # 4. Start Kafka consumer
    consumer = InferenceConsumer(settings, state, runner, circuit_cache)
    consumer_task = asyncio.create_task(consumer.start())

    app.state.race_state = state
    app.state.runner = runner
    app.state.consumer = consumer

    logger.info("inference_service_ready")
    yield

    # Shutdown
    consumer.stop()
    consumer_task.cancel()
    try:
        await consumer_task
    except asyncio.CancelledError:
        pass
    logger.info("inference_service_shutdown")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = get_settings()

    app = FastAPI(
        title="F1 Tire Cliff Prediction Service",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings

    app.include_router(router)

    Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
    ).instrument(app).expose(app, endpoint="/metrics")

    return app


# ---------------------------------------------------------------------------
# Entrypoint (``python -m src.inference``)
# ---------------------------------------------------------------------------

app = create_app()
