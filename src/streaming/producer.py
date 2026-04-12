"""Kafka producer that replays a historical race session at configurable speed.

Usage::

    python -m src.streaming.producer --season 2024 --round 12 --speed 10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from confluent_kafka import Producer
from loguru import logger

from src.ingestion.constants import RAW_DATA_DIR
from src.streaming.models import LapCompletedEvent, RaceControlEvent
from src.streaming.session_loader import load_session_timeline


def _delivery_callback(err, msg):
    if err is not None:
        logger.error("Message delivery failed: {}", err)
    else:
        logger.debug(
            "Delivered to {}[{}] @ offset {}",
            msg.topic(),
            msg.partition(),
            msg.offset(),
        )


def _build_producer(bootstrap_servers: str) -> Producer:
    return Producer({
        "bootstrap.servers": bootstrap_servers,
        "client.id": "f1-race-simulator",
        "acks": "all",
    })


def run(
    season: int,
    round_num: int,
    speed: float = 1.0,
    bootstrap_servers: str = "localhost:9092",
    lap_topic: str = "lap_completed",
    rc_topic: str = "race_control",
    raw_data_dir: Path = RAW_DATA_DIR,
) -> None:
    """Replay a historical race session through Kafka."""
    logger.info(
        "Loading session {}/{} Race (speed={:.1f}x)", season, round_num, speed
    )
    timeline = load_session_timeline(season, round_num, raw_data_dir=raw_data_dir)
    if not timeline:
        logger.warning("Empty timeline — nothing to replay")
        return

    producer = _build_producer(bootstrap_servers)
    total = len(timeline)

    logger.info("Starting replay: {} events at {:.1f}x speed", total, speed)

    for i, (delay, event) in enumerate(timeline, 1):
        if delay > 0 and speed > 0:
            time.sleep(delay / speed)

        if isinstance(event, LapCompletedEvent):
            topic = lap_topic
            key = event.driver_number
        elif isinstance(event, RaceControlEvent):
            topic = rc_topic
            key = event.event_type.value
        else:
            continue

        producer.produce(
            topic,
            key=key.encode("utf-8"),
            value=event.model_dump_json().encode("utf-8"),
            callback=_delivery_callback,
        )
        producer.poll(0)

        if isinstance(event, LapCompletedEvent):
            logger.info(
                "[{}/{}] Driver {} Lap {} — {:.3f}s ({})",
                i,
                total,
                event.driver_number,
                event.lap_number,
                event.lap_time or 0.0,
                event.compound or "?",
            )
        else:
            logger.info(
                "[{}/{}] Race Control: {} (lap {})",
                i,
                total,
                event.event_type.value,
                event.lap_number,
            )

    producer.flush(timeout=10)
    logger.info("Replay complete — {} events published", total)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a historical F1 race session through Kafka.",
    )
    parser.add_argument("--season", type=int, required=True, help="Season year")
    parser.add_argument("--round", type=int, required=True, help="Round number")
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speed multiplier (1.0 = real time, 10.0 = 10x fast-forward)",
    )
    parser.add_argument(
        "--bootstrap-servers",
        default="localhost:9092",
        help="Kafka bootstrap servers",
    )
    parser.add_argument("--lap-topic", default="lap_completed", help="Lap events topic")
    parser.add_argument("--rc-topic", default="race_control", help="Race control topic")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="Root directory for raw Parquet files",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    try:
        run(
            season=args.season,
            round_num=args.round,
            speed=args.speed,
            bootstrap_servers=args.bootstrap_servers,
            lap_topic=args.lap_topic,
            rc_topic=args.rc_topic,
            raw_data_dir=args.raw_dir,
        )
    except FileNotFoundError as exc:
        logger.error("{}", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
