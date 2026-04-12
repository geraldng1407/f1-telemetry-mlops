"""Dynamic feature computation from in-memory race state.

Mirrors the formulas in ``src.features.engineering`` but operates on the
per-driver lap list accumulated by :class:`RaceStateManager` rather than
a full Parquet DataFrame.
"""

from __future__ import annotations

from typing import Any

from src.features.constants import (
    DIRTY_AIR_THRESHOLD_S,
    ENGINEERED_FEATURE_COLUMNS,
    FUEL_CORRECTION_PER_LAP,
    ROLLING_WINDOWS,
)
from src.inference.race_state import DriverState, SessionState


def _rolling_mean(values: list[float], window: int) -> float:
    """Rolling mean with min_periods=1 (matches pandas default)."""
    tail = values[-window:]
    if not tail:
        return float("nan")
    return sum(tail) / len(tail)


def _rolling_var(values: list[float], window: int) -> float:
    """Sample variance with min_periods=2 (matches pandas default)."""
    tail = values[-window:]
    n = len(tail)
    if n < 2:
        return float("nan")
    mean = sum(tail) / n
    return sum((x - mean) ** 2 for x in tail) / (n - 1)


def compute_features(
    driver: DriverState,
    session: SessionState,
    circuit_features: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the full feature dict for the driver's most recent lap.

    Returns a dict keyed by ``ENGINEERED_FEATURE_COLUMNS`` names (plus
    ``compound``).  Values that cannot be computed are set to ``NaN``.
    """
    features: dict[str, Any] = {}
    laps = driver.laps

    if not laps:
        return {col: float("nan") for col in ENGINEERED_FEATURE_COLUMNS}

    current = laps[-1]
    stint_lap = driver.stint_lap_number  # 1-indexed count of laps in stint

    # --- stint_lap_number ---
    features["stint_lap_number"] = stint_lap

    # --- rolling stats (lap times within the current stint) ---
    lap_times = [lap.lap_time for lap in laps if lap.lap_time is not None]
    for w in ROLLING_WINDOWS:
        features[f"rolling_mean_laptime_{w}"] = (
            _rolling_mean(lap_times, w) if lap_times else float("nan")
        )
        features[f"rolling_var_laptime_{w}"] = (
            _rolling_var(lap_times, w) if lap_times else float("nan")
        )

    # --- sector deltas from stint-best ---
    for i in range(3):
        col = f"sector{i + 1}_delta_from_best"
        sector_vals = [
            lap.sector_times[i] for lap in laps if lap.sector_times[i] is not None
        ]
        cur = current.sector_times[i]
        if sector_vals and cur is not None:
            features[col] = cur - min(sector_vals)
        else:
            features[col] = float("nan")

    # --- tire age ---
    features["tire_age_laps"] = (
        float(current.tire_age) if current.tire_age is not None else float("nan")
    )

    # --- fuel-corrected lap time ---
    if current.lap_time is not None:
        features["fuel_corrected_laptime"] = current.lap_time + FUEL_CORRECTION_PER_LAP * stint_lap
    else:
        features["fuel_corrected_laptime"] = float("nan")

    # --- environmental ---
    features["track_temp_c"] = (
        current.track_temp if current.track_temp is not None else float("nan")
    )
    features["air_temp_c"] = float("nan")  # not on the event; filled from Feast if available
    features["humidity"] = float("nan")
    features["rainfall"] = float("nan")

    features["track_evolution_index"] = min(current.lap_number / 57.0, 1.0)

    # --- dirty air ---
    features["gap_to_car_ahead_s"] = (
        current.gap_ahead if current.gap_ahead is not None else float("nan")
    )
    is_dirty = (
        current.gap_ahead is not None and current.gap_ahead < DIRTY_AIR_THRESHOLD_S
    )
    features["is_dirty_air"] = int(is_dirty)
    features["dirty_air_cumulative_laps"] = driver.dirty_air_cumulative

    # --- lap flags ---
    features["is_inlap"] = 0  # not detectable from streaming events alone
    features["is_outlap"] = 0
    features["is_sc_lap"] = int(current.is_sc_lap)

    # --- circuit static features (from Feast or cache) ---
    circuit_keys = [
        "circuit_high_speed_corners",
        "circuit_medium_speed_corners",
        "circuit_low_speed_corners",
        "circuit_abrasiveness",
        "circuit_altitude_m",
        "circuit_tire_limitation",
        "circuit_street_circuit",
    ]
    if circuit_features:
        for k in circuit_keys:
            features[k] = circuit_features.get(k, float("nan"))
    else:
        for k in circuit_keys:
            features[k] = float("nan")

    # compound is appended separately (not in ENGINEERED_FEATURE_COLUMNS but
    # added by prepare_features)
    features["compound"] = driver.compound

    return features
