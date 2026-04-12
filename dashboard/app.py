"""F1 Tire Strategy Dashboard — Streamlit MVP.

Polls the FastAPI inference service for live race state and displays:
- A leaderboard with tire cliff predictions and risk levels
- Per-driver degradation charts with predicted cliff overlay
"""

from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_URL = os.getenv("API_URL", "http://localhost:8000")
REFRESH_INTERVAL_MS = int(os.getenv("REFRESH_INTERVAL_MS", "3000"))

RISK_THRESHOLDS = {"red": 5, "yellow": 10}

COMPOUND_COLORS: dict[str | None, str] = {
    "SOFT": "#FF3333",
    "MEDIUM": "#FFD700",
    "HARD": "#CCCCCC",
    "INTERMEDIATE": "#39B54A",
    "WET": "#3399FF",
    None: "#888888",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _risk_level(laps_to_cliff: float | None) -> str:
    if laps_to_cliff is None:
        return "—"
    if laps_to_cliff <= RISK_THRESHOLDS["red"]:
        return "🔴 Critical"
    if laps_to_cliff <= RISK_THRESHOLDS["yellow"]:
        return "🟡 Warning"
    return "🟢 Safe"


def _fetch_race_state(session_id: str) -> dict | None:
    try:
        resp = requests.get(f"{API_URL}/race/{session_id}/state", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


def _build_leaderboard(drivers: list[dict]) -> pd.DataFrame:
    rows = []
    for d in drivers:
        pred = d.get("latest_prediction")
        laps_to_cliff = pred["predicted_laps_to_cliff"] if pred else None
        conf_lo = pred["confidence_lower"] if pred else None
        conf_hi = pred["confidence_upper"] if pred else None
        confidence = f"[{conf_lo:.1f}, {conf_hi:.1f}]" if conf_lo is not None else "—"

        rows.append(
            {
                "Driver": d["driver_number"],
                "Compound": d.get("compound") or "—",
                "Tire Age": d.get("tire_age") or 0,
                "Laps to Cliff": round(laps_to_cliff, 1) if laps_to_cliff is not None else None,
                "Confidence": confidence,
                "Risk": _risk_level(laps_to_cliff),
                "_sort_key": laps_to_cliff if laps_to_cliff is not None else 999,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("_sort_key").drop(columns=["_sort_key"]).reset_index(drop=True)


def _build_degradation_chart(driver: dict) -> go.Figure:
    lap_times = driver.get("lap_times", [])
    pred = driver.get("latest_prediction")
    compound = driver.get("compound")

    valid = [lt for lt in lap_times if lt.get("lap_time") is not None]
    ages = [lt["tire_age"] for lt in valid]
    times = [lt["lap_time"] for lt in valid]

    fig = go.Figure()

    line_color = COMPOUND_COLORS.get(compound, "#888888")
    fig.add_trace(
        go.Scatter(
            x=ages,
            y=times,
            mode="lines+markers",
            name="Lap Time",
            line={"color": line_color, "width": 2},
            marker={"size": 5},
        )
    )

    if pred and ages:
        current_age = ages[-1]
        cliff_age = current_age + pred["predicted_laps_to_cliff"]
        cliff_lo = current_age + pred["confidence_lower"]
        cliff_hi = current_age + pred["confidence_upper"]

        fig.add_vrect(
            x0=cliff_lo,
            x1=cliff_hi,
            fillcolor="rgba(255, 0, 0, 0.08)",
            line_width=0,
            annotation_text="Cliff zone",
            annotation_position="top left",
        )
        fig.add_vline(
            x=cliff_age,
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text=f"Cliff ~{cliff_age:.0f}",
            annotation_position="top right",
        )

    fig.update_layout(
        xaxis_title="Tire Age (laps)",
        yaxis_title="Lap Time (s)",
        template="plotly_dark",
        height=400,
        margin={"l": 50, "r": 20, "t": 30, "b": 50},
        legend={"orientation": "h", "y": -0.15},
    )

    return fig


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="F1 Tire Strategy Dashboard",
    page_icon="🏎️",
    layout="wide",
)

st_autorefresh(interval=REFRESH_INTERVAL_MS, key="auto_refresh")

st.title("F1 Tire Strategy Dashboard")

# ---------------------------------------------------------------------------
# Session selector
# ---------------------------------------------------------------------------

session_id = st.sidebar.text_input("Session ID", value="latest", help="Race session identifier")
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**API:** `{API_URL}`  \n"
    f"**Refresh:** every {REFRESH_INTERVAL_MS / 1000:.1f}s"
)

# ---------------------------------------------------------------------------
# Fetch data
# ---------------------------------------------------------------------------

state = _fetch_race_state(session_id)

if state is None:
    st.warning(
        f"Cannot reach inference API at `{API_URL}`. "
        "Make sure the FastAPI service is running."
    )
    st.stop()

drivers: list[dict] = state.get("drivers", [])

if not drivers:
    st.info(f"No driver data for session **{state['session_id']}** yet. Waiting for lap events…")
    st.stop()

# ---------------------------------------------------------------------------
# Status bar
# ---------------------------------------------------------------------------

col_status, col_drivers, col_session = st.columns(3)
rc_status = state.get("race_control_status", "CLEAR")
status_emoji = {"CLEAR": "🟢", "SAFETY_CAR": "🟡", "VSC": "🟡", "RED_FLAG": "🔴"}.get(
    rc_status, "⚪"
)
col_status.metric("Race Status", f"{status_emoji} {rc_status}")
col_drivers.metric("Drivers Tracked", len(drivers))
col_session.metric("Session", state["session_id"])

# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

st.subheader("Live Leaderboard")

df = _build_leaderboard(drivers)

st.dataframe(
    df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Tire Age": st.column_config.NumberColumn(format="%d laps"),
        "Laps to Cliff": st.column_config.NumberColumn(format="%.1f"),
    },
)

# ---------------------------------------------------------------------------
# Per-driver degradation chart
# ---------------------------------------------------------------------------

st.subheader("Degradation Curve")

driver_numbers = [d["driver_number"] for d in drivers]
selected = st.selectbox("Select Driver", driver_numbers)

if selected:
    driver_data = next((d for d in drivers if d["driver_number"] == selected), None)
    if driver_data:
        chart_col, info_col = st.columns([3, 1])

        with chart_col:
            fig = _build_degradation_chart(driver_data)
            st.plotly_chart(fig, use_container_width=True)

        with info_col:
            pred = driver_data.get("latest_prediction")
            st.markdown(f"**Driver** {driver_data['driver_number']}")
            st.markdown(f"**Compound** {driver_data.get('compound') or '—'}")
            st.markdown(f"**Tire Age** {driver_data.get('tire_age') or 0} laps")
            st.markdown(f"**Stint** {driver_data['stint_number']}")
            if pred:
                st.markdown(f"**Laps to Cliff** {pred['predicted_laps_to_cliff']:.1f}")
                st.markdown(
                    f"**Confidence** [{pred['confidence_lower']:.1f}, "
                    f"{pred['confidence_upper']:.1f}]"
                )
                st.markdown(f"**Risk** {_risk_level(pred['predicted_laps_to_cliff'])}")
            else:
                st.markdown("*No prediction yet*")
