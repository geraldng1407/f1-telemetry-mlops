# F1 Telemetry MLOps — Tire Degradation & Pit Strategy Predictor

A production-grade machine learning system that predicts tire degradation "cliffs" and optimal pit-stop windows for Formula 1 races. Built on real telemetry data from the FastF1 API, orchestrated with modern MLOps tooling, and designed to handle the concept drift caused by mid-season car upgrades.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture Diagram](#architecture-diagram)
- [Tech Stack](#tech-stack)
- [Phase 1 — Data Acquisition & Feature Store](#phase-1--data-acquisition--feature-store)
- [Phase 2 — Modeling the Tire Cliff](#phase-2--modeling-the-tire-cliff)
- [Phase 3 — Concept Drift & MLOps](#phase-3--concept-drift--mlops)
- [Phase 4 — Live Inference Pipeline](#phase-4--live-inference-pipeline)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [License](#license)

---

## Project Overview

### The Problem

Tire degradation in F1 is non-linear. A car can post consistent lap times for 15+ laps, then suddenly lose 2+ seconds per lap — the **tire cliff**. Teams that predict this moment accurately gain a massive strategic advantage: pitting one lap too late costs track position; pitting one lap too early wastes tire life.

### The Goal

Build an end-to-end ML system that:

1. Ingests historical and simulated-live F1 telemetry data.
2. Engineers features that capture the hidden signals of degradation (rolling variance, track temperature interaction, dirty-air exposure).
3. Predicts **"laps remaining until lap time drops by >1.5s from the stint average"** — the cliff.
4. Handles **concept drift** caused by aerodynamic upgrades, regulation changes, and varying track surfaces.
5. Serves predictions through a low-latency inference pipeline, visualized on a real-time dashboard.

### Why This Matters (for MLOps)

This isn't a static Kaggle dataset. The data distribution shifts every 2 weeks (new race), every few races (car upgrades), and every season (regulation changes). Solving concept drift, automated retraining, and model monitoring in this domain demonstrates production ML engineering at its most demanding.

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          DATA ACQUISITION                               │
│                                                                          │
│   FastF1 API ──► Dagster Pipeline ──► Raw Data Lake (Parquet/S3)        │
│                                           │                              │
│                                           ▼                              │
│                                   Feature Engineering                    │
│                                           │                              │
│                                           ▼                              │
│                                   Feast Feature Store                    │
│                              ┌────────────┴────────────┐                │
│                              │                         │                 │
│                        Offline Store              Online Store           │
│                        (Training)                 (Inference)            │
└──────────────────────────────┼─────────────────────────┼─────────────────┘
                               │                         │
                               ▼                         ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                          MODEL TRAINING                                  │
│                                                                          │
│   XGBoost / LightGBM (baseline)                                         │
│   LSTM / Temporal Fusion Transformer (advanced)                         │
│                         │                                                │
│                         ▼                                                │
│              MLflow Experiment Tracking                                  │
│              Model Registry & Versioning                                 │
└─────────────────────────┼────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                          MLOPS / CI-CD                                   │
│                                                                          │
│   GitHub Actions ──► Drift Detection ──► Triggered Retraining           │
│                      (Evidently AI)       (Sample Decay Weighting)       │
│                                                                          │
│   Model Validation Gates ──► Champion/Challenger Promotion               │
└──────────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                       LIVE INFERENCE PIPELINE                            │
│                                                                          │
│   Kafka Stream Simulator ──► FastAPI Service ──► WebSocket Push         │
│   (replays historical data)   (feature lookup     (React/Vue            │
│                                + model predict)    Dashboard)            │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer              | Tool                              | Purpose                                            |
| ------------------ | --------------------------------- | -------------------------------------------------- |
| Data Source         | FastF1                            | Historical F1 telemetry, lap times, weather         |
| Orchestration      | Dagster                           | Pipeline scheduling, dependency management          |
| Storage            | Parquet on S3 / local filesystem  | Columnar storage for telemetry data                 |
| Feature Store      | Feast                             | Offline/online feature serving with point-in-time joins |
| Experiment Tracking| MLflow                            | Metrics logging, model versioning, registry         |
| Baseline Model     | XGBoost / LightGBM               | Gradient-boosted trees for non-linear thresholds    |
| Advanced Model     | PyTorch (LSTM / TFT)             | Sequence modeling for degradation trajectories      |
| Drift Detection    | Evidently AI                      | Data and model drift monitoring                     |
| CI/CD              | GitHub Actions                    | Automated testing, retraining triggers              |
| Streaming          | Apache Kafka (or Redis Streams)   | Simulated live telemetry ingestion                  |
| Inference API      | FastAPI                           | Low-latency model serving with WebSocket support    |
| Dashboard          | React + Recharts (or Streamlit)   | Real-time visualization of predictions              |
| Containerization   | Docker / Docker Compose           | Reproducible environments for all services          |

---

## Phase 1 — Data Acquisition & Feature Store

**Objective:** Build the data foundation — ingest, clean, and serve ML-ready features from raw F1 telemetry.

### Step 1.1 — Raw Data Ingestion Pipeline

Build a Dagster pipeline that pulls historical race data from FastF1 and writes it to structured Parquet files.

**Tasks:**

- [x] Set up Python environment with `fastf1`, `dagster`, `pandas`, `pyarrow`
- [x] Write Dagster assets (multi-partitioned by season + round) that download FastF1 data into Parquet:
  - `raw_session_metadata` — event/session metadata (name, date, location, session type, format)
  - `raw_lap_data` — lap-by-lap timing (sectors, pit in/out); includes compound, tyre life, and stint columns per lap
  - `raw_telemetry_data` — car telemetry (~4 Hz: speed, RPM, gear, throttle, brake, DRS, etc.), per session and driver
  - `raw_weather_data` — session weather (air/track temp, humidity, rainfall, wind)
- [x] Implement incremental loading: track which (season, round) pairs have been ingested; skip re-downloading
- [ ] Backfill seasons 2021–2025 to build a training corpus (partition grid is defined in code; materialize each pair via the job or UI)
- [x] Store raw data in `data/raw/{season}/{round}/` as Parquet files, partitioned by session type

**Key Files:**

```
src/ingestion/
├── assets.py              # Dagster assets for each data domain
├── resources.py           # FastF1 session loader, S3 client configs
├── schedules.py           # Cron schedule for post-race ingestion
└── partitions.py          # Season/round partition definitions
```

### Step 1.2 — Feature Engineering

Transform raw telemetry into features that capture degradation signals.

**Tasks:**

- [x] **Stint-Level Aggregation:** Group laps by stint (pit-to-pit). Calculate stint-normalized lap number (lap 1 of stint, lap 2, etc.)
- [x] **Rolling Statistics:** For each driver's stint, compute rolling mean and rolling variance of lap times over windows of 3 and 5 laps. Rising variance is an early cliff signal.
- [x] **Sector Deltas:** Compute per-sector time deltas from stint-best. Rear-limited degradation shows up in slow corners (Sector 3 at Barcelona, Sector 2 at Silverstone).
- [x] **Tire Age Features:** Current tire age in laps, total distance on compound, fuel-corrected lap time (subtract ~0.06s per lap of fuel burn).
- [x] **Environmental Features:** Track temperature at lap start, track evolution index (how much rubber is laid down — compute from session-wide lap time improvement).
- [x] **Traffic/Dirty Air Features:** Gap to car ahead (seconds). If gap < 1.5s, flag as "dirty air exposure." Cumulative dirty-air laps in current stint.
- [x] **Track-Specific Static Features:** Manually curate a reference table of circuit characteristics — number of high-speed corners, surface abrasiveness rating, altitude, typical tire-limited nature (front vs. rear).

**Key Output Features:**

| Feature                       | Type    | Update Frequency |
| ----------------------------- | ------- | ---------------- |
| `tire_age_laps`               | Dynamic | Per lap          |
| `rolling_mean_laptime_5`      | Dynamic | Per lap          |
| `rolling_var_laptime_5`       | Dynamic | Per lap          |
| `sector3_delta_from_best`     | Dynamic | Per lap          |
| `fuel_corrected_laptime`      | Dynamic | Per lap          |
| `track_temp_c`                | Dynamic | Per lap          |
| `dirty_air_cumulative_laps`   | Dynamic | Per lap          |
| `compound`                    | Static  | Per stint         |
| `circuit_high_speed_corners`  | Static  | Per circuit       |
| `circuit_abrasiveness`        | Static  | Per circuit       |

### Step 1.3 — Feature Store Setup (Feast)

Register features in Feast for reproducible training and low-latency online serving.

**Tasks:**

- [x] Initialize a Feast feature repository in `feature_repo/`
- [x] Define **Entities**: `session`, `driver`, `stint` (composite join keys: `session_id + driver_number + stint_number`); `circuit` (join key: `location`)
- [x] Define **Feature Views**:
  - `stint_telemetry_features` — dynamic per-lap features sourced from the offline Parquet store
  - `circuit_features` — static track characteristics sourced from a reference CSV
  - `weather_features` — per-lap environmental data
- [x] Define **Feature Services** that bundle the views needed for training vs. inference
- [x] Configure an **offline store** (file-based) for `get_historical_features()` calls during training
- [x] Configure an **online store** (SQLite for local dev, Redis for production) for real-time feature retrieval
- [x] Write integration tests: materialize features, retrieve them, assert schema and value ranges

**Key Files:**

```
feature_repo/
├── feature_store.yaml         # Offline (file) + online (SQLite) store config
├── entities.py                # session, driver, stint, circuit entities
├── feature_views.py           # stint_telemetry_features, weather_features, circuit_features
└── feature_services.py        # training_feature_service, inference_feature_service

src/features/
├── feast_prep.py              # Consolidate processed parquets into Feast data sources
├── engineering.py             # (updated) adds Feast entity keys & event_timestamp
└── constants.py               # (updated) FEAST_DATA_DIR, SESSION_HOUR_OFFSETS

tests/
├── conftest.py                # Shared fixtures (project_root, feast paths)
└── test_feast.py              # Integration tests for apply/materialize/retrieve
```

**Quickstart:**

```bash
# 1. Prepare Feast data sources (consolidate processed features)
python -m src.features.feast_prep

# 2. Apply Feast definitions (creates registry + SQLite online store)
feast -c feature_repo apply

# 3. Materialize features into the online store
feast -c feature_repo materialize 2025-03-14T00:00:00 2025-03-17T00:00:00

# 4. Run integration tests
pytest tests/test_feast.py -v
```

### Step 1.4 — Target Variable Construction

- [x] For each stint in the historical data, identify the "cliff lap" — the first lap where lap time exceeds `stint_average + 1.5s` (excluding laps with yellow flags, pit-in laps, or safety car periods)
- [x] Compute target: `laps_to_cliff = cliff_lap - current_lap`. If no cliff occurred (stint ended by strategic pit), label as censored and handle via survival analysis framing or clip to stint length.
- [x] Build a labeled training dataset by joining features and targets via point-in-time join through Feast

**Key Files:**

```
src/features/
├── target.py                 # Cliff detection and laps_to_cliff computation
├── dataset.py                # Feast point-in-time join, labeled dataset builder
├── quality.py                # Data quality checks and feature distribution report
└── constants.py              # (updated) CLIFF_THRESHOLD_S, MIN_STINT_LAPS, TRAINING_DATA_DIR

tests/
└── test_target.py            # Unit tests for cliff detection and target construction
```

**Quickstart:**

```bash
# 1. Compute targets and build labeled dataset (with Feast)
python -m src.features.dataset

# 1b. Or without Feast (directly from consolidated features)
python -m src.features.dataset --no-feast

# 2. Run data quality checks
python -m src.features.quality

# 3. Run unit tests
pytest tests/test_target.py -v
```

**Phase 1 Deliverables:**
- Dagster pipeline that ingests 4+ seasons of data on a schedule
- Feast feature store with offline and online stores configured
- Labeled training dataset with ~50k+ stint-lap observations
- Data quality checks and documentation of feature distributions

---

## Phase 2 — Modeling the Tire Cliff

**Objective:** Train, evaluate, and iterate on models that predict the tire cliff, validated against real-world strategic decisions.

### Step 2.1 — Experiment Infrastructure

**Tasks:**

- [ ] Set up MLflow tracking server (local or remote) with a PostgreSQL backend and S3 artifact store
- [ ] Create a base training script that:
  - Pulls features from the Feast offline store via `get_historical_features()`
  - Splits data by time (train on 2021–2023, validate on 2024 early-season, test on 2024 late-season)
  - Logs all hyperparameters, metrics, and artifacts to MLflow
- [ ] Define evaluation metrics:
  - **MAE** of predicted `laps_to_cliff` vs. actual
  - **Precision@3**: is the predicted cliff lap within 3 laps of the actual?
  - **Strategy Accuracy**: would the model's recommendation have beaten the team's actual strategy?
- [ ] Set up MLflow model registry with `Staging` and `Production` stages

### Step 2.2 — Baseline Model (XGBoost / LightGBM)

Tree-based models excel at discovering non-linear interaction thresholds — exactly the pattern tire cliffs exhibit.

**Tasks:**

- [ ] Train an XGBoost regressor on the tabular feature set, predicting `laps_to_cliff`
- [ ] Perform hyperparameter tuning via Optuna (n_estimators, max_depth, learning_rate, min_child_weight)
- [ ] Analyze feature importance: expect `tire_age_laps`, `rolling_var_laptime_5`, `track_temp_c`, and `compound` to dominate
- [ ] Generate SHAP explanations for individual predictions (e.g., "Why does the model think Norris's softs will cliff on lap 22?")
- [ ] Log the best model to MLflow, register it, and promote to `Staging`

**Expected Baseline Performance:**

| Metric        | Target   |
| ------------- | -------- |
| MAE           | < 4 laps |
| Precision@3   | > 55%    |

### Step 2.3 — Sequential Model (LSTM / Temporal Fusion Transformer)

Lap times are a time series within each stint. Sequential models can learn the *shape* of the degradation curve.

**Tasks:**

- [ ] Prepare sequence data: for each stint, create a sliding window of the last N laps' features as input, with `laps_to_cliff` as the target
- [ ] Implement an LSTM-based model in PyTorch:
  - Input: sequence of per-lap feature vectors (length 5–10)
  - Architecture: 2-layer LSTM → fully connected head → scalar output
  - Loss: Huber loss (robust to outliers from safety car periods)
- [ ] Implement a Temporal Fusion Transformer (TFT) variant using `pytorch-forecasting`:
  - Leverage its built-in handling of static covariates (circuit, compound) alongside temporal inputs
  - Use its interpretable attention to identify which past laps influence the prediction most
- [ ] Compare LSTM and TFT against the XGBoost baseline on the same test split
- [ ] Ensemble: weighted average of XGBoost + best sequential model

### Step 2.4 — The Williams Midfield Validation

Standard evaluation on front-runners in clean air is misleading. The real test is the chaotic midfield.

**Tasks:**

- [ ] Filter the test set to only midfield constructors (Williams, Haas, Alpine, RB, Sauber)
- [ ] Evaluate model accuracy specifically for:
  - Stints with significant dirty-air exposure (>5 cumulative laps within 1.5s of car ahead)
  - Stints on non-optimal tire strategies (e.g., Hard-Medium instead of Medium-Hard)
  - Races with variable weather (drying track, late rain threat)
- [ ] Perform case-study analysis: pick 3–5 real midfield battles and show whether the model would have recommended a better pit window than the team actually chose
- [ ] Document findings and failure modes in a model card

**Phase 2 Deliverables:**
- MLflow experiment with 20+ logged runs and clear champion model
- Trained XGBoost and LSTM/TFT models registered in MLflow
- SHAP analysis notebook
- Williams Midfield Validation report
- Model card documenting performance, limitations, and intended use

---

## Phase 3 — Concept Drift & MLOps

**Objective:** Build the automated infrastructure to detect when the model is going stale and retrain it — the hallmark of a production ML system.

### Step 3.1 — Drift Detection

**Tasks:**

- [ ] Integrate **Evidently AI** to monitor:
  - **Data drift**: are incoming feature distributions (e.g., `rolling_var_laptime_5`) shifting from the training distribution? Use the Kolmogorov-Smirnov test per feature.
  - **Prediction drift**: is the distribution of predicted `laps_to_cliff` changing significantly?
  - **Model performance drift**: when actuals become available post-race, compute rolling MAE and compare to the baseline.
- [ ] Generate Evidently reports as HTML artifacts, stored alongside MLflow runs
- [ ] Define alert thresholds:
  - Data drift: >30% of features drifting triggers a warning
  - Performance drift: MAE increases by >1.5 laps from baseline triggers retraining

### Step 3.2 — Sample Weighting with Temporal Decay

Car upgrades make older data less representative. Recent data should matter more.

**Tasks:**

- [ ] Implement an exponential decay weighting function:
  ```
  weight(race) = exp(-λ * days_since_race)
  ```
  where λ is tuned to give the most recent race ~1.0 weight and races from 6 months ago ~0.3 weight
- [ ] Pass sample weights to XGBoost via `sample_weight` parameter and to PyTorch via weighted loss
- [ ] Experiment with team-specific decay: if Team X brings a major upgrade, increase decay rate for Team X's pre-upgrade data only (detected via spike in feature drift for that team)
- [ ] Log decay parameters as MLflow hyperparameters for reproducibility

### Step 3.3 — Automated Retraining Pipeline

**Tasks:**

- [ ] Build a Dagster job `retrain_champion_model` that:
  1. Pulls the latest features from the Feast offline store
  2. Applies sample weights with temporal decay
  3. Trains the champion model architecture with the current best hyperparameters
  4. Evaluates on a holdout set (most recent 2 race weekends)
  5. Compares against the current production model
  6. Promotes to production only if performance improves (champion/challenger pattern)
  7. Logs everything to MLflow
- [ ] Create a Dagster sensor that triggers retraining when:
  - A new race weekend's data has been ingested
  - Drift detection flags are raised
- [ ] Build GitHub Actions workflow:
  - On push to `main`: run unit tests, lint, type-check
  - On new data tag (`data/v2024-R05`): trigger retraining pipeline
  - Post-retraining: run validation suite, generate model card, create PR with updated model metadata

### Step 3.4 — Model Validation Gates

**Tasks:**

- [ ] Define a validation suite that any candidate model must pass before promotion:
  - MAE on full test set < current production model's MAE
  - MAE on midfield subset < threshold
  - No single circuit has MAE > 6 laps (guards against catastrophic failure on specific tracks)
  - Prediction latency < 50ms at p99
- [ ] Implement as a Dagster asset that produces a pass/fail report
- [ ] Block promotion in the CI pipeline if validation fails

**Phase 3 Deliverables:**
- Evidently drift monitoring dashboard
- Automated retraining pipeline with champion/challenger promotion
- GitHub Actions CI/CD workflows for testing, retraining, and deployment
- Validation gate suite with documented thresholds

---

## Phase 4 — Live Inference Pipeline

**Objective:** Simulate a race weekend in real time — streaming telemetry into the model and visualizing predictions on a dashboard.

### Step 4.1 — Stream Simulation with Kafka

Since live FIA telemetry isn't publicly accessible in real time, simulate it by replaying historical data at real-world pace.

**Tasks:**

- [ ] Set up a Kafka cluster (single-broker, via Docker Compose)
- [ ] Write a **Kafka producer** that:
  - Takes a historical session (e.g., 2024 British GP Race)
  - Reads the telemetry Parquet files
  - Publishes lap-completion events to a `lap_completed` topic at the cadence they originally occurred (e.g., one event every ~90 seconds)
  - Each message contains: `session_id`, `driver_number`, `lap_number`, `lap_time`, `sector_times`, `compound`, `tire_age`, `track_temp`, `gap_ahead`
- [ ] Write a configurable speed multiplier (1x = real time, 10x = fast-forward for testing)
- [ ] Include a `race_control` topic for safety car, red flag, and VSC events

### Step 4.2 — FastAPI Inference Service

**Tasks:**

- [ ] Build a FastAPI application with these endpoints:
  - `POST /predict` — given a lap event, return predicted `laps_to_cliff` and confidence interval
  - `GET /race/{session_id}/state` — return current race state (all drivers' latest predictions, stint info)
  - `WebSocket /ws/race/{session_id}` — push predictions to connected clients in real time
- [ ] On each incoming lap event:
  1. Compute dynamic features (rolling stats, dirty-air flags) from the in-memory race state
  2. Pull static features from the Feast online store
  3. Run inference through the production model (loaded from MLflow registry)
  4. Cache the prediction and update the race state
- [ ] Implement model warm-up on startup (load model, run dummy prediction to JIT-compile)
- [ ] Add structured logging (JSON) and Prometheus metrics (prediction latency histogram, request count)

### Step 4.3 — Real-Time Dashboard

**Tasks:**

- [ ] **Option A — Streamlit (MVP):**
  - Build a Streamlit app that connects to the FastAPI WebSocket
  - Display a live leaderboard with columns: Position, Driver, Compound, Tire Age, Predicted Laps to Cliff, Risk Level (Green/Yellow/Red)
  - Plot a per-driver degradation curve (actual lap times + predicted cliff overlay)

- [ ] **Option B — React Dashboard (Production):**
  - Scaffold a React app with Vite
  - Use Recharts or Visx for real-time charting
  - Components:
    - `RaceLeaderboard` — sortable table with live cliff predictions
    - `DegradationChart` — per-driver line chart of lap times with cliff prediction overlay
    - `StrategyTimeline` — horizontal Gantt-style chart showing each driver's stint history and predicted remaining stint life
    - `DriftMonitor` — small panel showing model confidence and any active drift alerts
  - Connect to FastAPI via WebSocket for push updates

### Step 4.4 — End-to-End Integration Test

**Tasks:**

- [ ] Write a Docker Compose file that spins up:
  - Kafka (+ Zookeeper or KRaft)
  - FastAPI inference service
  - Feast online store (Redis)
  - Dashboard (Streamlit or React dev server)
  - Kafka producer simulator
- [ ] Run a full simulated race (e.g., 2024 Spanish GP):
  - Producer replays all 66 laps at 10x speed
  - Inference service processes events and pushes to dashboard
  - Verify predictions update in real time on the dashboard
- [ ] Record a demo video of the full pipeline running

**Phase 4 Deliverables:**
- Kafka-based stream simulator
- FastAPI inference service with WebSocket support
- Real-time dashboard (Streamlit MVP or React production)
- Docker Compose for one-command local deployment
- Demo video of a simulated race

---

## Project Structure

```
f1-telemetry-mlops/
│
├── README.md
├── pyproject.toml                    # Python project config (dependencies, linting)
├── Makefile                          # Common commands (setup, test, train, serve)
├── docker-compose.yml                # Full-stack local deployment
├── .github/
│   └── workflows/
│       ├── ci.yml                    # Lint, type-check, unit tests on push
│       ├── retrain.yml               # Triggered retraining pipeline
│       └── deploy.yml                # Model deployment workflow
│
├── src/
│   ├── ingestion/                    # Phase 1: Dagster data pipeline
│   │   ├── assets.py
│   │   ├── resources.py
│   │   ├── schedules.py
│   │   ├── partitions.py
│   │   └── __init__.py
│   │
│   ├── features/                     # Phase 1: Feature engineering
│   │   ├── engineering.py            # Feature transformation logic
│   │   ├── target.py                 # Cliff detection and target variable construction
│   │   ├── dataset.py               # Labeled training dataset builder (Feast PIT join)
│   │   ├── quality.py               # Data quality checks and distribution report
│   │   ├── feast_prep.py            # Consolidate features into Feast data sources
│   │   ├── constants.py             # Feature engineering constants and thresholds
│   │   └── __init__.py
│   │
│   ├── training/                     # Phase 2: Model training
│   │   ├── baseline.py               # XGBoost / LightGBM training
│   │   ├── sequential.py             # LSTM / TFT training
│   │   ├── evaluation.py             # Metrics, SHAP, validation suite
│   │   ├── hyperopt.py               # Optuna hyperparameter search
│   │   └── __init__.py
│   │
│   ├── drift/                        # Phase 3: Drift detection
│   │   ├── detection.py              # Evidently AI integration
│   │   ├── decay.py                  # Temporal sample weighting
│   │   └── __init__.py
│   │
│   ├── serving/                      # Phase 4: Inference service
│   │   ├── app.py                    # FastAPI application
│   │   ├── race_state.py             # In-memory race state manager
│   │   ├── predictor.py              # Model loading and inference
│   │   └── __init__.py
│   │
│   └── streaming/                    # Phase 4: Kafka simulation
│       ├── producer.py               # Historical data replay producer
│       ├── consumer.py               # Lap event consumer
│       └── __init__.py
│
├── feature_repo/                     # Feast feature store definitions
│   ├── feature_store.yaml
│   ├── entities.py
│   ├── feature_views.py
│   └── feature_services.py
│
├── notebooks/                        # Exploration and analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   ├── 03_shap_analysis.ipynb
│   └── 04_midfield_validation.ipynb
│
├── dashboard/                        # React dashboard (Phase 4)
│   ├── package.json
│   ├── src/
│   │   ├── components/
│   │   │   ├── RaceLeaderboard.tsx
│   │   │   ├── DegradationChart.tsx
│   │   │   ├── StrategyTimeline.tsx
│   │   │   └── DriftMonitor.tsx
│   │   └── App.tsx
│   └── ...
│
├── data/
│   ├── raw/                          # Raw Parquet files from FastF1
│   ├── processed/                    # Engineered features
│   ├── feast/                        # Consolidated Feast data sources, registry, online store
│   ├── training/                     # Labeled dataset, quality reports, dataset summary
│   └── reference/                    # Static circuit characteristics CSV
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_features.py
│   ├── test_target.py                # Cliff detection and target variable tests
│   ├── test_feast.py                 # Feast feature store integration tests
│   ├── test_training.py
│   ├── test_serving.py
│   └── conftest.py
│
├── configs/
│   ├── training.yaml                 # Model hyperparameters
│   ├── decay.yaml                    # Temporal decay parameters
│   └── drift_thresholds.yaml         # Drift alert thresholds
│
└── docs/
    ├── model_card.md                 # Model documentation
    └── feature_dictionary.md         # Feature definitions and sources
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Node.js 20+ (for React dashboard, Phase 4)

### Local Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/f1-telemetry-mlops.git
cd f1-telemetry-mlops

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -e ".[dev]"

# Prepare Feast data sources & initialize the feature store
python -m src.features.feast_prep
feast -c feature_repo apply

# Start MLflow tracking server
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --port 5000

# Run the ingestion job for one weekend partition (round | season, e.g. round 5, 2024)
dagster job execute -m src.ingestion -a defs -j ingest_race_weekend --tags '{"dagster/partition": "5|2024"}'
```

### Run the Full Stack (Phase 4)

```bash
docker-compose up --build
```

This starts Kafka, Redis (Feast online store), the FastAPI service, and the dashboard. Then trigger the simulator:

```bash
python -m src.streaming.producer --session "2024 British Grand Prix Race" --speed 10
```

---

## Development Workflow

| Action                    | Command                                  |
| ------------------------- | ---------------------------------------- |
| Run tests                 | `pytest tests/ -v`                       |
| Lint & format             | `ruff check . && ruff format .`          |
| Type check                | `mypy src/`                              |
| Train baseline model      | `python -m src.training.baseline`        |
| Train sequential model    | `python -m src.training.sequential`      |
| Generate drift report     | `python -m src.drift.detection --report` |
| Start API server (dev)    | `uvicorn src.serving.app:app --reload`   |
| Materialize features      | `cd feature_repo && feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")` |

---

## License

This project is for educational and portfolio purposes. F1 telemetry data is sourced from the [FastF1](https://github.com/theOehrly/Fast-F1) library, which provides access to publicly available FIA timing data.
