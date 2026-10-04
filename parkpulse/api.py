from __future__ import annotations

import json
import os
import time
from datetime import date, time as clock_time
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
import pytz
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .data import exact_targets, load_clean_data
from .model import FEATURES, add_features, train_model
from .monitoring import replay_report

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path(os.getenv("PARKPULSE_DATA", ROOT / "parking_birmingham" / "dataset.csv"))
ARTIFACT_DIR = Path(os.getenv("PARKPULSE_ARTIFACTS", ROOT / "artifacts"))
app = FastAPI(title="ParkPulse", version="0.1.0")
MODEL = None
MANIFEST = {}
FRAME = pd.DataFrame()
EXAMPLES = pd.DataFrame()
REQUESTS = {"count": 0, "errors": 0, "latencies_ms": []}
FRONTEND_DIR = ROOT / "frontend" / "dist"


class PredictionRequest(BaseModel):
    mode: Literal["historical", "what_if"] = "historical"
    car_park_id: str = Field(min_length=1)
    observation_date: date
    observation_time: clock_time
    assumed_occupancy: float | None = Field(default=None, ge=0)


def local_timestamp(observation_date: date, observation_time: clock_time) -> pd.Timestamp:
    try:
        return pd.Timestamp(f"{observation_date.isoformat()} {observation_time.isoformat()}").tz_localize("Europe/London")
    except (pytz.NonExistentTimeError, pytz.AmbiguousTimeError) as error:
        raise HTTPException(422, "The selected local time is ambiguous or does not exist under Europe/London timezone rules") from error


def split_for(timestamp: pd.Timestamp, target_timestamp: pd.Timestamp) -> str:
    boundaries = MANIFEST.get("split_boundaries", {})
    train_end = pd.Timestamp(boundaries["train_end"]) if boundaries.get("train_end") else None
    validation_end = pd.Timestamp(boundaries["validation_end"]) if boundaries.get("validation_end") else None
    if train_end is not None and timestamp < train_end and target_timestamp <= train_end:
        return "train"
    if train_end is not None and validation_end is not None and train_end <= timestamp < validation_end and target_timestamp <= validation_end:
        return "validation"
    return "test" if validation_end is not None and timestamp >= validation_end else "outside_evaluation_split"


def historical_observation(car_park_id: str, timestamp: pd.Timestamp) -> pd.Series:
    matches = FRAME[(FRAME["SystemCodeNumber"] == car_park_id) & (FRAME["LastUpdated"] == timestamp)]
    if matches.empty:
        raise HTTPException(422, "No recorded observation exists for that car park, date, and time")
    return matches.iloc[0]


def load_artifact(artifact_dir: Path | None = None) -> None:
    global MODEL, MANIFEST, FRAME, EXAMPLES
    artifact_dir = artifact_dir or ARTIFACT_DIR
    manifest_path = artifact_dir / "manifest.json"
    model_path = artifact_dir / "model.joblib"
    if manifest_path.exists() and model_path.exists() and DATA_PATH.exists():
        MANIFEST = json.loads(manifest_path.read_text(encoding="utf-8"))
        MODEL = joblib.load(model_path)
        FRAME, _ = load_clean_data(DATA_PATH)
        EXAMPLES = exact_targets(FRAME).dropna(subset=["occupancy_future"])


@app.on_event("startup")
def startup() -> None:
    current = ARTIFACT_DIR / "current.json"
    if current.exists():
        load_artifact(ARTIFACT_DIR / json.loads(current.read_text(encoding="utf-8"))["path"])
    else:
        load_artifact()


@app.get("/", response_class=FileResponse)
def index() -> str:
    return str(FRONTEND_DIR / "index.html" if (FRONTEND_DIR / "index.html").exists() else ROOT / "frontend" / "index.html")


if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": MODEL is not None, "model_version": MANIFEST.get("model_version")}


@app.get("/api/model")
def model_metadata() -> dict:
    return MANIFEST


@app.get("/api/models")
def models() -> dict:
    versions = []
    for manifest_path in sorted((ARTIFACT_DIR / "versions").glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        versions.append({"model_version": manifest.get("model_version"), "trained_at": manifest.get("trained_at"), "path": str(manifest_path.parent.relative_to(ARTIFACT_DIR))})
    if MANIFEST:
        versions.insert(0, {"model_version": MANIFEST.get("model_version"), "trained_at": MANIFEST.get("trained_at"), "path": "current"})
    return {"current": MANIFEST.get("model_version"), "versions": versions}


class TrainRequest(BaseModel):
    version: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{1,63}$")


@app.post("/api/models/train")
def train(request: TrainRequest) -> dict:
    if os.getenv("PARKPULSE_ENABLE_TRAINING", "0") != "1":
        raise HTTPException(403, "Training is disabled; set PARKPULSE_ENABLE_TRAINING=1 for local use")
    version_dir = ARTIFACT_DIR / "versions" / request.version
    if version_dir.exists():
        raise HTTPException(409, "That model version already exists")
    manifest = train_model(DATA_PATH, version_dir, version=request.version)
    (ARTIFACT_DIR / "current.json").write_text(json.dumps({"path": str(version_dir.relative_to(ARTIFACT_DIR))}), encoding="utf-8")
    load_artifact(version_dir)
    return {"status": "trained", "model_version": manifest["model_version"], "metrics": manifest["metrics"]}


@app.get("/api/options")
def options() -> dict:
    if FRAME.empty:
        raise HTTPException(503, "Model artifact is not loaded")
    test_examples = EXAMPLES[EXAMPLES["LastUpdated"] >= pd.Timestamp(MANIFEST.get("split_boundaries", {}).get("validation_end"))]
    default = test_examples.sort_values("LastUpdated").iloc[0] if not test_examples.empty else EXAMPLES.sort_values("LastUpdated").iloc[0]
    capacities = FRAME.groupby("SystemCodeNumber")["Capacity"].first().astype(int).to_dict()
    return {"car_parks": sorted(FRAME.SystemCodeNumber.unique().tolist()), "capacities": capacities, "default_historical": {"car_park_id": default.SystemCodeNumber, "observation_date": default.LastUpdated.date().isoformat(), "observation_time": default.LastUpdated.strftime("%H:%M:%S")}}


@app.get("/api/historical/options")
def historical_options(car_park_id: str, observation_date: date | None = None) -> dict:
    if FRAME.empty:
        raise HTTPException(503, "Model artifact is not loaded")
    park = FRAME[FRAME["SystemCodeNumber"] == car_park_id].sort_values("LastUpdated")
    if park.empty:
        raise HTTPException(422, "Unknown car park ID")
    dates = sorted({timestamp.date().isoformat() for timestamp in park["LastUpdated"]})
    observations = []
    if observation_date is not None:
        selected = park[park["LastUpdated"].dt.date == observation_date]
        for row in selected.itertuples():
            target = row.LastUpdated + pd.Timedelta(minutes=30)
            outcome = FRAME[(FRAME["SystemCodeNumber"] == car_park_id) & (FRAME["LastUpdated"] == target)]
            observations.append({"observation_time": row.LastUpdated.strftime("%H:%M:%S"), "occupancy": int(row.Occupancy), "capacity": int(row.Capacity), "current_available_spaces": int(row.Capacity - row.Occupancy), "split": split_for(row.LastUpdated, target), "outcome_available": not outcome.empty})
    return {"car_park_id": car_park_id, "available_dates": dates, "observations": observations}


@app.post("/api/predict")
def predict(request: PredictionRequest) -> dict:
    started = time.perf_counter()
    REQUESTS["count"] += 1
    try:
        if MODEL is None or FRAME.empty:
            raise HTTPException(503, "Model artifact is not loaded")
        matches = FRAME[FRAME.SystemCodeNumber == request.car_park_id]
        if matches.empty:
            raise HTTPException(422, "Unknown car park ID")
        capacity = float(matches.Capacity.iloc[0])
        observed = local_timestamp(request.observation_date, request.observation_time)
        if request.mode == "historical":
            source = historical_observation(request.car_park_id, observed)
            occupancy = float(source.Occupancy)
            target = observed + pd.Timedelta(minutes=30)
            outcome_available = not FRAME[(FRAME["SystemCodeNumber"] == request.car_park_id) & (FRAME["LastUpdated"] == target)].empty
            split = split_for(observed, target)
        else:
            scenario_start = pd.Timestamp("2017-01-01", tz="Europe/London")
            scenario_end = pd.Timestamp("2017-03-31 23:59:59", tz="Europe/London")
            if observed < scenario_start or observed > scenario_end:
                raise HTTPException(422, "What-if dates must be between 2017-01-01 and 2017-03-31")
            if request.assumed_occupancy is None:
                raise HTTPException(422, "What-if scenarios require an assumed current occupancy")
            occupancy = float(request.assumed_occupancy)
            outcome_available = False
            split = "scenario"
        if occupancy > capacity:
            raise HTTPException(422, "Occupancy cannot exceed capacity")
        row = pd.DataFrame([{ "occupancy_now": occupancy, "Capacity": capacity, "SystemCodeNumber": request.car_park_id, "LastUpdated": observed, "target_time": observed + pd.Timedelta(minutes=30)}])
        prepared = add_features(row)
        raw = float(MODEL.predict(prepared[FEATURES])[0])
        bounded = max(0.0, min(capacity, raw))
        return {"mode": request.mode, "car_park_id": request.car_park_id, "observation_timestamp": observed.isoformat(), "forecast_timestamp": row.target_time.iloc[0].isoformat(), "current_occupancy": occupancy, "capacity": capacity, "current_available_spaces": capacity - occupancy, "raw_occupancy_forecast": raw, "bounded_occupancy_forecast": bounded, "predicted_available_spaces": capacity - bounded, "model_version": MANIFEST.get("model_version"), "split": split, "outcome_available": outcome_available, "capacity_assumption": request.mode == "what_if"}
    except HTTPException:
        REQUESTS["errors"] += 1
        raise
    finally:
        REQUESTS["latencies_ms"].append((time.perf_counter() - started) * 1000)


@app.get("/api/replay/{mode}")
def replay(mode: str) -> dict:
    if MODEL is None or EXAMPLES.empty:
        raise HTTPException(503, "Model artifact is not loaded")
    if mode not in {"baseline", "data_drift", "concept_drift"}:
        raise HTTPException(400, "Unknown replay mode")
    report = replay_report(add_features(EXAMPLES), MODEL, mode=mode)
    report["data_quality"] = MANIFEST.get("cleaning", {})
    return report


@app.get("/api/operations")
def operations() -> dict:
    latencies = REQUESTS["latencies_ms"]
    return {"scope": "this process since startup", "request_count": REQUESTS["count"], "error_count": REQUESTS["errors"], "mean_latency_ms": sum(latencies) / len(latencies) if latencies else None}
