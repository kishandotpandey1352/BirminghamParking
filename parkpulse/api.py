from __future__ import annotations

import json
import os
import time
from pathlib import Path

import joblib
import pandas as pd
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
    car_park_id: str = Field(min_length=1)
    observed_at: str
    occupancy: float = Field(ge=0)


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
    examples = EXAMPLES.sort_values("LastUpdated").head(100).tail(20)
    return {"car_parks": sorted(FRAME.SystemCodeNumber.unique().tolist()), "examples": [{"car_park_id": r.SystemCodeNumber, "observed_at": r.LastUpdated.isoformat(), "occupancy": r.occupancy_now, "capacity": r.Capacity} for r in examples.itertuples()]}


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
        if request.occupancy > capacity:
            raise HTTPException(422, "Occupancy cannot exceed capacity")
        observed = pd.Timestamp(request.observed_at)
        if observed.tzinfo is None:
            observed = observed.tz_localize("Europe/London")
        row = pd.DataFrame([{ "occupancy_now": request.occupancy, "Capacity": capacity, "SystemCodeNumber": request.car_park_id, "LastUpdated": observed, "target_time": observed + pd.Timedelta(minutes=30)}])
        prepared = add_features(row)
        raw = float(MODEL.predict(prepared[FEATURES])[0])
        bounded = max(0.0, min(capacity, raw))
        return {"car_park_id": request.car_park_id, "observation_timestamp": observed.isoformat(), "forecast_timestamp": row.target_time.iloc[0].isoformat(), "raw_occupancy_forecast": raw, "bounded_occupancy_forecast": bounded, "predicted_available_spaces": capacity - bounded, "model_version": MANIFEST.get("model_version")}
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
