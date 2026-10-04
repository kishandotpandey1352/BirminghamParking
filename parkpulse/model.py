from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURES = ["occupancy_now", "Capacity", "SystemCodeNumber", "forecast_hour", "forecast_weekday"]
NUMERIC = ["occupancy_now", "Capacity", "forecast_hour", "forecast_weekday"]
CATEGORICAL = ["SystemCodeNumber"]


def add_features(examples: pd.DataFrame) -> pd.DataFrame:
    result = examples.copy()
    result["forecast_hour"] = result["target_time"].dt.hour
    result["forecast_weekday"] = result["target_time"].dt.dayofweek
    return result


def chronological_split(examples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    times = examples["LastUpdated"].sort_values()
    first, second = times.quantile([0.6, 0.8])
    train = examples[(examples.LastUpdated < first) & (examples.target_time <= first)].copy()
    validation = examples[(examples.LastUpdated >= first) & (examples.LastUpdated < second) & (examples.target_time <= second)].copy()
    test = examples[examples.LastUpdated >= second].copy()
    return train, validation, test, {"train_end": first.isoformat(), "validation_end": second.isoformat()}


def metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    errors = np.asarray(predicted) - actual.to_numpy()
    return {"mae": float(np.abs(errors).mean()), "rmse": float(np.sqrt(np.square(errors).mean()))}


def evaluate(model: Pipeline, examples: pd.DataFrame) -> dict:
    features = add_features(examples)
    predicted = model.predict(features[FEATURES])
    persistence = features["occupancy_now"].to_numpy()
    output = {"model": metrics(features.occupancy_future, predicted), "persistence": metrics(features.occupancy_future, persistence), "samples": int(len(features))}
    output["clipping_frequency"] = float(np.mean((predicted < 0) | (predicted > features.Capacity.to_numpy())))
    output["per_car_park"] = {}
    for park, group in features.assign(predicted=predicted).groupby("SystemCodeNumber"):
        output["per_car_park"][park] = {"samples": int(len(group)), **metrics(group.occupancy_future, group.predicted.to_numpy())}
    return output


def train_model(data_path: Path, artifact_dir: Path, version: str = "real-2016-01") -> dict:
    from .data import SOURCE_URL, exact_targets, load_clean_data, sha256_file

    frame, cleaning = load_clean_data(data_path)
    all_examples = exact_targets(frame)
    examples = all_examples.dropna(subset=["occupancy_future"])
    train, validation, test, boundaries = chronological_split(examples)
    preprocessor = ColumnTransformer([("numeric", "passthrough", NUMERIC), ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", LinearRegression())])
    train_features = add_features(train)
    pipeline.fit(train_features[FEATURES], train_features["occupancy_future"])
    evaluation = {"validation": evaluate(pipeline, validation), "test": evaluate(pipeline, test)}
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, artifact_dir / "model.joblib")
    manifest = {
        "model_version": version, "test_only": False, "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_schema": FEATURES, "dataset_sha256": sha256_file(data_path), "source_url": SOURCE_URL,
        "cleaning": cleaning, "split_boundaries": boundaries, "exact_target_coverage": float(len(examples) / len(all_examples)),
        "metrics": evaluation, "supported_date_range": [frame.LastUpdated.min().isoformat(), frame.LastUpdated.max().isoformat()],
        "package_versions": {"python": platform.python_version(), "pandas": pd.__version__},
        "source_commit": _git_commit(),
    }
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
