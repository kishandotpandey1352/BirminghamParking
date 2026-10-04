from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, min_samples: int = 20, threshold: float = 0.15) -> dict:
    if len(current) < min_samples or len(reference) < min_samples:
        return {"status": "insufficient_data", "threshold": threshold, "sample_count": len(current), "features": {}}
    features = {}
    for column in ["occupancy_now", "Capacity", "forecast_hour", "forecast_weekday"]:
        scale = max(float(reference[column].std()), 1.0)
        distance = float(wasserstein_distance(reference[column], current[column]) / scale)
        features[column] = {"normalised_wasserstein": distance, "alert": distance > threshold}
    for column in ["SystemCodeNumber"]:
        ref = reference[column].value_counts(normalize=True)
        cur = current[column].value_counts(normalize=True)
        keys = set(ref.index) | set(cur.index)
        tv = 0.5 * sum(abs(ref.get(key, 0) - cur.get(key, 0)) for key in keys)
        features[column] = {"total_variation": float(tv), "alert": tv > threshold}
    alerts = [name for name, values in features.items() if values["alert"]]
    return {"status": "warning" if alerts else "healthy", "threshold": threshold, "sample_count": len(current), "features": features, "alerts": alerts}


def replay_report(examples: pd.DataFrame, model, mode: str = "baseline", min_samples: int = 20) -> dict:
    ordered = examples.sort_values("LastUpdated").copy()
    if mode == "data_drift":
        ordered = ordered.sort_values("occupancy_now", ascending=False).head(max(min_samples, len(ordered) // 2))
    features = ordered.copy()
    if mode == "concept_drift":
        counterfactual = np.clip(features["occupancy_future"] + 0.15 * features["Capacity"], 0, features["Capacity"])
    else:
        counterfactual = features["occupancy_future"]
    predicted = model.predict(features[["occupancy_now", "Capacity", "SystemCodeNumber", "forecast_hour", "forecast_weekday"]])
    reference = features.head(max(min_samples, len(features) // 2))
    current = features.tail(max(min_samples, len(features) // 2))
    errors_available = features["target_time"] <= ordered["LastUpdated"].max()
    errors = np.abs(predicted - counterfactual) if errors_available.any() else np.array([])
    result = {
        "mode": mode, "simulation": mode != "baseline", "sample_count": len(features),
        "labelled_samples": int(errors_available.sum()), "mae": float(errors.mean()) if len(errors) else None,
        "baseline_mae": float(np.abs(features.loc[errors_available, "occupancy_now"] - counterfactual[errors_available]).mean()) if errors_available.any() else None,
        "initial_window": {"status": "insufficient_data", "labelled_samples": 0, "mae": None, "baseline_mae": None, "reason": "Outcomes are not available until target timestamps arrive."},
        "drift": drift_report(reference, current, min_samples=min_samples),
        "concept_drift_inputs_unchanged": bool(mode != "concept_drift" or features[["occupancy_now", "Capacity", "SystemCodeNumber"]].equals(ordered[["occupancy_now", "Capacity", "SystemCodeNumber"]])),
        "note": "Synthetic teaching scenario; not a discovered Birmingham event." if mode != "baseline" else "Historical held-out replay.",
    }
    return result
