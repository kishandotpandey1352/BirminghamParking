from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

SOURCE_URL = "https://archive.ics.uci.edu/dataset/482/parking+birmingham"
REQUIRED_COLUMNS = ["SystemCodeNumber", "Capacity", "Occupancy", "LastUpdated"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_clean_data(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    raw = pd.read_csv(path)
    missing_columns = set(REQUIRED_COLUMNS) - set(raw.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    frame = raw[REQUIRED_COLUMNS].copy()
    counts = {"raw_rows": len(frame), "duplicate_rows": int(frame.duplicated().sum())}
    frame["LastUpdated"] = pd.to_datetime(frame["LastUpdated"], errors="coerce")
    frame["LastUpdated"] = frame["LastUpdated"].dt.tz_localize("Europe/London", ambiguous="NaT", nonexistent="NaT")
    for column in ["Capacity", "Occupancy"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    counts["invalid_timestamp"] = int(frame["LastUpdated"].isna().sum())
    counts["invalid_capacity"] = int((frame["Capacity"] <= 0).fillna(True).sum())
    counts["negative_occupancy"] = int((frame["Occupancy"] < 0).fillna(True).sum())
    counts["over_capacity"] = int((frame["Occupancy"] > frame["Capacity"]).fillna(True).sum())
    frame = frame.dropna(subset=REQUIRED_COLUMNS).drop_duplicates()
    frame = frame[(frame["Capacity"] > 0) & (frame["Occupancy"] >= 0)]
    frame = frame[frame["Occupancy"] <= frame["Capacity"]].copy()
    frame = frame.sort_values(["LastUpdated", "SystemCodeNumber"]).reset_index(drop=True)
    counts["clean_rows"] = len(frame)
    counts["excluded_rows"] = counts["raw_rows"] - counts["clean_rows"]
    return frame, counts


def exact_targets(frame: pd.DataFrame, horizon_minutes: int = 30) -> pd.DataFrame:
    left = frame.rename(columns={"Occupancy": "occupancy_now"}).copy()
    left["target_time"] = left["LastUpdated"] + pd.Timedelta(minutes=horizon_minutes)
    right = frame[["SystemCodeNumber", "LastUpdated", "Occupancy"]].rename(
        columns={"LastUpdated": "target_time", "Occupancy": "occupancy_future"}
    )
    result = left.merge(right, on=["SystemCodeNumber", "target_time"], how="left", validate="many_to_one")
    result["horizon_minutes"] = horizon_minutes
    result["available_future"] = result["Capacity"] - result["occupancy_future"]
    return result
