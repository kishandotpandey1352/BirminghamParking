import pandas as pd

from parkpulse.data import exact_targets
from parkpulse.model import add_features


def test_target_matching_does_not_cross_car_parks():
    frame = pd.DataFrame([["A", 10, 2, "2024-01-01 10:00"], ["B", 20, 9, "2024-01-01 10:00"], ["B", 20, 11, "2024-01-01 10:30"]], columns=["SystemCodeNumber", "Capacity", "Occupancy", "LastUpdated"])
    frame["LastUpdated"] = pd.to_datetime(frame.LastUpdated)
    result = exact_targets(frame)
    assert result.loc[result.SystemCodeNumber == "A", "occupancy_future"].isna().all()
    assert result.loc[result.SystemCodeNumber == "B", "occupancy_future"].iloc[0] == 11


def test_missing_reading_is_not_next_row_target():
    frame = pd.DataFrame([["A", 10, 2, "2024-01-01 10:00"], ["A", 10, 8, "2024-01-01 10:20"]], columns=["SystemCodeNumber", "Capacity", "Occupancy", "LastUpdated"])
    frame["LastUpdated"] = pd.to_datetime(frame.LastUpdated)
    assert exact_targets(frame).occupancy_future.isna().all()


def test_features_use_prediction_time_only():
    row = pd.DataFrame({"occupancy_now": [2], "Capacity": [10], "SystemCodeNumber": ["A"], "LastUpdated": pd.to_datetime(["2024-01-01 10:00"]), "target_time": pd.to_datetime(["2024-01-01 10:30"]), "occupancy_future": [8]})
    assert add_features(row).loc[0, "forecast_hour"] == 10
