import pandas as pd
import pytest

from parkpulse import api


class FakeModel:
    def predict(self, features):
        assert "occupancy_future" not in features.columns
        return [42.0]


def setup_module():
    timestamp = pd.Timestamp("2016-12-01 10:00", tz="Europe/London")
    api.FRAME = pd.DataFrame([
        {"SystemCodeNumber": "PARK-1", "Capacity": 100, "Occupancy": 30, "LastUpdated": timestamp},
        {"SystemCodeNumber": "PARK-1", "Capacity": 100, "Occupancy": 35, "LastUpdated": timestamp + pd.Timedelta(minutes=30)},
    ])
    api.MANIFEST = {"model_version": "test", "split_boundaries": {"train_end": "2016-11-01T00:00:00+00:00", "validation_end": "2016-11-20T00:00:00+00:00"}}
    api.MODEL = FakeModel()


def test_historical_forecast_is_exactly_30_minutes_and_hides_future_occupancy():
    result = api.predict(api.PredictionRequest(mode="historical", car_park_id="PARK-1", observation_date="2016-12-01", observation_time="10:00:00"))
    assert result["forecast_timestamp"] == "2016-12-01T10:30:00+00:00"
    assert result["current_occupancy"] == 30
    assert result["outcome_available"] is True
    assert "future_occupancy" not in result


def test_missing_historical_observation_is_rejected_without_substitution():
    with pytest.raises(api.HTTPException) as error:
        api.predict(api.PredictionRequest(mode="historical", car_park_id="PARK-1", observation_date="2016-12-01", observation_time="10:15:00"))
    assert "No recorded observation" in error.value.detail


def test_what_if_requires_assumed_occupancy_and_stays_outside_historical_split():
    with pytest.raises(api.HTTPException) as error:
        api.predict(api.PredictionRequest(mode="what_if", car_park_id="PARK-1", observation_date="2017-01-01", observation_time="08:00:00"))
    assert "require an assumed" in error.value.detail
    result = api.predict(api.PredictionRequest(mode="what_if", car_park_id="PARK-1", observation_date="2017-01-01", observation_time="08:00:00", assumed_occupancy=40))
    assert result["split"] == "scenario"
    assert result["outcome_available"] is False
    assert result["capacity_assumption"] is True


def test_invalid_calendar_date_is_rejected_and_scenario_is_not_a_historical_split():
    with pytest.raises(api.HTTPException) as error:
        api.predict(api.PredictionRequest(mode="what_if", car_park_id="PARK-1", observation_date="2017-04-01", observation_time="08:00:00", assumed_occupancy=40))
    assert "between 2017-01-01 and 2017-03-31" in error.value.detail
    scenario = api.predict(api.PredictionRequest(mode="what_if", car_park_id="PARK-1", observation_date="2017-01-01", observation_time="08:00:00", assumed_occupancy=40))
    assert scenario["split"] == "scenario"
    assert scenario["outcome_available"] is False