from fastapi.testclient import TestClient

from parkpulse.api import app


def test_prediction_endpoints_smoke():
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        options = client.get("/api/options")
        assert options.status_code == 200
        default = options.json()["default_historical"]
        available = client.get("/api/historical/options", params={"car_park_id": default["car_park_id"], "observation_date": default["observation_date"]})
        assert available.status_code == 200
        observation = available.json()["observations"][0]
        prediction = client.post("/api/predict", json={"mode": "historical", "car_park_id": default["car_park_id"], "observation_date": default["observation_date"], "observation_time": observation["observation_time"]})
        assert prediction.status_code == 200
        assert prediction.json()["forecast_timestamp"]
        scenario = client.post("/api/predict", json={"mode": "what_if", "car_park_id": default["car_park_id"], "observation_date": "2017-01-01", "observation_time": "08:00:00", "assumed_occupancy": 10})
        assert scenario.status_code == 200
        assert scenario.json()["split"] == "scenario"