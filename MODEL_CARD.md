# Model card

ParkPulse uses `sklearn.linear_model.LinearRegression` inside a `Pipeline` and `ColumnTransformer`. Inputs are current occupancy, known capacity, car-park code, and calendar fields at the forecast timestamp. The model predicts occupied spaces 30 minutes ahead; availability is `capacity - bounded forecast`.

The chronological split is approximately 60/20/20, with target-crossing rows removed at each boundary. The persistence baseline is “occupancy in 30 minutes equals occupancy now.” On the real supplied data, the test set contains 45 exact-target examples: regression MAE **59.13** spaces and RMSE **74.86**, versus persistence MAE **40.09** and RMSE **61.90**. Raw regression predictions were outside physical bounds on **8.9%** of test examples and are clipped only for the application. These are historical 2016 results, not a current service guarantee.

The exact-target coverage is recorded in the manifest. Small per-car-park counts make those estimates unstable. Monitoring uses training-derived immutable reference data, normalised Wasserstein distance for numeric features and total variation for the car-park category. Thresholds are demonstrations and need validation and calibration before production. Drift alerts recommend investigation; they do not retrain or deploy automatically.
