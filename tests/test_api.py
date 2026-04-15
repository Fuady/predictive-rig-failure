"""
tests/test_api.py
-----------------
Integration tests for the FastAPI inference endpoint.
Tests run without a trained model (uses mock fallback).

Run: pytest tests/test_api.py -v
"""

import sys
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.serving.api import app

client = TestClient(app)

SAMPLE_READING = {
    "timestamp": "2024-01-15T10:30:00",
    "op_setting_1": 0.0023,
    "op_setting_2": 0.0003,
    "op_setting_3": 100.0,
    "sensors": {
        "sensor_1": 2.0,
        "sensor_2": 641.82,
        "sensor_3": 1589.7,
        "sensor_4": 1400.6,
        "sensor_5": 14.62,
        "sensor_6": 21.61,
        "sensor_7": 554.36,
        "sensor_8": 2388.1,
        "sensor_9": 9046.2,
        "sensor_10": 1.30,
        "sensor_11": 47.47,
        "sensor_12": 521.66,
        "sensor_13": 2388.1,
        "sensor_14": 8138.6,
        "sensor_15": 8.42,
        "sensor_16": 0.03,
        "sensor_17": 392.0,
        "sensor_18": 2388.1,
        "sensor_19": 100.0,
        "sensor_20": 38.83,
        "sensor_21": 23.42,
    },
}

SAMPLE_REQUEST = {
    "asset_id": "PUMP-07",
    "readings": [SAMPLE_READING] * 5,
    "subset": "FD001",
}


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:

    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_version" in data
        assert "uptime_seconds" in data

    def test_health_uptime_positive(self):
        response = client.get("/health")
        assert response.json()["uptime_seconds"] >= 0


# ---------------------------------------------------------------------------
# Predict endpoint
# ---------------------------------------------------------------------------

class TestPredictEndpoint:

    def test_predict_returns_200_or_503(self):
        """Either succeeds (model loaded) or returns 503 (no model yet)."""
        response = client.post("/predict", json=SAMPLE_REQUEST)
        assert response.status_code in (200, 503)

    def test_predict_response_structure_when_model_loaded(self):
        response = client.post("/predict", json=SAMPLE_REQUEST)
        if response.status_code == 503:
            pytest.skip("Model not trained yet — run train.py first")

        data = response.json()
        required_keys = [
            "asset_id",
            "failure_probability_96h",
            "predicted_rul_cycles",
            "alert_level",
            "recommendation",
            "model_version",
            "inference_time_ms",
            "timestamp",
        ]
        for key in required_keys:
            assert key in data, f"Missing key: {key}"

    def test_predict_probability_range(self):
        response = client.post("/predict", json=SAMPLE_REQUEST)
        if response.status_code != 200:
            pytest.skip("Model not loaded")
        data = response.json()
        prob = data["failure_probability_96h"]
        assert 0.0 <= prob <= 1.0

    def test_predict_rul_nonnegative(self):
        response = client.post("/predict", json=SAMPLE_REQUEST)
        if response.status_code != 200:
            pytest.skip("Model not loaded")
        assert response.json()["predicted_rul_cycles"] >= 0

    def test_predict_alert_level_valid(self):
        response = client.post("/predict", json=SAMPLE_REQUEST)
        if response.status_code != 200:
            pytest.skip("Model not loaded")
        valid_levels = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
        assert response.json()["alert_level"] in valid_levels

    def test_predict_inference_time_logged(self):
        response = client.post("/predict", json=SAMPLE_REQUEST)
        if response.status_code != 200:
            pytest.skip("Model not loaded")
        assert response.json()["inference_time_ms"] >= 0

    def test_predict_asset_id_echoed(self):
        response = client.post("/predict", json=SAMPLE_REQUEST)
        if response.status_code != 200:
            pytest.skip("Model not loaded")
        assert response.json()["asset_id"] == SAMPLE_REQUEST["asset_id"]

    def test_predict_invalid_sensor_name(self):
        bad_request = {
            "asset_id": "PUMP-01",
            "readings": [{
                **SAMPLE_READING,
                "sensors": {"sensor_99": 123.0},   # invalid sensor name
            }],
        }
        response = client.post("/predict", json=bad_request)
        assert response.status_code == 422

    def test_predict_empty_readings_rejected(self):
        bad_request = {"asset_id": "PUMP-01", "readings": []}
        response = client.post("/predict", json=bad_request)
        assert response.status_code == 422

    def test_predict_missing_asset_id_rejected(self):
        bad_request = {"readings": [SAMPLE_READING]}
        response = client.post("/predict", json=bad_request)
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Batch endpoint
# ---------------------------------------------------------------------------

class TestBatchEndpoint:

    def test_batch_predict_returns_200_or_503(self):
        payload = {"assets": [SAMPLE_REQUEST, {**SAMPLE_REQUEST, "asset_id": "COMP-01"}]}
        response = client.post("/predict/batch", json=payload)
        assert response.status_code in (200, 503)

    def test_batch_response_count(self):
        assets = [
            {**SAMPLE_REQUEST, "asset_id": f"PUMP-0{i}"}
            for i in range(1, 4)
        ]
        response = client.post("/predict/batch", json={"assets": assets})
        if response.status_code != 200:
            pytest.skip("Model not loaded")
        data = response.json()
        assert data["n_assets"] == 3
        assert len(data["predictions"]) == 3


# ---------------------------------------------------------------------------
# Asset history endpoint
# ---------------------------------------------------------------------------

class TestHistoryEndpoint:

    def test_history_returns_200(self):
        response = client.get("/assets/PUMP-07/history")
        assert response.status_code == 200

    def test_history_structure(self):
        response = client.get("/assets/PUMP-07/history")
        data = response.json()
        assert "asset_id" in data
        assert "history" in data
        assert isinstance(data["history"], list)

    def test_history_has_30_points(self):
        response = client.get("/assets/PUMP-07/history")
        assert len(response.json()["history"]) == 30

    def test_history_point_structure(self):
        response = client.get("/assets/PUMP-07/history")
        point = response.json()["history"][0]
        for key in ["cycle", "anomaly_score", "failure_probability", "predicted_rul"]:
            assert key in point

    def test_history_probabilities_in_range(self):
        response = client.get("/assets/PUMP-07/history")
        for point in response.json()["history"]:
            assert 0.0 <= point["failure_probability"] <= 1.0
