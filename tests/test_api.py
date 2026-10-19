import pandas as pd
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_endpoint():
    mock_forecast = pd.DataFrame([
        {
            "timestamp": f"2015-01-01T{i:02d}:00:00",
            "predicted_demand": 100000.0 + i
        }
        for i in range(24)
    ])

    with patch(
        "src.api.main.get_artifacts",
        return_value=(None, None, None)
    ), patch(
        "src.api.main.load_data"
    ) as mock_load_data, patch(
        "src.api.main.build_features"
    ) as mock_build_features, patch(
        "src.api.main.prepare_input"
    ) as mock_prepare_input, patch(
        "src.api.main.generate_forecast",
        return_value=mock_forecast
    ):

        mock_load_data.return_value = pd.DataFrame({
            "timestamp": pd.date_range(
                "2014-12-31 00:00:00",
                periods=24,
                freq="h"
            )
        })

        mock_build_features.return_value = None
        mock_prepare_input.return_value = None

        response = client.get("/predict")

    assert response.status_code == 200

    data = response.json()

    assert data["forecast_horizon"] == 24
    assert data["input_window"] == 24
    assert data["model"] == "GRU-128"
    assert len(data["forecast"]) == 24
    
def test_predict_response_structure():
    mock_forecast = pd.DataFrame([
        {
            "timestamp": f"2015-01-01T{i:02d}:00:00",
            "predicted_demand": 100000.0 + i
        }
        for i in range(24)
    ])

    with patch(
        "src.api.main.get_artifacts",
        return_value=(None, None, None)
    ), patch(
        "src.api.main.load_data"
    ) as mock_load_data, patch(
        "src.api.main.build_features"
    ) as mock_build_features, patch(
        "src.api.main.prepare_input"
    ) as mock_prepare_input, patch(
        "src.api.main.generate_forecast",
        return_value=mock_forecast
    ):

        mock_load_data.return_value = pd.DataFrame({
            "timestamp": pd.date_range(
                "2014-12-31 00:00:00",
                periods=24,
                freq="h"
            )
        })

        mock_build_features.return_value = None
        mock_prepare_input.return_value = None

        response = client.get("/predict")

    assert response.status_code == 200

    data = response.json()
    first_point = data["forecast"][0]

    assert "timestamp" in first_point
    assert "predicted_demand" in first_point
    assert isinstance(first_point["predicted_demand"], float)

def make_history_payload(periods=24, freq="h"):
    timestamps = pd.date_range("2014-12-31 00:00:00", periods=periods, freq=freq)

    return {
        "history": [
            {"timestamp": ts.isoformat(), "total_demand": 150000.0 + i}
            for i, ts in enumerate(timestamps)
        ]
    }


def test_post_predict_with_custom_history():
    mock_forecast = pd.DataFrame({
        "timestamp": pd.date_range("2015-01-01", periods=24, freq="h"),
        "predicted_demand": [100000.0] * 24,
    })

    with patch(
        "src.api.main.get_artifacts",
        return_value=(None, None, None)
    ), patch(
        "src.api.main.forecast_from_history",
        return_value=mock_forecast
    ) as mock_forecast_from_history:
        response = client.post("/predict", json=make_history_payload())

    assert response.status_code == 200
    assert len(response.json()["forecast"]) == 24

    history = mock_forecast_from_history.call_args.args[0]
    assert len(history) == 24
    assert list(history.columns) == ["timestamp", "total_demand"]


def test_post_predict_rejects_short_history():
    response = client.post("/predict", json=make_history_payload(periods=10))

    assert response.status_code == 422


def test_post_predict_rejects_gaps_in_history():
    with patch(
        "src.api.main.get_artifacts",
        return_value=(None, None, None)
    ):
        response = client.post(
            "/predict",
            json=make_history_payload(freq="2h")
        )

    assert response.status_code == 422
    assert "consecutive" in response.json()["detail"]


def test_predict_returns_503_when_artifacts_missing():
    with patch(
        "src.api.main.get_artifacts",
        side_effect=OSError("model file not found")
    ):
        response = client.get("/predict")

    assert response.status_code == 503


def test_history_endpoint():
    mock_data = pd.DataFrame({
        "timestamp": pd.date_range("2014-12-01", periods=200, freq="h"),
        "total_demand": [150000.0] * 200,
    })

    with patch("src.api.main.load_data", return_value=mock_data):
        response = client.get("/history?hours=48")

    assert response.status_code == 200
    assert len(response.json()["history"]) == 48


def test_model_info_endpoint():
    response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json()["forecast_horizon"] == 24
