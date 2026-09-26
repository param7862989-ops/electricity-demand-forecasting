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