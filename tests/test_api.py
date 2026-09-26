from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_endpoint():
    response = client.get("/predict")

    assert response.status_code == 200

    data = response.json()

    assert data["forecast_horizon"] == 24
    assert data["input_window"] == 24
    assert data["model"] == "GRU-128"
    assert len(data["forecast"]) == 24


def test_predict_response_structure():
    response = client.get("/predict")

    data = response.json()
    first_point = data["forecast"][0]

    assert "timestamp" in first_point
    assert "predicted_demand" in first_point
    assert isinstance(first_point["predicted_demand"], float)