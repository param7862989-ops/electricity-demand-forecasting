
from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel
from src.inference.predict import (
    load_artifacts,
    load_data,
    build_features,
    prepare_input,
    generate_forecast
)

class ForecastPoint(BaseModel):
    timestamp: datetime
    predicted_demand: float

class ForecastResponse(BaseModel):
    forecast_horizon: int
    input_window: int
    model: str
    forecast: list[ForecastPoint]
app = FastAPI(
    title="Electricity Demand Forecasting API",
    description="API for 24-hour electricity demand forecasting",
    version="1.0.0"
)
model, feature_scaler, target_scaler = load_artifacts()


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "electricity-demand-forecasting-api"
    }


@app.get("/predict", response_model=ForecastResponse)
def predict():

    df = load_data()
    features = build_features(df)
    model_input = prepare_input(features, feature_scaler)

    last_timestamp = df["timestamp"].iloc[-1]

    forecast = generate_forecast(
        model,
        target_scaler,
        model_input,
        last_timestamp
    )

    return {
        "forecast_horizon": 24,
        "input_window": 24,
        "model": "GRU-128",
        "forecast": forecast.to_dict(orient="records")
    }