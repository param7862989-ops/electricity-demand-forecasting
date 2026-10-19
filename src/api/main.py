from datetime import datetime
from functools import lru_cache
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
import pandas as pd
from src.inference.predict import (
    HORIZON,
    INPUT_WINDOW,
    MODEL_NAME,
    load_artifacts,
    load_data,
    build_features,
    prepare_input,
    generate_forecast,
    forecast_from_history,
)


class ForecastPoint(BaseModel):
    timestamp: datetime
    predicted_demand: float


class ForecastResponse(BaseModel):
    forecast_horizon: int
    input_window: int
    model: str
    forecast: list[ForecastPoint]


class DemandPoint(BaseModel):
    timestamp: datetime
    total_demand: float = Field(ge=0)


class ForecastRequest(BaseModel):
    history: list[DemandPoint] = Field(
        min_length=INPUT_WINDOW,
        description=(
            f"Hourly aggregate demand. The last {INPUT_WINDOW} points "
            "must be consecutive hours."
        ),
    )


class HistoryResponse(BaseModel):
    history: list[DemandPoint]


app = FastAPI(
    title="Electricity Demand Forecasting API",
    description="API for 24-hour electricity demand forecasting",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_artifacts():
    return load_artifacts()


def load_artifacts_or_503():
    try:
        return get_artifacts()
    except (OSError, ValueError) as error:
        raise HTTPException(
            status_code=503,
            detail=f"Model artifacts are unavailable: {error}"
        )


def load_data_or_503():
    try:
        return load_data()
    except (OSError, ValueError) as error:
        raise HTTPException(
            status_code=503,
            detail=f"Demand data is unavailable: {error}"
        )


def build_response(forecast):
    return {
        "forecast_horizon": HORIZON,
        "input_window": INPUT_WINDOW,
        "model": MODEL_NAME,
        "forecast": forecast.to_dict(orient="records")
    }


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "electricity-demand-forecasting-api"
    }


@app.get("/model-info")
def model_info():
    return {
        "model": MODEL_NAME,
        "input_window": INPUT_WINDOW,
        "forecast_horizon": HORIZON,
        "frequency": "hourly",
    }


@app.get("/predict", response_model=ForecastResponse)
def predict():
    """Forecast the 24 hours after the end of the bundled dataset."""
    model, feature_scaler, target_scaler = load_artifacts_or_503()
    df = load_data_or_503()
    features = build_features(df)
    model_input = prepare_input(features, feature_scaler)

    last_timestamp = df["timestamp"].iloc[-1]

    forecast = generate_forecast(
        model,
        target_scaler,
        model_input,
        last_timestamp
    )

    return build_response(forecast)


@app.post("/predict", response_model=ForecastResponse)
def predict_from_history(request: ForecastRequest):
    """Forecast the 24 hours after a caller-supplied demand history."""
    df = pd.DataFrame([point.model_dump() for point in request.history])
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

    model, feature_scaler, target_scaler = load_artifacts_or_503()

    try:
        forecast = forecast_from_history(
            df,
            model,
            feature_scaler,
            target_scaler
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))

    return build_response(forecast)


@app.get("/history", response_model=HistoryResponse)
def history(hours: int = Query(default=168, ge=1, le=24 * 365)):
    """Return the most recent observed hourly demand."""
    df = load_data_or_503().tail(hours)

    return {"history": df.to_dict(orient="records")}
