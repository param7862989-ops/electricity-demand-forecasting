import os
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predict import (  # noqa: E402
    HORIZON,
    INPUT_WINDOW,
    MODEL_NAME,
    forecast_from_history,
    load_artifacts,
    load_data,
)

# When API_URL is set the dashboard calls the FastAPI backend; otherwise it
# runs the model in-process so it can be deployed on its own.
API_URL = os.environ.get("API_URL", "").rstrip("/")
HISTORY_HOURS = 168

st.set_page_config(
    page_title="Electricity Demand Forecast",
    page_icon="⚡",
    layout="wide"
)


@st.cache_resource(show_spinner="Loading model...")
def get_artifacts():
    return load_artifacts()


@st.cache_data
def get_bundled_history():
    return load_data()


def forecast_via_api(history):
    payload = {
        "history": [
            {
                "timestamp": row.timestamp.isoformat(),
                "total_demand": float(row.total_demand),
            }
            for row in history.itertuples()
        ]
    }
    response = requests.post(f"{API_URL}/predict", json=payload, timeout=60)
    response.raise_for_status()

    forecast = pd.DataFrame(response.json()["forecast"])
    forecast["timestamp"] = pd.to_datetime(forecast["timestamp"])
    return forecast


def run_forecast(history):
    if API_URL:
        return forecast_via_api(history)

    return forecast_from_history(history, *get_artifacts())


def read_uploaded_history(uploaded_file):
    df = pd.read_csv(uploaded_file)

    missing = {"timestamp", "total_demand"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df[["timestamp", "total_demand"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["total_demand"] = pd.to_numeric(df["total_demand"])
    return df.sort_values("timestamp").reset_index(drop=True)


st.title("⚡ Electricity Demand Forecasting")
st.write(
    f"Forecast the next {HORIZON} hours of aggregate electricity demand "
    f"from the previous {INPUT_WINDOW} hours."
)

with st.sidebar:
    st.header("Model Information")
    st.write(f"**Model:** {MODEL_NAME}")
    st.write(f"**Input window:** {INPUT_WINDOW} hours")
    st.write(f"**Forecast horizon:** {HORIZON} hours")
    st.write("**Forecast frequency:** Hourly")
    st.write(f"**Backend:** {'FastAPI at ' + API_URL if API_URL else 'In-process'}")

    st.divider()
    st.header("Input Data")
    source = st.radio(
        "Demand history",
        ["Bundled dataset (UCI 2011–2014)", "Upload CSV"],
    )

    uploaded_file = None
    if source == "Upload CSV":
        uploaded_file = st.file_uploader(
            f"CSV with `timestamp` and `total_demand` columns "
            f"(at least {INPUT_WINDOW} consecutive hourly rows)",
            type="csv",
        )

st.divider()

try:
    if uploaded_file is not None:
        history = read_uploaded_history(uploaded_file)
    elif source == "Upload CSV":
        st.info("Upload a CSV file in the sidebar to generate a forecast.")
        st.stop()
    else:
        history = get_bundled_history()
except (OSError, ValueError) as error:
    st.error(f"Could not load demand history: {error}")
    st.stop()

st.subheader(f"{HORIZON}-Hour Forecast")
st.caption(
    f"History ends at {history['timestamp'].iloc[-1]:%Y-%m-%d %H:%M}; "
    f"forecast covers the following {HORIZON} hours."
)

if st.button("Generate Forecast", type="primary"):
    try:
        with st.spinner("Generating forecast..."):
            forecast = run_forecast(history)

    except requests.exceptions.RequestException as error:
        st.error(
            f"Unable to get a forecast from the API at {API_URL}: {error}"
        )

    except (OSError, ValueError) as error:
        st.error(f"Unable to generate a forecast: {error}")

    else:
        st.success("Forecast generated successfully.")

        predictions = forecast["predicted_demand"]

        col1, col2, col3 = st.columns(3)

        col1.metric("Peak Demand", f"{predictions.max():,.0f}")
        col2.metric("Minimum Demand", f"{predictions.min():,.0f}")
        col3.metric("Average Demand", f"{predictions.mean():,.0f}")

        recent = history.tail(HISTORY_HOURS).rename(
            columns={"total_demand": "Observed Demand"}
        )
        predicted = forecast.rename(
            columns={"predicted_demand": "Predicted Demand"}
        )
        chart_data = (
            pd.concat([recent, predicted])
            .rename(columns={"timestamp": "Timestamp"})
            .set_index("Timestamp")
        )

        st.line_chart(
            chart_data,
            y=["Observed Demand", "Predicted Demand"],
        )

        st.subheader("Forecast Details")

        display_data = predicted.rename(columns={"timestamp": "Timestamp"})
        display_data["Predicted Demand"] = display_data["Predicted Demand"].round(2)

        st.dataframe(
            display_data,
            width="stretch",
            hide_index=True
        )

        st.download_button(
            "Download forecast CSV",
            display_data.to_csv(index=False),
            file_name="forecast.csv",
            mime="text/csv",
        )
