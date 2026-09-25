import requests
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Electricity Demand Forecast",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Electricity Demand Forecasting")
st.write("24-hour electricity demand forecasting dashboard")

with st.sidebar:
    st.header("Model Information")
    st.write("**Model:** GRU-128")
    st.write("**Input window:** 24 hours")
    st.write("**Forecast horizon:** 24 hours")
    st.write("**Forecast frequency:** Hourly")

st.divider()

st.subheader("24-Hour Forecast")

if st.button("Generate Forecast", type="primary"):
    try:
        response = requests.get(
            "http://127.0.0.1:8000/predict",
            timeout=30
        )

        response.raise_for_status()

    except requests.exceptions.RequestException:
        st.error(
            "Unable to connect to the forecasting API. "
            "Make sure the FastAPI server is running."
        )

    else:
        data = response.json()

        st.success("Forecast generated successfully.")

        forecast = data["forecast"]

        timestamps = [point["timestamp"] for point in forecast]
        predictions = [point["predicted_demand"] for point in forecast]

        max_demand = max(predictions)
        min_demand = min(predictions)
        avg_demand = sum(predictions) / len(predictions)

        col1, col2, col3 = st.columns(3)

        col1.metric("Peak Demand", f"{max_demand:,.0f}")
        col2.metric("Minimum Demand", f"{min_demand:,.0f}")
        col3.metric("Average Demand", f"{avg_demand:,.0f}")

        chart_data = pd.DataFrame({
            "Timestamp": pd.to_datetime(timestamps),
            "Predicted Demand": predictions
        })

        st.line_chart(
            chart_data,
            x="Timestamp",
            y="Predicted Demand"
        )

        st.subheader("Forecast Details")

        display_data = chart_data.copy()
        display_data["Predicted Demand"] = display_data["Predicted Demand"].round(2)

        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True
        )