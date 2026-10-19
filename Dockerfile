FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=8000

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY dashboard ./dashboard
COPY models/final_gru_best.keras models/feature_scaler.joblib models/target_scaler.joblib ./models/
COPY data/processed/hourly_demand.csv ./data/processed/
COPY start.sh ./start.sh

RUN chmod +x start.sh && useradd --create-home appuser && mkdir -p reports && chown -R appuser /app
USER appuser

EXPOSE 8000

# Serves the API by default. To run the dashboard from the same image instead:
#   docker run -p 8501:8501 -e APP=dashboard -e PORT=8501 <image>
CMD ["./start.sh"]
