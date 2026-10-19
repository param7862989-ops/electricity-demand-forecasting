#!/bin/sh
# Container entrypoint. APP selects the service: "api" (default) or "dashboard".
set -e

PORT="${PORT:-8000}"

case "${APP:-api}" in
    api)
        exec uvicorn src.api.main:app --host 0.0.0.0 --port "$PORT"
        ;;
    dashboard)
        exec streamlit run dashboard/app.py \
            --server.port "$PORT" \
            --server.address 0.0.0.0 \
            --server.headless true \
            --browser.gatherUsageStats false
        ;;
    *)
        echo "Unknown APP '$APP' (expected 'api' or 'dashboard')" >&2
        exit 1
        ;;
esac
