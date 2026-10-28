#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
pkill -f "streamlit run app/streamlit_app.py" 2>/dev/null || true
streamlit cache clear || true
streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
