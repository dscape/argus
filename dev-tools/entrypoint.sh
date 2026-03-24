#!/bin/sh
set -e

# Start the FastAPI dev-tools API
cd /app/dev-tools
exec uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
