#!/bin/sh
set -e

# Start Blender render server in background (for synthetic data generation)
if command -v blender >/dev/null 2>&1 && [ -f /app/blender/render_server.py ]; then
    echo "Starting Blender render server on port 9876..."
    blender --background --python /app/blender/render_server.py -- \
        --port 9876 --quality training \
        > /tmp/blender-server.log 2>&1 &
    echo "Blender render server started (PID $!, log: /tmp/blender-server.log)"
else
    echo "Blender not found or render_server.py missing — skipping render server"
fi

# Start the FastAPI dev-tools API
cd /app/dev-tools
exec uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
