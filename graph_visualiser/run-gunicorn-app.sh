#!/bin/bash

GV_WORKERS=4
GV_PORT=8050
GV_HOST=0.0.0.0

uvicorn asgi:asgi_app \
  --host "${GV_HOST}" \
  --port "${GV_PORT}" \
  --workers "${GV_WORKERS}"

