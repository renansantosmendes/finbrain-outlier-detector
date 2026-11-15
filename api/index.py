"""Vercel entry point: exposes the ASGI application as ``app``.

Vercel runs this module as a Python serverless function. ``vercel.json``
rewrites every path to it, and the original path travels in the
``__vercel_path`` query parameter, which the middleware restores. Locally,
run the same application with ``uv run uvicorn api.index:app --reload``.
"""

import importlib
import sys
from pathlib import Path

SOURCE_DIRECTORY = Path(__file__).resolve().parent.parent / "src"
if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

API_MODULE = importlib.import_module("outlier_detector.serving.api")
VERCEL_MODULE = importlib.import_module("outlier_detector.serving.vercel")

app = API_MODULE.create_app()
app.add_middleware(VERCEL_MODULE.RestoreOriginalPathMiddleware)
