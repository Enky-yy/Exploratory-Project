# backend/app/loaders.py
import os
import joblib
from typing import Tuple, Optional

BASE = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.normpath(os.path.join(BASE, "models"))
UCS_PIPE = os.path.join(MODELS_DIR, "ucs_pipeline.joblib")
SLOPE_PIPE = os.path.join(MODELS_DIR, "slope_failure_pipeline.joblib")

def load_pipeline(path: str):
    if not os.path.exists(path):
        return None
    try:
        pipe = joblib.load(path)
        return pipe
    except Exception as e:
        print(f"Failed to load pipeline {path}: {e}")
        return None

def load_pipelines() -> Tuple[Optional[object], Optional[object]]:
    ucs = load_pipeline(UCS_PIPE)
    slope = load_pipeline(SLOPE_PIPE)
    if ucs is None:
        print("Warning: UCS pipeline not loaded. Place ucs_pipeline.joblib in backend/app/models/")
    if slope is None:
        print("Warning: Slope pipeline not loaded. Place slope_failure_pipeline.joblib in backend/app/models/")
    return ucs, slope
