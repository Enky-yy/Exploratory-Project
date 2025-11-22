from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
import pandas as pd

from app.loaders import load_pipelines
from app.msi import compute_msi
import json
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi.responses import JSONResponse

app = FastAPI(title="Explo MSI API", version="0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("uvicorn")

# Load pipelines at startup
ucs_pipe, slope_pipe = load_pipelines()

class UCSRequest(BaseModel):
    density: float
    porosity: float
    moisture_content: float
    p_wave_velocity: float
    schmidt_rebound: float
    point_load_index: float
    depth: float

from typing import Optional
import pandas as pd
from fastapi import HTTPException

class SlopeRequest(BaseModel):
    cohesion: float
    friction_angle: float
    slope_angle: float
    slope_height: float
    pore_pressure: float         
    ucs: Optional[float] = None 

# --- Helper to map frontend slope keys to pipeline feature names ---
SLOPE_FRONTEND_TO_PIPELINE = {
    "pore_pressure": "pore water pressure ratio",
    "cohesion": "cohesion",
    "friction_angle": "friction_angle",
    "slope_angle": "slope_angle",
    "slope_height": "slope_height",
}

class MSIRequest(BaseModel):
    ucs: Optional[UCSRequest] = None
    slope: Optional[SlopeRequest] = None
    ucs_value: Optional[float] = None
    slope_failure_prob: Optional[float] = None
    weights: Optional[Dict[str, float]] = None

from fastapi import HTTPException
import pandas as pd

def _expected_features_from_pipeline(pipe):
    try:
        pre = pipe.named_steps["preprocessor"]
        cols = []
        for name, transformer, columns in pre.transformers:
            if isinstance(columns, (list, tuple)):
                cols.extend(list(columns))
        if cols:
            return cols
    except Exception:
        pass
    try:
        return list(pipe.feature_names_in_)
    except Exception:
        pass
    return None

FRONTEND_TO_PIPELINE = {
    "pore_pressure": "pore water pressure ratio",
    "cohesion": "cohesion",
    "friction_angle": "friction_angle",
    "slope_angle": "slope_angle",
    "slope_height": "slope_height",
}

# ----------------------
# Endpoints
# ----------------------
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict/ucs")
async def predict_ucs(payload: UCSRequest):
    if ucs_pipe is None:
        raise HTTPException(status_code=500, detail="UCS pipeline not loaded")
    try:
        X = pd.DataFrame([payload.dict()])
        ucs_pred = float(ucs_pipe.predict(X)[0])
        return {"ucs_mpa": ucs_pred}
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/slope")
async def predict_slope(payload: SlopeRequest):
    """
    Accepts frontend-friendly slope keys, maps them to the pipeline feature names,
    injects 'ucs' if the model expects it (and a value is provided), and returns
    failure probability and predicted class.
    """
    if slope_pipe is None:
        raise HTTPException(status_code=500, detail="Slope pipeline not loaded")
    try:
        # Map the payload to pipeline feature names
        raw = payload.dict()
        mapped = {}
        for k, v in raw.items():
            if k == "ucs":
                # handle ucs separately (don't map the key name)
                continue
            mapped_key = SLOPE_FRONTEND_TO_PIPELINE.get(k, k)
            mapped[mapped_key] = v

        # If slope pipeline expects 'ucs', ensure we have it (either provided in payload or compute via ucs_pipe)
        exp_slope = None
        try:
            pre = slope_pipe.named_steps["preprocessor"]
            exp_cols = []
            for _, _, cols in pre.transformers:
                if isinstance(cols, (list, tuple)):
                    exp_cols.extend(list(cols))
            if exp_cols:
                exp_slope = exp_cols
        except Exception:
            # fallback
            try:
                exp_slope = list(slope_pipe.feature_names_in_)
            except Exception:
                exp_slope = None

        # If 'ucs' required by model, attempt to get it
        if exp_slope and "ucs" in exp_slope:
            ucs_val = raw.get("ucs", None)
            if ucs_val is None:
                # if frontend didn't supply ucs, try computing via loaded ucs_pipe if available
                if ucs_pipe is not None:
                    # we need the UCS features to compute it; without them we cannot compute here
                    raise HTTPException(
                        status_code=400,
                        detail="Slope model requires 'ucs'. Provide 'ucs' value in the slope payload or call /predict/msi which computes it for you."
                    )
                else:
                    raise HTTPException(status_code=500, detail="Slope model requires 'ucs' but UCS pipeline is not available to compute it.")
            mapped["ucs"] = float(ucs_val)

        # Validate mapped columns presence (informative error)
        if exp_slope:
            missing = set(exp_slope) - set(mapped.keys())
            if missing:
                raise HTTPException(status_code=400, detail=f"Slope input missing columns: {missing}")

        slope_df = pd.DataFrame([mapped])

        probs = slope_pipe.predict_proba(slope_df)[0]
        prob_failure = float(probs[1])
        pred_class = int(slope_pipe.predict(slope_df)[0])

        return {"failure_probability": prob_failure, "predicted_class": pred_class}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/msi")
async def predict_msi(payload: MSIRequest):
    ucs_val = None
    slope_prob = None

    try:
        # 1) Compute UCS first if raw UCS features provided
        if payload.ucs is not None:
            if ucs_pipe is None:
                raise HTTPException(status_code=500, detail="UCS pipeline not loaded")
            raw_ucs = payload.ucs.dict()
            ucs_df = pd.DataFrame([raw_ucs])

            exp_ucs = _expected_features_from_pipeline(ucs_pipe)
            if exp_ucs:
                missing = set(exp_ucs) - set(ucs_df.columns)
                if missing:
                    raise HTTPException(status_code=400, detail=f"UCS input missing columns: {missing}")

            ucs_val = float(ucs_pipe.predict(ucs_df)[0])

        if payload.ucs_value is not None:
            ucs_val = float(payload.ucs_value)

        # 2) Prepare slope features and map to pipeline expected names
        if payload.slope is not None:
            if slope_pipe is None:
                raise HTTPException(status_code=500, detail="Slope pipeline not loaded")

            raw_slope = payload.slope.dict()

            # map friendly frontend names to pipeline names
            mapped_slope = {}
            for k, v in raw_slope.items():
                target_k = FRONTEND_TO_PIPELINE.get(k, k)
                mapped_slope[target_k] = v

            # If slope pipeline expects 'ucs' as a feature, ensure we have ucs_val and inject it
            exp_slope = _expected_features_from_pipeline(slope_pipe)
            if exp_slope and "ucs" in exp_slope:
                if ucs_val is None:
                    # if user didn't provide ucs features but provided ucs_value earlier, use that
                    if payload.ucs_value is not None:
                        ucs_val = float(payload.ucs_value)
                    else:
                        raise HTTPException(status_code=400, detail="Slope model expects 'ucs'. Provide 'ucs' features or 'ucs_value'.")
                mapped_slope["ucs"] = ucs_val

            slope_df = pd.DataFrame([mapped_slope])

            # Validate slope columns against expected
            if exp_slope:
                missing = set(exp_slope) - set(slope_df.columns)
                if missing:
                    raise HTTPException(status_code=400, detail=f"Slope input missing columns: {missing}")

            slope_prob = float(slope_pipe.predict_proba(slope_df)[0][1])

        if payload.slope_failure_prob is not None:
            slope_prob = float(payload.slope_failure_prob)

        if ucs_val is None or slope_prob is None:
            raise HTTPException(status_code=400, detail="Provide either features for both models or precomputed ucs_value and slope_failure_prob")

        weights = payload.weights or {"w_ucs": 0.6, "w_slope": 0.4}
        msi_score = compute_msi(ucs_val, slope_prob, weights=weights)

        return {
            "ucs": ucs_val,
            "slope_failure_prob": slope_prob,
            "msi": msi_score,
            "weights": weights
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))
    