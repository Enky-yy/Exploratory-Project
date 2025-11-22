# main.py
import os
import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# ========== Load Models and Scalers ==========
MODEL_DIR = "E:/Exploratory Project/MSPS/model"

import joblib
def safe_load_pickle(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return joblib.load(path)


UCS_MODEL = safe_load_pickle(os.path.join(MODEL_DIR, "ucs_model.pkl"))
UCS_SCALER = safe_load_pickle(os.path.join(MODEL_DIR, "ucs_scaler.pkl"))
SLOPE_MODEL = safe_load_pickle(os.path.join(MODEL_DIR, "slope_model.pkl"))
SLOPE_SCALER = safe_load_pickle(os.path.join(MODEL_DIR, "slope_scaler.pkl"))

# ========== FastAPI App ==========
app = FastAPI(title="Mine Safety Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # React runs on localhost:5173 or 3000
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== Input Schema ==========
class MineInput(BaseModel):
    cohesion: float
    friction_angle: float
    slope_angle: float
    slope_height: float
    PoreWaterPressureRatio: float
    density: float
    porosity: float
    moisture_content: float
    p_wave_velocity: float
    schmidt_rebound_value: float
    lithology: float
    point_load_index: float
    depth: float

# ========== Prediction Route ==========
@app.post("/predict")
def predict_mine_safety(data: MineInput):
    # Step 1: UCS Prediction
    ucs_features = np.array([[data.density, data.porosity, data.moisture_content,
                              data.p_wave_velocity, data.schmidt_rebound_value,
                              data.lithology, data.point_load_index, data.depth]])
    ucs_scaled = UCS_SCALER.transform(ucs_features)
    predicted_ucs = UCS_MODEL.predict(ucs_scaled)[0]

    # Step 2: Slope Stability Prediction (using predicted UCS)
    slope_features = np.array([[data.cohesion, data.friction_angle, data.slope_angle,
                                data.slope_height, data.PoreWaterPressureRatio, predicted_ucs]])
    slope_scaled = SLOPE_SCALER.transform(slope_features)
    fos = SLOPE_MODEL.predict(slope_scaled)[0]

    # Step 3: Determine Failure
    failure = "Stable ✅" if fos > 1.2 else "Potential Failure ⚠️"

    return {
        "Predicted_UCS": round(float(predicted_ucs), 2),
        "Predicted_FOS": round(float(fos), 3),
        "Failure_Status": failure
    }
