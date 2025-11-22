# ===============================================================
# ⛏️ Integrated Mine Safety Prediction API
# Combines UCS (Rock Strength) + Slope Failure Risk Models
# ===============================================================

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import shap

# ===============================================================
# 1️⃣ Initialize App
# ===============================================================
app = FastAPI(
    title="Mine Safety Integrated Prediction API",
    description="Predict UCS and slope failure risk from geotechnical inputs.",
    version="2.0"
)

# ===============================================================
# 2️⃣ Load Pre-Trained Models and Scalers
# ===============================================================
with open("app/model/ucs_model.pkl", "rb") as f:
    ucs_model = pickle.load(f)
with open("app/model/ucs_scaler.pkl", "rb") as f:
    ucs_scaler = pickle.load(f)

with open("app/model/SlopeFailureModel.pkl", "rb") as f:
    slope_model = pickle.load(f)
with open("app/model/SlopeFailureScaler.pkl", "rb") as f:
    slope_scaler = pickle.load(f)
with open("app/model/SlopeFailureExplainer.pkl", "rb") as f:
    slope_explainer = pickle.load(f)

# ===============================================================
# 3️⃣ Input Schemas
# ===============================================================
class UCSInput(BaseModel):
    density: float
    porosity: float
    moisture_content: float
    p_wave_velocity: float
    schmidt_rebound: float
    lithology: float
    point_load_index: float
    depth: float


class SlopeInput(BaseModel):
    cohesion: float
    friction_angle: float
    slope_angle: float
    slope_height: float
    pore_water_pressure_ratio: float


class IntegratedInput(UCSInput, SlopeInput):
    """Combines UCS and slope parameters"""


# ===============================================================
# 4️⃣ API Endpoints
# ===============================================================
@app.get("/")
def home():
    return {
        "message": "⛏️ Mine Safety Integrated Prediction API",
        "usage": "POST to /predict_mine_safety with UCS + slope parameters."
    }


@app.post("/predict_mine_safety")
def predict_mine_safety(data: IntegratedInput):
    # -------------------------------------
    # Step 1️⃣ Predict UCS from rock inputs
    # -------------------------------------
    ucs_features = np.array([[
        data.density,
        data.porosity,
        data.moisture_content,
        data.p_wave_velocity,
        data.schmidt_rebound,
        data.lithology,
        data.point_load_index,
        data.depth
    ]])
    ucs_scaled = ucs_scaler.transform(ucs_features)
    predicted_ucs = float(ucs_model.predict(ucs_scaled)[0])

    # -------------------------------------
    # Step 2️⃣ Predict Slope Failure using UCS
    # -------------------------------------
    slope_features = np.array([[
        data.cohesion,
        data.friction_angle,
        data.slope_angle,
        data.slope_height,
        data.pore_water_pressure_ratio,
        predicted_ucs
    ]])
    slope_scaled = slope_scaler.transform(slope_features)

    failure_prob = float(slope_model.predict_proba(slope_scaled)[0][1])
    failure_class = int(slope_model.predict(slope_scaled)[0])
    stability_status = "Failure Risk" if failure_class == 1 else "Stable"

    # -------------------------------------
    # Step 3️⃣ Calculate Approx. Factor of Safety
    # -------------------------------------
    # A simple empirical relation (you can calibrate from your dataset)
    fos_estimated = round((predicted_ucs / (data.slope_height * np.tan(np.radians(data.slope_angle)))) * (1 - data.pore_water_pressure_ratio), 2)

    # -------------------------------------
    # Step 4️⃣ SHAP Explainability
    # -------------------------------------
    shap_values = slope_explainer(slope_scaled)
    shap_contrib = shap_values[..., 1].values[0].tolist()

    feature_names = [
        "Cohesion", "Friction Angle", "Slope Angle",
        "Slope Height", "Pore Water Pressure Ratio", "UCS"
    ]
    shap_details = [{"feature": f, "impact": float(v)} for f, v in zip(feature_names, shap_contrib)]

    # -------------------------------------
    # Step 5️⃣ Return Response
    # -------------------------------------
    return {
        "UCS_predicted_MPa": round(predicted_ucs, 2),
        "Estimated_Factor_of_Safety": fos_estimated,
        "Predicted_Status": stability_status,
        "Failure_Probability": round(failure_prob, 4),
        "Feature_Impacts": shap_details
    }


# ===============================================================
# 🚀 Run the API
# ===============================================================
# Command: uvicorn integratedMineSafetyAPI:app --reload
