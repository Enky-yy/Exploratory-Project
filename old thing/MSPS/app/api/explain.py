from fastapi import APIRouter
import shap, joblib
import pandas as pd
from app.preprocessing import preprocess_row
from typing import Dict

router = APIRouter()
ucs_model = joblib.load("app/model/ucs_model.pkl")
# shap explainer prebuilt if possible:
ucs_explainer = shap.Explainer(ucs_model.predict, masker=shap.maskers.Independent)  

@router.post("/ucs")
def explain_ucs(sample: Dict):
    df = pd.DataFrame([sample])
    X = preprocess_row(df)
    shap_vals = ucs_explainer(X)
    # Return numeric arrays; front-end will render as popup using shap JS or image
    return {"shap_values": shap_vals.values.tolist(), "base_value": float(shap_vals.base_values)}
