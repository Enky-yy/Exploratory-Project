# backend/app/msi.py
from typing import Dict
import numpy as np

DEFAULT_UCS_MIN = 0.0
DEFAULT_UCS_MAX = 250.0

def compute_msi(ucs: float, slope_failure_prob: float, weights: Dict[str, float] = None, ucs_min: float = DEFAULT_UCS_MIN, ucs_max: float = DEFAULT_UCS_MAX) -> float:
    if weights is None:
        weights = {"w_ucs": 0.6, "w_slope": 0.4}

    norm_ucs = (ucs - ucs_min) / (ucs_max - ucs_min) if ucs_max > ucs_min else 0.0
    norm_ucs = float(np.clip(norm_ucs, 0.0, 1.0))

    safe_from_slope = 1.0 - float(np.clip(slope_failure_prob, 0.0, 1.0))

    w_ucs = float(weights.get("w_ucs", 0.6))
    w_slope = float(weights.get("w_slope", 0.4))
    s = w_ucs + w_slope
    if s <= 0:
        w_ucs, w_slope = 0.6, 0.4
    else:
        w_ucs /= s
        w_slope /= s

    composite = w_ucs * norm_ucs + w_slope * safe_from_slope
    return float(np.round(100.0 * composite, 3))
