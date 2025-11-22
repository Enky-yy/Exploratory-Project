import json
from app.db.session import get_db_session
from app.db.models import SampleRecord, Base
from sqlalchemy.orm import Session
import datetime


# MSI compute — adjust normalization constants according to your training
def compute_msi(ucs_pred, slope_prob, w_u=0.5, w_s=0.5, ucs_max=250.0):
 ucs_norm = max(0.0, min(1.0, ucs_pred / ucs_max))
 msi = w_u * (1 - ucs_norm) + w_s * slope_prob
 return float(msi * 100)


# simple DB persistence
def save_prediction_record(raw: dict, ucs_pred: float, slope_prob: float, msi: float):
 sess = get_db_session()
 rec = SampleRecord(raw=json.dumps(raw), ucs_pred=ucs_pred, slope_prob=slope_prob, msi=msi, msi_cat=("Safe" if msi < 30 else ("Watch" if msi < 60 else "HighRisk")))
 sess.add(rec)
 sess.commit()
 rid = rec.id
 sess.close()
 return rid