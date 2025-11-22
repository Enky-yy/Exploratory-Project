import os
import joblib
import numpy as np
import pandas as pd


MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models_artifacts')
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
lith_encoder = joblib.load(os.path.join(MODEL_DIR, 'lith_encoder.pkl'))


FEATURE_ORDER = ['density','porosity','moisture','p_wave_vel','schmidt','point_load_index','depth','lith_encoded']


def preprocess_row(df: pd.DataFrame):
# basic validation & fillna
 df2 = df.copy()
 df2 = df2.fillna(0)
 if 'lithology' in df2.columns:
    try:
      df2['lith_encoded'] = lith_encoder.transform(df2['lithology'].astype(str))
    except Exception:
# fallback: map unknowns to 0
      df2['lith_encoded'] = 0
    else:
      df2['lith_encoded'] = 0
    num_cols = ['density','porosity','moisture','p_wave_vel','schmidt','point_load_index','depth']
    Xnum = df2[num_cols].astype(float)
    Xnum_scaled = scaler.transform(Xnum)
    X = np.hstack([Xnum_scaled, df2[['lith_encoded']].values])
    return X