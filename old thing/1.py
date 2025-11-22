# ucs_scaler_generator.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Load your UCS dataset
df = pd.read_csv("rock_test_data.csv")

# Select same features used in model training
features = ['density', 'porosity', 'moisture_content', 'p_wave_velocity',
            'schmidt_rebound','point_load_index', 'depth']

X = df[features]

# Fit scaler
scaler = StandardScaler()
scaler.fit(X)

# Save properly as pickle
joblib.dump(scaler, "ucs_scaler.pkl")

print("✅ UCS scaler saved successfully!")
