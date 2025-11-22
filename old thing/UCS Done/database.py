# generate_rock_test_data.py
# 🪨 Synthetic Rock Test Dataset Generator for UCS Prediction

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# 1️⃣ Set parameters
# ---------------------------------------------------------------------
np.random.seed(42)
n_samples = 800  # You can increase this for more data

# ---------------------------------------------------------------------
# 2️⃣ Define lithology categories
# ---------------------------------------------------------------------
lithologies = ["Granite", "Basalt", "Sandstone", "Limestone", "Shale"]

# ---------------------------------------------------------------------
# 3️⃣ Generate synthetic feature data (based on realistic mining ranges)
# ---------------------------------------------------------------------
data = pd.DataFrame({
    "Density": np.random.uniform(2.3, 3.0, n_samples),           # g/cm³
    "Porosity": np.random.uniform(0.5, 10, n_samples),           # %
    "Moisture_Content": np.random.uniform(0.1, 5, n_samples),    # %
    "Pwave_Velocity": np.random.uniform(2000, 6500, n_samples),  # m/s
    "Schmidt_Rebound": np.random.uniform(20, 60, n_samples),     # rebound number
    "Lithology": np.random.choice(lithologies, n_samples),
    "Point_Load_Index": np.random.uniform(0.2, 8, n_samples),    # MPa
    "Depth": np.random.uniform(5, 100, n_samples)                # m
})

# ---------------------------------------------------------------------
# 4️⃣ Generate UCS based on a realistic physical correlation
# ---------------------------------------------------------------------
# Empirical correlation + lithology multiplier
lithology_strength_factor = {
    "Granite": 1.25,
    "Basalt": 1.15,
    "Sandstone": 0.85,
    "Limestone": 0.9,
    "Shale": 0.7
}

data["UCS"] = (
    15 * data["Density"] +
    0.015 * data["Pwave_Velocity"] +
    1.2 * data["Schmidt_Rebound"] +
    4 * data["Point_Load_Index"] -
    1.5 * data["Porosity"] -
    0.6 * data["Moisture_Content"] +
    0.03 * data["Depth"]
)

# Apply lithology effect
data["UCS"] = data.apply(lambda x: x["UCS"] * lithology_strength_factor[x["Lithology"]], axis=1)

# Add small random noise
data["UCS"] += np.random.normal(0, 10, n_samples)

# ---------------------------------------------------------------------
# 5️⃣ Save dataset
# ---------------------------------------------------------------------
data.to_csv("rock_test_data.csv", index=False)
print("✅ Synthetic rock test dataset generated and saved as 'rock_test_data.csv'")

# ---------------------------------------------------------------------
# 6️⃣ Quick summary
# ---------------------------------------------------------------------
print("\n📊 Dataset Summary:")
print(data.describe())
print("\n🪨 Lithology Distribution:")
print(data["Lithology"].value_counts())
