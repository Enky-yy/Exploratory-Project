# slopeModel.py

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# -----------------------------------------------------
# 1️⃣ Load dataset
# -----------------------------------------------------
file_path = r"slope.csv"  # update path if needed
df = pd.read_csv(file_path)

required_cols = ['cohesion', 'friction_angle', 'slope_angle', 'slope_height',
                 'Pore Water Pressure Ratio', 'UCS', 'Factor_of_Safety']

if not all(col in df.columns for col in required_cols):
    raise ValueError(f"❌ Missing one or more required columns. Found: {df.columns.tolist()}")

# -----------------------------------------------------
# 2️⃣ Define features and target
# -----------------------------------------------------
X = df[['cohesion', 'friction_angle', 'slope_angle', 'slope_height',
        'Pore Water Pressure Ratio', 'UCS']]
y = np.where(df['Factor_of_Safety'] < 1.3, 1, 0)  # 1 = Failure, 0 = Stable

# -----------------------------------------------------
# 3️⃣ Split dataset
# -----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------------------------------
# 4️⃣ Train model
# -----------------------------------------------------
model = XGBClassifier(
    n_estimators=150,
    learning_rate=0.08,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)
model.fit(X_train, y_train)

# -----------------------------------------------------
# 5️⃣ Model evaluation
# -----------------------------------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {acc:.3f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------------------------------
# 6️⃣ SHAP explainability (safe version)
# -----------------------------------------------------
print("🔍 Generating SHAP explanations...")

explainer = shap.Explainer(model.predict, X_train)
shap_values = explainer(X_test)

# SHAP summary plot popup
shap.summary_plot(shap_values, X_test, feature_names=X.columns)
plt.show()

# -----------------------------------------------------
# 7️⃣ Prediction Function
# -----------------------------------------------------
def predict_slope_failure(cohesion, friction_angle, slope_angle, slope_height, pwp_ratio, ucs):
    input_data = np.array([[cohesion, friction_angle, slope_angle, slope_height, pwp_ratio, ucs]])
    fos_pred = model.predict_proba(input_data)[0][1]  # probability of failure
    status = "Failure Likely" if fos_pred > 0.5 else "Stable"
    return {"Failure_Probability": float(fos_pred), "Predicted_Status": status}

# Example prediction
example = predict_slope_failure(25, 30, 45, 20, 0.3, 35)
print("\n🔹 Example Prediction:", example)
