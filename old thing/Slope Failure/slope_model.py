# ===============================================================
# 🏔️ Slope Failure Risk Prediction using XGBoost + Interactive SHAP + PKL Export
# ===============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier

# ===============================================================
# 1️⃣ Load Dataset
# ===============================================================
CSV_PATH = "slope_stability_dataset.csv"  # 🔁 Replace with your CSV file name
data = pd.read_csv(CSV_PATH)

print("\n📘 Columns detected:", list(data.columns))

# ===============================================================
# 2️⃣ Define Target Variable (based on Factor of Safety)
# ===============================================================
if 'Factor of Safety (FS)' not in data.columns:
    raise ValueError("❌ Column 'Factor of Safety (FS)' not found in CSV. Please check the file.")

# Create binary target: 1 = Failure Risk, 0 = Stable
data['Stability'] = np.where(data['Factor of Safety (FS)'] <= 1.3, 1, 0)

print("\n✅ Target column 'Stability' created (threshold FoS ≤ 1.3 = Failure Risk):")
print(data[['Factor of Safety (FS)', 'Stability']].head())

# ===============================================================
# 3️⃣ Prepare Features
# ===============================================================
drop_cols = ['Factor of Safety (FS)', 'Stability']
X = data.drop(columns=drop_cols)
y = data['Stability']

# Encode categorical variable
if 'Reinforcement Type' in X.columns:
    X = pd.get_dummies(X, columns=['Reinforcement Type'], drop_first=True)

feature_names = X.columns.tolist()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Scale data
scaler = StandardScaler()
preprocessed_X_train = scaler.fit_transform(X_train)
preprocessed_X_test = scaler.transform(X_test)

# ===============================================================
# 4️⃣ Train Model
# ===============================================================
clf = XGBClassifier(
    n_estimators=400,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    eval_metric='logloss'
)
clf.fit(preprocessed_X_train, y_train)

# ===============================================================
# 5️⃣ Evaluate Model
# ===============================================================
y_pred = clf.predict(preprocessed_X_test)
y_proba = clf.predict_proba(preprocessed_X_test)[:, 1]

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Stable", "Failure Risk"]))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stable", "Failure Risk"])
disp.plot(cmap="Reds")
plt.title("Confusion Matrix — Slope Failure Risk")
plt.show()

# ===============================================================
# 💾 6️⃣ Save Model and Scaler as PKL
# ===============================================================
os.makedirs("model_outputs", exist_ok=True)

with open("model_outputs/SlopeFailureModel.pkl", "wb") as f:
    pickle.dump(clf, f)

with open("model_outputs/SlopeFailureScaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\n💾 Model saved as 'model_outputs/SlopeFailureModel.pkl'")
print("💾 Scaler saved as 'model_outputs/SlopeFailureScaler.pkl'")

# ===============================================================
# 7️⃣ SHAP Explainability (Interactive)
# ===============================================================
print("\n🧠 Generating SHAP explainability plots (interactive)...")

# Initialize SHAP JS (interactive)
shap.initjs()

# Create directory (optional)
os.makedirs("shap_outputs", exist_ok=True)

# Build SHAP Explainer
explainer = shap.Explainer(clf.predict_proba, preprocessed_X_train, feature_names=feature_names)
shap_values = explainer(preprocessed_X_test)
shap_values_failure = shap_values[..., 1]

# Save explainer as PKL
with open("model_outputs/SlopeFailureExplainer.pkl", "wb") as f:
    pickle.dump(explainer, f)
print("💾 SHAP explainer saved as 'model_outputs/SlopeFailureExplainer.pkl'")

# --- Summary Plot (Interactive popup) ---
print("\n📈 Showing SHAP Summary Plot...")
plt.figure()
shap.summary_plot(
    shap_values_failure,
    preprocessed_X_test,
    feature_names=feature_names,
    show=True
)
plt.title("SHAP Summary Plot — Slope Failure Risk (Red → Higher Risk)")
plt.tight_layout()
plt.savefig("shap_outputs/shap_summary_plot.png", dpi=300)
plt.close()

# --- Feature Importance (Interactive popup) ---
print("\n📊 Showing SHAP Feature Importance...")
plt.figure()
shap.summary_plot(
    shap_values_failure,
    preprocessed_X_test,
    feature_names=feature_names,
    plot_type="bar",
    show=True
)
plt.title("Feature Importance — Drivers of Slope Failure Risk")
plt.tight_layout()
plt.savefig("shap_outputs/shap_feature_importance.png", dpi=300)
plt.close()

# --- Waterfall Plot (Interactive popup for highest risk sample) ---
failure_probs = clf.predict_proba(preprocessed_X_test)[:, 1]
high_risk_idx = np.argmax(failure_probs)

print(f"\n🚨 Showing Waterfall Plot for Highest Risk Sample (Index {high_risk_idx})...")
plt.figure()
shap.plots.waterfall(shap_values_failure[high_risk_idx], show=True)
plt.title(f"SHAP Waterfall — Highest Risk Sample (index {high_risk_idx})")
plt.tight_layout()
plt.savefig("shap_outputs/shap_waterfall_highest_risk.png", dpi=300)
plt.close()

# ===============================================================
# 8️⃣ Export Prediction Report
# ===============================================================
report_df = X_test.copy()
report_df['Actual_Stability'] = y_test.values
report_df['Predicted_Stability'] = y_pred
report_df['Failure_Probability'] = y_proba
report_df['Error'] = abs(report_df['Actual_Stability'] - report_df['Predicted_Stability'])

label_map = {0: "Stable", 1: "Failure Risk"}
report_df['Actual_Label'] = report_df['Actual_Stability'].map(label_map)
report_df['Predicted_Label'] = report_df['Predicted_Stability'].map(label_map)

report_path = "model_outputs/slope_failure_predictions.csv"
report_df.to_csv(report_path, index=False)

# ===============================================================
# ✅ Summary
# ===============================================================
print("\n✅ Model, SHAP, and prediction report complete.")
print("📂 Plots saved in: shap_outputs/")
print("📂 Model, scaler, and explainer saved in: model_outputs/")
print("📄 Prediction report saved in:", report_path)
print("\nLegend:")
print("• Blue → Pushes model toward 'Stable'")
print("• Red → Pushes model toward 'Failure Risk'")
print("• X-axis → Magnitude of influence on probability of failure\n")
