# ucs_predict_pipeline_fixed.py
# 🚀 Complete ML Pipeline for UCS Prediction with SHAP Explainability

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# ---------------------------------------------------------------------
# 1️⃣ Load dataset (CSV)
# ---------------------------------------------------------------------
data = pd.read_csv("rock_test_data.csv")

print("\n✅ Dataset Loaded Successfully")
print(data.head())

# ---------------------------------------------------------------------
# 2️⃣ Define features and target
# ---------------------------------------------------------------------
features = [
    "Density", "Porosity", "Moisture_Content", "Pwave_Velocity",
    "Schmidt_Rebound", "Lithology", "Point_Load_Index", "Depth"
]
target = "UCS"

X = data[features]
y = data[target]

# Identify numerical & categorical columns
numeric_features = [
    "Density", "Porosity", "Moisture_Content", "Pwave_Velocity",
    "Schmidt_Rebound", "Point_Load_Index", "Depth"
]
categorical_features = ["Lithology"]

# ---------------------------------------------------------------------
# 3️⃣ Preprocessing pipeline
# ---------------------------------------------------------------------
numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
categorical_transformer = Pipeline(steps=[("encoder", OneHotEncoder(drop="first"))])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# ---------------------------------------------------------------------
# 4️⃣ Model definition & pipeline
# ---------------------------------------------------------------------
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# ---------------------------------------------------------------------
# 5️⃣ Split dataset
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------------------
# 6️⃣ Train model
# ---------------------------------------------------------------------
print("\n🚀 Training model...")
pipeline.fit(X_train, y_train)

# ---------------------------------------------------------------------
# 7️⃣ Predictions & evaluation
# ---------------------------------------------------------------------
y_pred = pipeline.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n📊 Model Performance:")
print(f"R² Score: {r2:.3f}")
print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")

# ---------------------------------------------------------------------
# 8️⃣ Feature Importance Plot
# ---------------------------------------------------------------------

# 1️⃣ Actual vs Predicted
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred, color="royalblue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title("Actual vs Predicted UCS")
plt.xlabel("Actual UCS (MPa)")
plt.ylabel("Predicted UCS (MPa)")
plt.tight_layout()
plt.show()

model_step = pipeline.named_steps["model"]

if hasattr(model_step, "feature_importances_"):
    feature_names = list(preprocessor.transformers_[0][2]) + \
                    list(pipeline.named_steps["preprocessor"].transformers_[1][1]["encoder"].get_feature_names_out(categorical_features))

    importances = model_step.feature_importances_
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    importance_df = importance_df.sort_values("Importance", ascending=False)

    plt.figure(figsize=(10, 5))
    sns.barplot(x="Importance", y="Feature", data=importance_df)
    plt.title("Feature Importance (XGBoost)")
    plt.tight_layout()
    plt.show()
else:
    print("\n⚠️ Feature importance not available for this model type.")

# ---------------------------------------------------------------------
# 9️⃣ SHAP Explainability
# ---------------------------------------------------------------------
print("\n📈 Generating SHAP summary plot...")

# Get preprocessed training data
X_train_preprocessed = pipeline.named_steps["preprocessor"].transform(X_train)

# Option 1: TreeExplainer for tree-based models (recommended)
try:
    explainer = shap.TreeExplainer(model_step)
    shap_values = explainer.shap_values(X_train_preprocessed)
    shap.summary_plot(shap_values, features=X_train, feature_names=feature_names)
except Exception as e:
    print(f"\n⚠️ SHAP TreeExplainer failed: {e}")
    print("➡️ Trying fallback Explainer method...")
    # Option 2: Use generic SHAP explainer
    explainer = shap.Explainer(model_step.predict, X_train_preprocessed)
    shap_values = explainer(X_train_preprocessed)
    shap.summary_plot(shap_values, feature_names=feature_names)

# ---------------------------------------------------------------------
# 🔟 Save model & results
# ---------------------------------------------------------------------
import joblib
joblib.dump(pipeline, "ucs_model.pkl")
print("\n💾 Model saved as 'ucs_model.pkl'")
