# ucs_pipeline.py
# ------------------------------------------
# Machine Learning Pipeline for UCS Prediction
# with Cross-Validation & SHAP Explainability
# ------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import shap
import joblib
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------
# Step 1: Load Dataset
# ------------------------------------------
data_path = "rock_test_dataset.csv"  # change as needed
df = pd.read_csv(data_path)

# ------------------------------------------
# Step 2: Define features and target
# ------------------------------------------
features = [
    "Density", "Porosity", "Moisture_Content", "Pwave_Velocity",
    "Schmidt_Rebound_Value", "Lithology", "Point_Load_Index", "Depth"
]
target = "UCS"

X = df[features]
y = df[target]

# ------------------------------------------
# Step 3: Preprocessing
# ------------------------------------------
numeric_features = [
    "Density", "Porosity", "Moisture_Content",
    "Pwave_Velocity", "Schmidt_Rebound_Value",
    "Point_Load_Index", "Depth"
]
categorical_features = ["Lithology"]

numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
categorical_transformer = Pipeline(steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ------------------------------------------
# Step 4: Model Setup
# ------------------------------------------
model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# ------------------------------------------
# Step 5: Cross-Validation
# ------------------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X, y, cv=kf, scoring='r2')

print("Cross-Validation Results:")
print(f"R² Scores (5 folds): {np.round(cv_scores, 3)}")
print(f"Mean R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ------------------------------------------
# Step 6: Train-Test Split & Model Training
# ------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# ------------------------------------------
# Step 7: Model Evaluation
# ------------------------------------------
y_pred = pipeline.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nModel Performance on Test Set:")
print(f"R² Score: {r2:.3f}")
print(f"MAE: {mae:.3f} MPa")
print(f"RMSE: {rmse:.3f} MPa")

# ------------------------------------------
# Step 8: Visualization
# ------------------------------------------

# 1️⃣ Actual vs Predicted
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred, color="royalblue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title("Actual vs Predicted UCS")
plt.xlabel("Actual UCS (MPa)")
plt.ylabel("Predicted UCS (MPa)")
plt.tight_layout()
plt.show()

# 2️⃣ Feature Importance (for tree-based models)
model_step = pipeline.named_steps["model"]
if hasattr(model_step, "feature_importances_"):
    cat_cols = pipeline.named_steps["preprocessor"].transformers_[1][1]\
        .named_steps["encoder"].get_feature_names_out(categorical_features)
    all_features = numeric_features + list(cat_cols)
    importances = model_step.feature_importances_

    fi_df = pd.DataFrame({"Feature": all_features, "Importance": importances})
    fi_df = fi_df.sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(8,5))
    sns.barplot(x="Importance", y="Feature", data=fi_df, palette="viridis")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()

# ------------------------------------------
# Step 9: SHAP Explainability
# ------------------------------------------
print("\nGenerating SHAP summary plot...")
# Extract preprocessed data for SHAP
X_train_preprocessed = pipeline.named_steps["preprocessor"].transform(X_train)
# Get feature names after encoding
cat_cols = pipeline.named_steps["preprocessor"].transformers_[1][1]\
    .named_steps["encoder"].get_feature_names_out(categorical_features)
feature_names = numeric_features + list(cat_cols)

explainer = shap.Explainer(model_step, X_train_preprocessed, feature_names=feature_names)
shap_values = explainer(X_train_preprocessed)

# Global summary plot
shap.summary_plot(shap_values, features=X_train_preprocessed, feature_names=feature_names)

# Single-sample explanation
sample_idx = 0
print("\nSHAP explanation for sample 0:")
shap.plots.waterfall(shap_values[sample_idx])

# ------------------------------------------
# Step 10: Save Model
# ------------------------------------------
joblib.dump(pipeline, "ucs_model_pipeline.pkl")
print("✅ Model saved as ucs_model_pipeline.pkl")

# ------------------------------------------
# Step 11: Example Prediction
# ------------------------------------------
# sample = {
#     "Density": 2.65,
#     "Porosity": 5.2,
#     "Moisture_Content": 1.1,
#     "Pwave_Velocity": 5600,
#     "Schmidt_Rebound_Value": 45,
#     "Lithology": "Granite",
#     "Point_Load_Index": 80,
#     "Depth": 30
# }

# sample_df = pd.DataFrame([sample])
# pred_value = pipeline.predict(sample_df)[0]
# print(f"\nPredicted UCS for sample: {pred_value:.2f} MPa")
