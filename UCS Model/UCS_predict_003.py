import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import joblib

warnings.filterwarnings("ignore")


DATA_PATH = "UCS Model/rock_test_data.csv"
OUT_DIR = "UCS Model/model_outputs"
OUT_MODEL_PATH = os.path.join(OUT_DIR, "ucs_pipeline.joblib")
os.makedirs(OUT_DIR, exist_ok=True)


if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data file not found at: {DATA_PATH}")

data = pd.read_csv(DATA_PATH)
print("\n✅ Dataset loaded. Shape:", data.shape)
print(data.head())

# Basic sanity checks
required_cols = [
    "density", "porosity", "moisture_content", "p_wave_velocity",
    "schmidt_rebound", "point_load_index", "depth", "UCS"
]
missing = set(required_cols) - set(data.columns)
if missing:
    raise ValueError(f"Missing required columns in data: {missing}")

if data[required_cols].isnull().any().any():
    print("⚠️ Data contains NaNs. Dropping rows with NaNs for now.")
    data = data.dropna(subset=required_cols).reset_index(drop=True)

features = [
    "density", "porosity", "moisture_content",
    "p_wave_velocity", "schmidt_rebound",
    "point_load_index", "depth"
]
target = "UCS"

X = data[features]
y = data[target]


numeric_features = features
numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features)
    ],
    remainder="drop"
)


model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n🚀 Training model...")
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n📊 Model Performance:")
print(f"R² Score: {r2:.3f}")
print(f"MAE: {mae:.3f} MPa")
print(f"RMSE: {rmse:.3f} MPa")

plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title("Actual vs Predicted UCS")
plt.xlabel("Actual UCS (MPa)")
plt.ylabel("Predicted UCS (MPa)")
plt.tight_layout()
plt.show()

# Feature importance (from trained XGBoost model inside pipeline)
model_step = pipeline.named_steps["model"]
feature_names = numeric_features

if hasattr(model_step, "feature_importances_"):
    importances = model_step.feature_importances_
    if len(importances) != len(feature_names):
        print("⚠️ Feature importance length mismatch; skipping barplot.")
    else:
        importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        importance_df = importance_df.sort_values("Importance", ascending=False)

        plt.figure(figsize=(8, 4))
        sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")
        plt.title("Feature Importance (XGBoost)")
        plt.tight_layout()
        plt.show()
else:
    print("\n⚠️ Feature importance not available for this model type.")


print("\n📈 Generating SHAP summary plot...")


X_train_preprocessed = pipeline.named_steps["preprocessor"].transform(X_train)

# Try TreeExplainer first (works well with tree models)
try:
    explainer = shap.TreeExplainer(model_step)
    # shap_values may be array or list depending on shap version
    shap_values = explainer.shap_values(X_train_preprocessed)
    # Use the same preprocessed features for plotting and the original names for labels
    # shap.summary_plot accepts: shap_values, features (array), feature_names
    shap.summary_plot(shap_values, X_train_preprocessed, feature_names=feature_names, show=True)
except Exception as e:
    print(f"\n⚠️ SHAP TreeExplainer failed: {e}")
    print("➡️ Trying fallback: shap.Explainer with model.predict ...")
    try:
        explainer = shap.Explainer(model_step.predict, X_train_preprocessed)
        shap_expl = explainer(X_train_preprocessed)  # Explanation object
        # Explanation object works directly with summary_plot
        shap.summary_plot(shap_expl, feature_names=feature_names, show=True)
    except Exception as e2:
        print(f"\n❌ SHAP fallback also failed: {e2}")
        print("Skipping SHAP plots. You can debug SHAP locally or upgrade/downgrade shap package.")


joblib.dump(pipeline, OUT_MODEL_PATH)
print(f"\n💾 Model saved as '{OUT_MODEL_PATH}'")



