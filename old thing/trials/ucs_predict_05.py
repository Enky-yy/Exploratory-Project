# -------------------------------------------------------------
# UCS Prediction from Rock Properties with Cross-Validation
# Author: Harshvardhan Shah
# -------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# =====================================
# 📂 1. Load dataset
# =====================================
data = pd.read_csv("ucs_data_01.csv")  # 🔹 Replace with your actual CSV path

# =====================================
# 🧹 2. Data Preprocessing
# =====================================
features = [
    "Density", "Porosity", "Vp", "E", "Poisson_ratio",
    "Water_Absorption", "Grain_Size", "Weathering_Index",
    "Saturation", "Foliation_Angle", "Mineral_Hardness"
]

X = data[features]
y = data["UCS"]

# Handle missing values
X = X.fillna(X.mean())

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================
# ⚙️ 3. Pipeline Setup
# =====================================
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(random_state=42))
])

# Two model sets (RF + XGBoost)
param_grid = [
    {
        'model': [RandomForestRegressor(random_state=42)],
        'model__n_estimators': [100, 200],
        'model__max_depth': [10, 20, None]
    },
    {
        'model': [XGBRegressor(objective="reg:squarederror", random_state=42)],
        'model__n_estimators': [100, 200],
        'model__max_depth': [3, 10, 20]
    }
]

# =====================================
# 🚀 4. Model Training + Hyperparameter Tuning
# =====================================
grid = GridSearchCV(
    pipeline, param_grid,
    cv=5, scoring='r2',
    n_jobs=-1, verbose=2
)

print("⏳ Training the model... Please wait.")
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print("\n✅ Best Model:", best_model)
print("✅ Best Parameters:", grid.best_params_)

# =====================================
# 📊 5. Evaluation on Test Data
# =====================================
y_pred = best_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("\n📉 Performance Summary:")
print(f"🔹 R² Score     : {r2:.4f}")
print(f"🔹 RMSE         : {rmse:.4f}")
print(f"🔹 MAE          : {mae:.4f}")

# =====================================
# 🔁 6. Cross-Validation (5-Fold)
# =====================================
print("\n⚙️ Performing 5-Fold Cross-Validation...")

cv_r2 = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
cv_rmse = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
cv_rmse = -cv_rmse  # make positive

cv_summary = pd.DataFrame({
    "Fold": np.arange(1, 6),
    "R2_Score": cv_r2,
    "RMSE": cv_rmse
})
cv_summary.loc["Mean"] = ["Mean", cv_r2.mean(), cv_rmse.mean()]
cv_summary.loc["Std"] = ["Std Dev", cv_r2.std(), cv_rmse.std()]

print("\n✅ Cross-Validation Results:")
print(cv_summary)
cv_summary.to_csv("cv_performance_summary.csv", index=False)
print("💾 Cross-validation results saved to 'cv_performance_summary.csv'.")

# =====================================
# 📈 7. Visualizations
# =====================================

# --- Predicted vs Actual ---
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title("Predicted vs Actual UCS")
plt.xlabel("Actual UCS (MPa)")
plt.ylabel("Predicted UCS (MPa)")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Residual Plot ---
residuals = y_test.to_numpy() - y_pred
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, linestyle='--', color='r')
plt.title("Residual Plot")
plt.xlabel("Predicted UCS (MPa)")
plt.ylabel("Residuals")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Cross-validation Performance Plot ---
plt.figure(figsize=(8, 4))
plt.plot(range(1, 6), cv_r2, marker='o', label='R²')
plt.plot(range(1, 6), cv_rmse, marker='s', label='RMSE')
plt.title("Cross-Validation Performance per Fold")
plt.xlabel("Fold Number")
plt.ylabel("Metric Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Feature Importance ---
if hasattr(best_model.named_steps['model'], 'feature_importances_'):
    importances = best_model.named_steps['model'].feature_importances_
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print("\n📊 Feature Importance:")
    print(importance_df)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title("Feature Importance (Numeric Only)")
    plt.tight_layout()
    plt.show()

# =====================================
# 💾 8. Save Predictions
# =====================================
output_df = X_test.copy()
output_df["Actual_UCS"] = y_test
output_df["Predicted_UCS"] = y_pred
output_df["Residuals"] = y_test - y_pred

output_df.to_csv("predicted_ucs_results.csv", index=False)
print("\n💾 Predictions saved as 'predicted_ucs_results.csv'.")

print("\n🎯 UCS Prediction, Error Analysis & Validation Complete!")
