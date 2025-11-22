# ucs_predict.py
# 📦 1. Import dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# =====================================
# 📂 2. Load dataset
data = pd.read_csv("ucs_data_01.csv")

# =====================================
# 🧹 3. Data Preprocessing
features = [
    "Density", "Porosity", "Vp", "E", "Poisson_ratio",
    "Water_Absorption", "Grain_Size", "Weathering_Index",
    "Saturation", "Foliation_Angle", "Mineral_Hardness"
]


X = data[features]
y = data["UCS"]

# Handle missing values (if any)
X.fillna(X.mean(), inplace=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================
# ⚙️ 4. Model Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(random_state=42))
])

param_grid = [
    {
        'model': [RandomForestRegressor(random_state=42)],
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 10, 20]
    },
    {
        'model': [XGBRegressor(objective="reg:squarederror", random_state=42)],
        'model__n_estimators': [100, 200],
        # XGB max_depth must be an int; avoid None
        'model__max_depth': [3, 10, 20]
    }
]

# =====================================
# 🚀 5. Hyperparameter Tuning
grid = GridSearchCV(
    pipeline, param_grid,
    cv=5, scoring='r2',
    n_jobs=-1, verbose=2
)

print("⏳ Training the model... This may take some time.")
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print("\n✅ Best Model:", best_model)
print("✅ Best Parameters:", grid.best_params_)

# =====================================
# 📊 6. Model Evaluation
y_pred = best_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("\n📉 Performance Summary:")
print(f"🔹 R² Score     : {r2:.4f}")
print(f"🔹 RMSE         : {rmse:.4f}")
print(f"🔹 MAE          : {mae:.4f}")

# =====================================
# 📈 7. Visualization

# 7.1 Predicted vs Actual Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title("Predicted vs Actual UCS")
plt.xlabel("Actual UCS (MPa)")
plt.ylabel("Predicted UCS (MPa)")
plt.grid(True)
plt.show()

# 7.2 Residual Plot
residuals = y_test.to_numpy() - y_pred
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, linestyle='--', color='r')
plt.title("Residual Plot")
plt.xlabel("Predicted UCS (MPa)")
plt.ylabel("Residuals")
plt.grid(True)
plt.show()

# 7.3 Feature Importance (if applicable)
if hasattr(best_model.named_steps['model'], 'feature_importances_'):
    importances = best_model.named_steps['model'].feature_importances_
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title("Feature Importance")
    plt.show()
