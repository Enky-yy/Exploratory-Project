# =====================================
# UCS Prediction | Full ML Pipeline (With Rock_Type Support)
# =====================================

# 📦 1. Import dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# =====================================
# 📥 2. Load dataset
# =====================================
data = pd.read_csv("ucs_data.csv")

# =====================================
# 🏗️ 3. Define features & target (Rock_Type added)
# =====================================
features = [
    "Rock_Type", "Density", "Porosity", "Vp", "E", "Poisson_ratio",
    "Water_Absorption", "Grain_Size", "Weathering_Index",
    "Saturation", "Foliation_Angle", "Mineral_Hardness"
]

X = data[features]
y = data["UCS"]

# =====================================
# ✂️ 4. Split dataset
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================
# ⚙️ 5. Preprocessor (Encoding + Scaling)
# =====================================
numeric_features = [
    "Density", "Porosity", "Vp", "E", "Poisson_ratio",
    "Water_Absorption", "Grain_Size", "Weathering_Index",
    "Saturation", "Foliation_Angle", "Mineral_Hardness"
]

categorical_features = ["Rock_Type"]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# =====================================
# 🧠 6. Pipeline + Models
# =====================================
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

param_grid = {
    'model': [
        RandomForestRegressor(random_state=42),
        XGBRegressor(objective="reg:squarederror", random_state=42)
    ],
    'model__n_estimators': [100, 200],
    'model__max_depth': [None, 10, 20]
}

# =====================================
# 🔍 7. Hyperparameter tuning
# =====================================
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
# 📊 8. Evaluation
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
# 📈 9. Visualizations
# =====================================

# 9.1 Predicted vs Actual
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title("Predicted vs Actual UCS")
plt.xlabel("Actual UCS (MPa)")
plt.ylabel("Predicted UCS (MPa)")
plt.grid(True)
plt.show()

# 9.2 Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, linestyle='--', color='r')
plt.title("Residual Plot")
plt.xlabel("Predicted UCS (MPa)")
plt.ylabel("Residuals")
plt.grid(True)
plt.show()

# 9.3 Feature Importance (if supported)
model_step = best_model.named_steps['model']
if hasattr(model_step, 'feature_importances_'):
    # Get feature names after encoding
    encoded_features = (
        list(numeric_features) + 
        list(best_model.named_steps['preprocessor']
             .named_transformers_['cat']
             .get_feature_names_out(categorical_features))
    )
    importance_df = pd.DataFrame({
        'Feature': encoded_features,
        'Importance': model_step.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title("Feature Importance")
    plt.show()
