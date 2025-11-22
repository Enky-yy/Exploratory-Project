# ================================
# ✅ 1. Import Libraries
# ================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# ================================
# ✅ 2. Create Sample Dataset (Replace this with real data later)
# ================================
np.random.seed(42)
sample_size = 200

data = pd.DataFrame({
    "Rock_Type": np.random.choice(["Granite", "Basalt", "Limestone", "Shale"], sample_size),
    "Density": np.random.uniform(2400, 2900, sample_size),
    "Porosity": np.random.uniform(0.5, 15, sample_size),
    "Vp": np.random.uniform(3000, 6000, sample_size),
    "E": np.random.uniform(10, 80, sample_size),
    "Poisson_ratio": np.random.uniform(0.15, 0.35, sample_size),
    "Water_Absorption": np.random.uniform(0.2, 5, sample_size),
    "Grain_Size": np.random.uniform(0.01, 5, sample_size),
    "Weathering_Index": np.random.uniform(1, 10, sample_size),
    "Saturation": np.random.uniform(0, 1, sample_size),
    "Foliation_Angle": np.random.uniform(0, 90, sample_size),
    "Mineral_Hardness": np.random.uniform(3, 10, sample_size),
})

# Define UCS with some logical influence (just sample logic)
data["UCS"] = (
    0.03 * data["Density"] -
    0.5 * data["Porosity"] +
    0.002 * data["Vp"] +
    0.8 * data["E"] -
    10 * data["Water_Absorption"] +
    2 * data["Mineral_Hardness"] -
    0.1 * data["Weathering_Index"] +
    np.random.uniform(-10, 10, sample_size)  # noise
)

# ================================
# ✅ 3. Define Features & Target
# ================================
target = "UCS"
features = [
    "Rock_Type", "Density", "Porosity", "Vp", "E", "Poisson_ratio",
    "Water_Absorption", "Grain_Size", "Weathering_Index",
    "Saturation", "Foliation_Angle", "Mineral_Hardness"
]

numeric_features = [
    "Density", "Porosity", "Vp", "E", "Poisson_ratio",
    "Water_Absorption", "Grain_Size", "Weathering_Index",
    "Saturation", "Foliation_Angle", "Mineral_Hardness"
]

categorical_features = ["Rock_Type"]

# ================================
# ✅ 4. Preprocessor & Pipeline
# ================================
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

# ================================
# ✅ 5. Train-Test Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    data[features], data[target], test_size=0.2, random_state=42
)

# ================================
# ✅ 6. Model Training with Cross-Validation + GridSearch
# ================================
from sklearn.model_selection import GridSearchCV

param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [None, 10, 20],
    'model__min_samples_split': [2, 5],
    'model__min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,                # 5-Fold Cross Validation
    scoring='r2',        # You can change to 'neg_mean_squared_error' etc.
    n_jobs=-1,           # Use all CPU cores
    verbose=1
)

print("\n🔍 Running Grid Search with Cross-Validation...")
grid_search.fit(X_train, y_train)

print("\n✅ Best Parameters Found:")
print(grid_search.best_params_)

print(f"\n📊 Best Cross-Validated R² Score: {grid_search.best_score_:.3f}")

# Update pipeline to best found model
best_model = grid_search.best_estimator_


# ================================
# ✅ 7. Prediction using Best Model
# ================================
y_pred = best_model.predict(X_test)

# ================================
# ✅ 8. Error Metrics
# ================================
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n✅ Final Model Performance on Test Data:")
print(f"MAE:  {mae:.3f}")
print(f"MSE:  {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R² Score: {r2:.3f}")

# ================================
# ✅ 9. Feature Importances (Numeric Only)
# ================================
importances = pipeline.named_steps['model'].feature_importances_

# OneHotEncoded rock types appear after numeric features, so skip them
num_feature_count = len(numeric_features)
numeric_importances = importances[:num_feature_count]

importance_df = pd.DataFrame({
    "Feature": numeric_features,
    "Importance": numeric_importances
}).sort_values(by="Importance", ascending=False)

print("\n📊 Numeric Feature Importances (excluding Rock_Type dummy variables):")
print(importance_df)

# Plot
plt.figure(figsize=(10, 5))
sns.barplot(x="Importance", y="Feature", data=importance_df)
plt.title("Numeric Feature Importances (Excluding Rock_Type)")
plt.show()
# =====================================