import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


CSV_PATH = "Slope Failure Model/slope.csv"
OUT_DIR = "Slope Failure Model/model_outputs"
SHAP_OUT_DIR = "Slope Failure Model/shap_outputs"
OUT_PIPELINE = os.path.join(OUT_DIR, "slope_failure_pipeline.joblib")
REPORT_PATH = os.path.join(OUT_DIR, "slope_failure_predictions.csv")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SHAP_OUT_DIR, exist_ok=True)


if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

data = pd.read_csv(CSV_PATH)
print("\n📘 Columns detected:", list(data.columns))

required_col = "factor of safety"
if required_col not in data.columns:
    raise ValueError(f"❌ Column '{required_col}' not found in CSV. Please check the file.")


data['Stability'] = np.where(data['factor of safety'] <= 1.3, 1, 0)
print("\n✅ Target column 'Stability' created (threshold FoS ≤ 1.3 = Failure Risk)")
print(data[[required_col, 'Stability']].head())


drop_cols = ['factor of safety', 'Stability']
feature_cols = [c for c in data.columns if c not in drop_cols]
if data[feature_cols + ['Stability']].isnull().any().any():
    print("⚠️ NaNs detected in features/target. Dropping rows with NaNs.")
    data = data.dropna(subset=feature_cols + ['Stability']).reset_index(drop=True)

X = data[feature_cols].copy()
y = data['Stability'].copy()
feature_names = X.columns.tolist()


numeric_features = feature_names  # all numeric in your dataset
numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

preprocessor = ColumnTransformer(
    transformers=[("num", numeric_transformer, numeric_features)],
    remainder="drop"
)

clf = XGBClassifier(
    n_estimators=400,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    verbosity=0
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", clf)
])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)


print("\n🚀 Training classifier pipeline...")
pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]  # probability of class 1 (Failure Risk)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Stable", "Failure Risk"]))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stable", "Failure Risk"])
disp.plot(cmap="Reds")
plt.title("Confusion Matrix — Slope Failure Risk")
plt.tight_layout()
plt.show()


joblib.dump(pipeline, OUT_PIPELINE)
print(f"\n💾 Pipeline saved as '{OUT_PIPELINE}'")


print("\n🧠 Generating SHAP explainability plots (computed on preprocessed features)...")

# get model inside pipeline and preprocessed arrays (NumPy)
model_step = pipeline.named_steps["model"]
X_train_preprocessed = pipeline.named_steps["preprocessor"].transform(X_train)
X_test_preprocessed = pipeline.named_steps["preprocessor"].transform(X_test)

# Try TreeExplainer first (fast for tree models)
shap_values_for_class1 = None
try:
    tree_explainer = shap.TreeExplainer(model_step)
    shap_vals = tree_explainer.shap_values(X_test_preprocessed)
    
    if isinstance(shap_vals, list) or (isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3):
        # multi-output / multiclass -> take index 1 as "failure" class if present
        try:
            shap_values_for_class1 = shap_vals[1]
        except Exception:
            # fallback: if last dimension corresponds to classes
            if shap_vals.shape[-1] > 1:
                shap_values_for_class1 = shap_vals[..., 1]
            else:
                shap_values_for_class1 = shap_vals
    else:
        shap_values_for_class1 = shap_vals  # binary returns 2D array usually
except Exception as e:
    print(f"⚠️ TreeExplainer failed: {e}\n➡️ Trying shap.Explainer(pipeline.predict_proba, raw X)...")
    try:
        explainer = shap.Explainer(pipeline.predict_proba, X_train)  # pass raw X (DataFrame)
        shap_exp = explainer(X_test)  # Explanation object
        # shap_exp.values shape can be: (n_samples, n_features, n_classes)
        vals = shap_exp.values
        if vals.ndim == 3 and vals.shape[-1] > 1:
            shap_values_for_class1 = vals[..., 1]
        elif vals.ndim == 2:
            shap_values_for_class1 = vals
        else:
            raise ValueError("Unexpected SHAP values shape from shap.Explainer.")
    except Exception as e2:
        print(f"❌ shap.Explainer fallback also failed: {e2}")
        shap_values_for_class1 = None

# Plot SHAP summary and bar plots if we obtained values
if shap_values_for_class1 is not None:
    # Summary (be careful: shap.summary_plot expects either (shap_values, features array) or Explanation object)
    print("\n📈 Saving SHAP summary plot...")
    plt.figure()
    shap.summary_plot(shap_values_for_class1, X_test_preprocessed, feature_names=feature_names, show=True)
    plt.title("SHAP Summary Plot — Slope Failure Risk (class=Failure)")
    plt.tight_layout()
    plt.savefig(os.path.join(SHAP_OUT_DIR, "shap_summary_plot.png"), dpi=300)
    plt.close()

    print("\n📊 Saving SHAP feature importance (bar)...")
    plt.figure()
    shap.summary_plot(shap_values_for_class1, X_test_preprocessed, feature_names=feature_names, plot_type="bar", show=True)
    plt.title("Feature Importance — Drivers of Slope Failure Risk")
    plt.tight_layout()
    plt.savefig(os.path.join(SHAP_OUT_DIR, "shap_feature_importance.png"), dpi=300)
    plt.close()

    # Waterfall for one sample (highest predicted failure probability)
    high_risk_idx = int(np.argmax(y_proba))
    print(f"\n🚨 Saving SHAP Waterfall for highest risk sample (test index {high_risk_idx})...")
    plt.figure()
    try:
        # If shap_values_for_class1 is 2D (n_samples, n_features)
        shap.plots.waterfall(shap.Explanation(values=shap_values_for_class1[high_risk_idx],
                                             base_values=None,
                                             data=X_test_preprocessed[high_risk_idx],
                                             feature_names=feature_names),
                             show=True)
    except Exception:
        # alternative: if shap package expects explanation object already produced earlier
        try:
            shap.plots.waterfall(shap_values_for_class1[high_risk_idx], show=True)
        except Exception as e:
            print("⚠️ Could not draw waterfall plot:", e)
    plt.title(f"SHAP Waterfall — Highest Risk Sample (test idx {high_risk_idx})")
    plt.tight_layout()
    plt.savefig(os.path.join(SHAP_OUT_DIR, "shap_waterfall_highest_risk.png"), dpi=300)
    plt.close()
else:
    print("\n⚠️ SHAP values not available — skipping SHAP plots. If you want, I can debug specific SHAP errors.")


report_df = X_test.copy()
report_df['Actual_Stability'] = y_test.values
report_df['Predicted_Stability'] = y_pred
report_df['Failure_Probability'] = y_proba
report_df['Error'] = np.abs(report_df['Actual_Stability'] - report_df['Predicted_Stability'])

label_map = {0: "Stable", 1: "Failure Risk"}
report_df['Actual_Label'] = report_df['Actual_Stability'].map(label_map)
report_df['Predicted_Label'] = report_df['Predicted_Stability'].map(label_map)

report_df.to_csv(REPORT_PATH, index=False)
print("\n✅ Prediction report saved to:", REPORT_PATH)

print("\n✅ Model pipeline and SHAP outputs complete.")
print("📂 Plots saved in:", SHAP_OUT_DIR)
print("📂 Pipeline saved in:", OUT_PIPELINE)
