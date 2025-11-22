# run in backend/app directory or a small script where joblib is available
import joblib, os, pprint
PIPE_DIR = os.path.join(os.path.dirname(__file__), "models")
ucs_path = os.path.join(PIPE_DIR, "ucs_pipeline.joblib")
slope_path = os.path.join(PIPE_DIR, "slope_failure_pipeline.joblib")

ucs_pipe = joblib.load(ucs_path)
slope_pipe = joblib.load(slope_path)

def get_expected_features(pipe):
    # If pipeline is sklearn.pipeline.Pipeline with a preprocessor ColumnTransformer:
    try:
        pre = pipe.named_steps["preprocessor"]
        # Attempt to extract numeric transformer columns
        for name, transformer, cols in pre.transformers:
            # transformer could be ('num', Pipeline(...), cols)
            return cols
    except Exception:
        pass
    # fallback: sklearn >= 1.0 stores feature names in estimator
    try:
        return list(pipe.feature_names_in_)
    except Exception:
        return None

print("UCS expected features:")
pprint.pprint(get_expected_features(ucs_pipe))

print("Slope expected features:")
pprint.pprint(get_expected_features(slope_pipe))
