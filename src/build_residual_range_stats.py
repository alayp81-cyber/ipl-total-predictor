from pathlib import Path
import pandas as pd
import pickle
import json
import numpy as np

# ---------------------------------------
# Paths
# ---------------------------------------

PROJECT_DIR = Path("/Users/alay/Desktop/ipl_total_predictior")

FEATURE_FILE = PROJECT_DIR / "data" / "features" / "match_feature_view_with_xi_weighted.csv"
MODEL_FILE = PROJECT_DIR / "models" / "latest_model.pkl"
OUTPUT_FILE = PROJECT_DIR / "models" / "residual_range_stats.json"

# ---------------------------------------
# Load
# ---------------------------------------

print("Loading data and model...")

df = pd.read_csv(FEATURE_FILE)

with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

print("Dataset shape:", df.shape)

# ---------------------------------------
# Use recent test period for residuals
# ---------------------------------------

test_df = df[df["season"] >= 2024].copy()

print("Residual test rows:", len(test_df))

# ---------------------------------------
# Feature Columns
# ---------------------------------------

model_features = model.feature_names_

X = test_df[model_features].copy()
y = test_df["match_total_runs"].copy()

# Ensure categorical columns are strings
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = X[col].astype(str)

# ---------------------------------------
# Predict and calculate residuals
# ---------------------------------------

preds = model.predict(X)

residuals = y - preds
abs_errors = np.abs(residuals)

mae = float(abs_errors.mean())
p50 = float(np.percentile(abs_errors, 50))
p75 = float(np.percentile(abs_errors, 75))
p80 = float(np.percentile(abs_errors, 80))
p90 = float(np.percentile(abs_errors, 90))

# Recommended practical range = 75th percentile absolute error
range_half_width = round(p75)

stats = {
    "mae": round(mae, 2),
    "p50_abs_error": round(p50, 2),
    "p75_abs_error": round(p75, 2),
    "p80_abs_error": round(p80, 2),
    "p90_abs_error": round(p90, 2),
    "range_half_width": range_half_width,
    "test_start_season": int(test_df["season"].min()),
    "test_end_season": int(test_df["season"].max()),
    "test_rows": int(len(test_df))
}

# ---------------------------------------
# Save
# ---------------------------------------

with open(OUTPUT_FILE, "w") as f:
    json.dump(stats, f, indent=4)

print("\nResidual Range Stats:")
print(json.dumps(stats, indent=4))

print("\nSaved to:", OUTPUT_FILE)
print("\nDone.")