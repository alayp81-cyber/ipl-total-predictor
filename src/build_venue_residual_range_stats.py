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
GLOBAL_STATS_FILE = PROJECT_DIR / "models" / "residual_range_stats.json"

OUTPUT_FILE = PROJECT_DIR / "models" / "venue_residual_range_stats.json"

# ---------------------------------------
# Load Model + Data
# ---------------------------------------

print("Loading model and feature data...")

df = pd.read_csv(FEATURE_FILE)

with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

with open(GLOBAL_STATS_FILE, "r") as f:
    global_stats = json.load(f)

global_half_width = global_stats["range_half_width"]

print("Global fallback width:", global_half_width)

# ---------------------------------------
# Use recent seasons only
# ---------------------------------------

test_df = df[df["season"] >= 2024].copy()

print("Recent rows:", len(test_df))

# ---------------------------------------
# Prepare Predictions
# ---------------------------------------

model_features = model.feature_names_

X = test_df[model_features].copy()
y = test_df["match_total_runs"].copy()

for col in X.columns:
    if X[col].dtype == "object":
        X[col] = X[col].astype(str)

preds = model.predict(X)

test_df["prediction"] = preds
test_df["abs_error"] = np.abs(y - preds)

# ---------------------------------------
# Build Venue Stats
# ---------------------------------------

venue_stats = {}

MIN_MATCHES_REQUIRED = 15

for venue in test_df["venue"].unique():

    venue_df = test_df[test_df["venue"] == venue]

    n_matches = len(venue_df)

    if n_matches >= MIN_MATCHES_REQUIRED:

        p75_error = float(
            np.percentile(
                venue_df["abs_error"],
                75
            )
        )

        half_width = round(p75_error)

        source = "venue_specific"

    else:

        half_width = global_half_width
        source = "global_fallback"

    venue_stats[venue] = {
        "matches_used": int(n_matches),
        "range_half_width": int(half_width),
        "source": source
    }

# ---------------------------------------
# Save Output
# ---------------------------------------

with open(OUTPUT_FILE, "w") as f:
    json.dump(venue_stats, f, indent=4)

print("\nVenue Residual Stats Built")

print("\nSample Output:")

for i, (venue, stats) in enumerate(venue_stats.items()):

    print(venue, stats)

    if i >= 5:
        break

print("\nSaved to:", OUTPUT_FILE)

print("\nDone.")