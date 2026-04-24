import pandas as pd

# ---------------------------------------
# File Paths
# ---------------------------------------

INPUT_PATH = "../data/features/match_feature_view_with_xi.csv"
OUTPUT_PATH = "../data/features/match_feature_view_with_xi_weighted.csv"

print("Loading dataset...")

df = pd.read_csv(INPUT_PATH)

print("Shape:", df.shape)

# ---------------------------------------
# Auction-Aware Weight Mapping
# ---------------------------------------

season_weights = {

    2026: 1.00,
    2025: 0.90,

    2024: 0.70,
    2023: 0.60,
    2022: 0.55,

    2021: 0.40,
    2020: 0.30,
    2019: 0.25,
    2018: 0.25

}

# ---------------------------------------
# Apply Weights
# ---------------------------------------

print("Applying auction-aware weights...")

df["recency_weight"] = df["season"].map(season_weights)

# Safety check

if df["recency_weight"].isna().sum() > 0:

    missing = df[df["recency_weight"].isna()]["season"].unique()

    raise ValueError(
        f"Missing weights for seasons: {missing}"
    )

# ---------------------------------------
# Diagnostics
# ---------------------------------------

print("\nWeight Summary:")

print(
    df.groupby("season")["recency_weight"]
    .mean()
    .sort_index()
)

# ---------------------------------------
# Save Output
# ---------------------------------------

df.to_csv(OUTPUT_PATH, index=False)

print("\nSaved weighted dataset to:")

print(OUTPUT_PATH)

print("\nDone.")