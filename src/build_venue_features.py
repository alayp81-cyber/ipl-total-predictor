from pathlib import Path
import pandas as pd

# ---------------------------------------
# Paths
# ---------------------------------------

PROJECT_DIR = Path("/Users/alay/Desktop/ipl_total_predictior")

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
FEATURE_DIR = PROJECT_DIR / "data" / "features"

INPUT_FILE = PROCESSED_DIR / "clean_team_innings.csv"

OUTPUT_FILE = FEATURE_DIR / "venue_features.csv"

# ---------------------------------------
# Load
# ---------------------------------------

print("Loading team innings...")

df = pd.read_csv(INPUT_FILE)

print("Original shape:", df.shape)

# ---------------------------------------
# Normalize Venue Names
# ---------------------------------------

print("\nNormalizing venue names...")

def normalize_venue(v):
    if pd.isna(v):
        return v

    v = v.lower()

    replacements = {
        "arun jaitley stadium, delhi": "arun jaitley stadium",
        "wankhede stadium, mumbai": "wankhede stadium",
        "m. a. chidambaram stadium": "m a chidambaram stadium",
        "m chinnaswamy stadium": "m chinnaswamy stadium",
        "eden gardens, kolkata": "eden gardens",
        "narendra modi stadium, ahmedabad": "narendra modi stadium",
        "rajiv gandhi international stadium, uppal": "rajiv gandhi international stadium",
        "punjab cricket association stadium": "pca stadium",
        "punjab cricket association is bindra stadium": "pca stadium"
    }

    return replacements.get(v, v)

df["venue"] = df["venue"].apply(normalize_venue)

# ---------------------------------------
# Build Venue Averages
# ---------------------------------------

print("\nBuilding venue aggregates...")

venue_stats = (
    df.groupby("venue")
    .agg({
        "runs_scored": "mean",
        "powerplay_runs_scored": "mean",
        "death_overs_runs_scored": "mean",
        "wickets_lost": "mean"
    })
    .reset_index()
)

venue_stats = venue_stats.rename(columns={
    "runs_scored": "venue_avg_runs",
    "powerplay_runs_scored": "venue_avg_powerplay",
    "death_overs_runs_scored": "venue_avg_death",
    "wickets_lost": "venue_avg_wickets"
})

# ---------------------------------------
# Checks
# ---------------------------------------

print("\nVenue count:", len(venue_stats))

print("\nPreview:")
print(venue_stats.head())

print("\nRuns range:",
      venue_stats["venue_avg_runs"].min(),
      "to",
      venue_stats["venue_avg_runs"].max())

# ---------------------------------------
# Save
# ---------------------------------------

print("\nSaving to:", OUTPUT_FILE)

venue_stats.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nDone.")
