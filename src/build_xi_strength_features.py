from pathlib import Path
import pandas as pd

# ---------------------------------------
# Paths
# ---------------------------------------

PROJECT_DIR = Path("/Users/alay/Desktop/ipl_total_predictior")

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
FEATURE_DIR = PROJECT_DIR / "data" / "features"

PLAYING_XI_FILE = PROCESSED_DIR / "clean_playing_xi.csv"
PLAYER_STATS_FILE = PROCESSED_DIR / "clean_player_match_stats.csv"

OUTPUT_FILE = FEATURE_DIR / "xi_strength_features.csv"

# ---------------------------------------
# Load
# ---------------------------------------

print("Loading datasets...")

playing_xi = pd.read_csv(PLAYING_XI_FILE)
player_stats = pd.read_csv(PLAYER_STATS_FILE)

print("playing_xi shape:", playing_xi.shape)
print("player_stats shape:", player_stats.shape)

# ---------------------------------------
# Date Handling
# ---------------------------------------

playing_xi["match_date"] = pd.to_datetime(playing_xi["match_date"])
player_stats["match_date"] = pd.to_datetime(player_stats["match_date"])

# ---------------------------------------
# Sort Player Stats for Rolling Features
# ---------------------------------------

print("\nBuilding rolling player statistics...")

player_stats = player_stats.sort_values(["player_name", "match_date", "match_id"])

ROLLING_WINDOW = 5

player_stats["bat_runs_avg_last5"] = (
    player_stats.groupby("player_name")["runs_scored"]
    .shift(1)
    .rolling(ROLLING_WINDOW)
    .mean()
)

player_stats["bat_balls_avg_last5"] = (
    player_stats.groupby("player_name")["balls_faced"]
    .shift(1)
    .rolling(ROLLING_WINDOW)
    .mean()
)

player_stats["bat_fours_avg_last5"] = (
    player_stats.groupby("player_name")["fours"]
    .shift(1)
    .rolling(ROLLING_WINDOW)
    .mean()
)

player_stats["bat_sixes_avg_last5"] = (
    player_stats.groupby("player_name")["sixes"]
    .shift(1)
    .rolling(ROLLING_WINDOW)
    .mean()
)

player_stats["bowl_wickets_avg_last5"] = (
    player_stats.groupby("player_name")["wickets_taken"]
    .shift(1)
    .rolling(ROLLING_WINDOW)
    .mean()
)

player_stats["bowl_runs_conceded_avg_last5"] = (
    player_stats.groupby("player_name")["runs_conceded"]
    .shift(1)
    .rolling(ROLLING_WINDOW)
    .mean()
)

player_stats["bowl_dotballs_avg_last5"] = (
    player_stats.groupby("player_name")["dot_balls"]
    .shift(1)
    .rolling(ROLLING_WINDOW)
    .mean()
)

# Keep only needed rolling columns from player stats
player_rollups = player_stats[
    [
        "match_id",
        "player_name",
        "bat_runs_avg_last5",
        "bat_balls_avg_last5",
        "bat_fours_avg_last5",
        "bat_sixes_avg_last5",
        "bowl_wickets_avg_last5",
        "bowl_runs_conceded_avg_last5",
        "bowl_dotballs_avg_last5",
    ]
].copy()

# ---------------------------------------
# Merge Player Stats into Playing XI
# ---------------------------------------

print("\nMerging player stats into XI...")

xi_stats = playing_xi.merge(
    player_rollups,
    on=["match_id", "player_name"],
    how="left"
)

print("After merge:", xi_stats.shape)

# ---------------------------------------
# Aggregate to Team Level
# ---------------------------------------

print("\nAggregating XI strength features...")

xi_features = (
    xi_stats.groupby(["match_id", "team"])
    .agg({
        "bat_runs_avg_last5": "mean",
        "bat_balls_avg_last5": "mean",
        "bat_fours_avg_last5": "mean",
        "bat_sixes_avg_last5": "mean",
        "bowl_wickets_avg_last5": "mean",
        "bowl_runs_conceded_avg_last5": "mean",
        "bowl_dotballs_avg_last5": "mean",
        "player_name": "count"
    })
    .reset_index()
)

xi_features = xi_features.rename(columns={
    "player_name": "xi_player_count"
})

print("xi_features shape:", xi_features.shape)

# ---------------------------------------
# Fill Missing Values
# ---------------------------------------

print("\nFilling missing values...")

xi_features = xi_features.fillna(0)

# ---------------------------------------
# Preview
# ---------------------------------------

print("\nPreview:")
print(xi_features.head())

# ---------------------------------------
# Save
# ---------------------------------------

print("\nSaving to:", OUTPUT_FILE)

xi_features.to_csv(OUTPUT_FILE, index=False)

print("\nDone.")