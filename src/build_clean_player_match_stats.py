from pathlib import Path
import pandas as pd

# ---------------------------------------
# Paths
# ---------------------------------------

PROJECT_DIR = Path("/Users/alay/Desktop/ipl_total_predictior")

INTERIM_DIR = PROJECT_DIR / "data" / "interim"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

PLAYER_FILE = INTERIM_DIR / "raw_player_match_stats.csv"
MATCH_FILE = PROCESSED_DIR / "clean_matches.csv"

OUTPUT_FILE = PROCESSED_DIR / "clean_player_match_stats.csv"

# ---------------------------------------
# Load
# ---------------------------------------

print("Loading datasets...")

player_stats = pd.read_csv(PLAYER_FILE)
clean_matches = pd.read_csv(MATCH_FILE)

print("Player stats shape:", player_stats.shape)
print("Clean matches shape:", clean_matches.shape)

# ---------------------------------------
# Filter Only Clean Matches
# ---------------------------------------

print("\nFiltering player stats to clean matches...")

player_stats = player_stats.merge(
    clean_matches[["match_id"]],
    on="match_id",
    how="inner"
)

print("After match filtering:", player_stats.shape)

# ---------------------------------------
# Remove Impossible Player Records
# ---------------------------------------

print("\nFiltering unrealistic player records...")

player_stats = player_stats[
    player_stats["balls_faced"] >= 0
]

player_stats = player_stats[
    player_stats["runs_scored"] >= 0
]

player_stats = player_stats[
    player_stats["runs_conceded"] >= 0
]

print("After player filtering:", player_stats.shape)

# ---------------------------------------
# Basic Checks
# ---------------------------------------

print("\nRuns scored max:",
      player_stats["runs_scored"].max())

print("Wickets taken max:",
      player_stats["wickets_taken"].max())

print("Unique matches:",
      player_stats["match_id"].nunique())

# ---------------------------------------
# Save
# ---------------------------------------

print("\nSaving to:", OUTPUT_FILE)

player_stats.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nDone.")