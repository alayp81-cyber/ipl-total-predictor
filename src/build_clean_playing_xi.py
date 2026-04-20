from pathlib import Path
import pandas as pd

# ---------------------------------------
# Paths
# ---------------------------------------

PROJECT_DIR = Path("/Users/alay/Desktop/ipl_total_predictior")

INTERIM_DIR = PROJECT_DIR / "data" / "interim"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

PLAYING_XI_FILE = INTERIM_DIR / "raw_playing_xi.csv"
MATCH_FILE = PROCESSED_DIR / "clean_matches.csv"

OUTPUT_FILE = PROCESSED_DIR / "clean_playing_xi.csv"

# ---------------------------------------
# Load
# ---------------------------------------

print("Loading datasets...")

playing_xi = pd.read_csv(PLAYING_XI_FILE)
clean_matches = pd.read_csv(MATCH_FILE)

print("Playing XI shape:", playing_xi.shape)
print("Clean matches shape:", clean_matches.shape)

# ---------------------------------------
# Filter Only Clean Matches
# ---------------------------------------

print("\nFiltering playing XI to clean matches...")

playing_xi = playing_xi.merge(
    clean_matches[["match_id"]],
    on="match_id",
    how="inner"
)

print("After match filtering:", playing_xi.shape)

# ---------------------------------------
# Drop Exact Duplicates
# ---------------------------------------

print("\nDropping duplicate rows...")

before = len(playing_xi)

playing_xi = playing_xi.drop_duplicates()

after = len(playing_xi)

print("Duplicates removed:", before - after)
print("After duplicate removal:", playing_xi.shape)

# ---------------------------------------
# Basic Checks
# ---------------------------------------

players_per_match = playing_xi.groupby("match_id").size()

print("\nPlayers per match summary:")
print(players_per_match.describe())

print("\nUnique matches:", playing_xi["match_id"].nunique())

# ---------------------------------------
# Save
# ---------------------------------------

print("\nSaving to:", OUTPUT_FILE)

playing_xi.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nDone.")