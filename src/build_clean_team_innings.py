from pathlib import Path
import pandas as pd

# ---------------------------------------
# Paths
# ---------------------------------------

PROJECT_DIR = Path("/Users/alay/Desktop/ipl_total_predictior")

INTERIM_DIR = PROJECT_DIR / "data" / "interim"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

INPUT_FILE = INTERIM_DIR / "raw_team_innings.csv"
MATCHES_FILE = PROCESSED_DIR / "clean_matches.csv"

OUTPUT_FILE = PROCESSED_DIR / "clean_team_innings.csv"

# ---------------------------------------
# Load
# ---------------------------------------

print("Loading datasets...")

team_innings = pd.read_csv(INPUT_FILE)
clean_matches = pd.read_csv(MATCHES_FILE)

print("Team innings shape:", team_innings.shape)
print("Clean matches shape:", clean_matches.shape)

# ---------------------------------------
# Keep Only Clean Matches
# ---------------------------------------

print("\nFiltering team innings to clean matches...")

team_innings = team_innings.merge(
    clean_matches[["match_id"]],
    on="match_id",
    how="inner"
)

print("After match filtering:", team_innings.shape)

# ---------------------------------------
# Remove Unrealistic Scores
# ---------------------------------------

print("\nFiltering unrealistic totals...")

team_innings = team_innings[
    team_innings["runs_scored"] >= 50
]

team_innings = team_innings[
    team_innings["runs_scored"] <= 300
]

print("After score filtering:", team_innings.shape)

# ---------------------------------------
# Final Checks
# ---------------------------------------

print("\nRuns range:",
      team_innings["runs_scored"].min(),
      "to",
      team_innings["runs_scored"].max())

print("Unique matches:",
      team_innings["match_id"].nunique())

# ---------------------------------------
# Save
# ---------------------------------------

print("\nSaving to:", OUTPUT_FILE)

team_innings.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nDone.")