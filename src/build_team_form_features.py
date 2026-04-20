from pathlib import Path
import pandas as pd

# ---------------------------------------
# Paths
# ---------------------------------------

PROJECT_DIR = Path("/Users/alay/Desktop/ipl_total_predictior")

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
FEATURE_DIR = PROJECT_DIR / "data" / "features"

FEATURE_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = PROCESSED_DIR / "clean_team_innings.csv"

OUTPUT_FILE = FEATURE_DIR / "team_form_features.csv"

# ---------------------------------------
# Load
# ---------------------------------------

print("Loading team innings...")

df = pd.read_csv(INPUT_FILE)

print("Original shape:", df.shape)

# ---------------------------------------
# Sort for Rolling Windows
# ---------------------------------------

df["match_date"] = pd.to_datetime(df["match_date"])

df = df.sort_values(
    ["team", "match_date"]
)

# ---------------------------------------
# Rolling Features
# ---------------------------------------

print("\nBuilding rolling features...")

ROLLING_WINDOW = 5

df["team_runs_avg_last5"] = (
    df.groupby("team")["runs_scored"]
    .shift(1)
    .rolling(ROLLING_WINDOW)
    .mean()
)

df["team_wickets_lost_avg_last5"] = (
    df.groupby("team")["wickets_lost"]
    .shift(1)
    .rolling(ROLLING_WINDOW)
    .mean()
)

df["team_powerplay_avg_last5"] = (
    df.groupby("team")["powerplay_runs_scored"]
    .shift(1)
    .rolling(ROLLING_WINDOW)
    .mean()
)

df["team_death_avg_last5"] = (
    df.groupby("team")["death_overs_runs_scored"]
    .shift(1)
    .rolling(ROLLING_WINDOW)
    .mean()
)

df["team_runs_conceded_avg_last5"] = (
    df.groupby("team")["runs_conceded"]
    .shift(1)
    .rolling(ROLLING_WINDOW)
    .mean()
)

# ---------------------------------------
# Remove Early Matches Without History
# ---------------------------------------

print("\nDropping rows without history...")

before = len(df)

df = df.dropna()

after = len(df)

print("Rows removed:", before - after)

# ---------------------------------------
# Save
# ---------------------------------------

print("\nSaving to:", OUTPUT_FILE)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nDone.")