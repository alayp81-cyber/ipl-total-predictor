from pathlib import Path
import pandas as pd

# ---------------------------------------
# Paths
# ---------------------------------------

PROJECT_DIR = Path("/Users/alay/Desktop/ipl_total_predictior")

FEATURE_DIR = PROJECT_DIR / "data" / "features"

MATCH_FEATURE_FILE = FEATURE_DIR / "match_feature_view.csv"
XI_FEATURE_FILE = FEATURE_DIR / "xi_strength_features.csv"

OUTPUT_FILE = FEATURE_DIR / "match_feature_view_with_xi.csv"

# ---------------------------------------
# Load
# ---------------------------------------

print("Loading datasets...")

matches = pd.read_csv(MATCH_FEATURE_FILE)
xi_features = pd.read_csv(XI_FEATURE_FILE)

print("match_feature_view shape:", matches.shape)
print("xi_strength_features shape:", xi_features.shape)

# ---------------------------------------
# Team 1 XI Merge
# ---------------------------------------

print("\nMerging team1 XI features...")

team1_xi = xi_features.copy()

team1_xi = team1_xi.rename(columns={
    "team": "team1",
    "bat_runs_avg_last5": "team1_xi_bat_runs_avg_last5",
    "bat_balls_avg_last5": "team1_xi_bat_balls_avg_last5",
    "bat_fours_avg_last5": "team1_xi_bat_fours_avg_last5",
    "bat_sixes_avg_last5": "team1_xi_bat_sixes_avg_last5",
    "bowl_wickets_avg_last5": "team1_xi_bowl_wickets_avg_last5",
    "bowl_runs_conceded_avg_last5": "team1_xi_bowl_runs_conceded_avg_last5",
    "bowl_dotballs_avg_last5": "team1_xi_bowl_dotballs_avg_last5",
    "xi_player_count": "team1_xi_player_count"
})

team1_keep = [
    "match_id",
    "team1",
    "team1_xi_bat_runs_avg_last5",
    "team1_xi_bat_balls_avg_last5",
    "team1_xi_bat_fours_avg_last5",
    "team1_xi_bat_sixes_avg_last5",
    "team1_xi_bowl_wickets_avg_last5",
    "team1_xi_bowl_runs_conceded_avg_last5",
    "team1_xi_bowl_dotballs_avg_last5",
    "team1_xi_player_count"
]

matches = matches.merge(
    team1_xi[team1_keep],
    on=["match_id", "team1"],
    how="left"
)

print("After team1 XI merge:", matches.shape)

# ---------------------------------------
# Team 2 XI Merge
# ---------------------------------------

print("\nMerging team2 XI features...")

team2_xi = xi_features.copy()

team2_xi = team2_xi.rename(columns={
    "team": "team2",
    "bat_runs_avg_last5": "team2_xi_bat_runs_avg_last5",
    "bat_balls_avg_last5": "team2_xi_bat_balls_avg_last5",
    "bat_fours_avg_last5": "team2_xi_bat_fours_avg_last5",
    "bat_sixes_avg_last5": "team2_xi_bat_sixes_avg_last5",
    "bowl_wickets_avg_last5": "team2_xi_bowl_wickets_avg_last5",
    "bowl_runs_conceded_avg_last5": "team2_xi_bowl_runs_conceded_avg_last5",
    "bowl_dotballs_avg_last5": "team2_xi_bowl_dotballs_avg_last5",
    "xi_player_count": "team2_xi_player_count"
})

team2_keep = [
    "match_id",
    "team2",
    "team2_xi_bat_runs_avg_last5",
    "team2_xi_bat_balls_avg_last5",
    "team2_xi_bat_fours_avg_last5",
    "team2_xi_bat_sixes_avg_last5",
    "team2_xi_bowl_wickets_avg_last5",
    "team2_xi_bowl_runs_conceded_avg_last5",
    "team2_xi_bowl_dotballs_avg_last5",
    "team2_xi_player_count"
]

matches = matches.merge(
    team2_xi[team2_keep],
    on=["match_id", "team2"],
    how="left"
)

print("After team2 XI merge:", matches.shape)

# ---------------------------------------
# Fill Missing XI Values
# ---------------------------------------

print("\nFilling missing XI values...")

xi_cols = [
    col for col in matches.columns
    if "_xi_" in col
]

matches[xi_cols] = matches[xi_cols].fillna(0)

# ---------------------------------------
# Preview / Checks
# ---------------------------------------

print("\nPreview:")
print(matches.head())

print("\nFinal shape:")
print(matches.shape)

print("\nMissing values in XI columns:")
print(matches[xi_cols].isna().sum().sum())

# ---------------------------------------
# Save
# ---------------------------------------

print("\nSaving to:", OUTPUT_FILE)

matches.to_csv(OUTPUT_FILE, index=False)

print("\nDone.")