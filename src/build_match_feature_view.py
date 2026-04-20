from pathlib import Path
import pandas as pd

# ---------------------------------------
# Paths
# ---------------------------------------

PROJECT_DIR = Path("/Users/alay/Desktop/ipl_total_predictior")

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
FEATURE_DIR = PROJECT_DIR / "data" / "features"

MATCH_FILE = PROCESSED_DIR / "clean_matches.csv"
TEAM_FORM_FILE = FEATURE_DIR / "team_form_features.csv"
VENUE_FILE = FEATURE_DIR / "venue_features.csv"
TEAM_INNINGS_FILE = PROCESSED_DIR / "clean_team_innings.csv"

OUTPUT_FILE = FEATURE_DIR / "match_feature_view.csv"

# ---------------------------------------
# Helpers
# ---------------------------------------

def normalize_venue(v):
    if pd.isna(v):
        return v

    v = str(v).lower().strip()

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


# ---------------------------------------
# Load
# ---------------------------------------

print("Loading datasets...")

matches = pd.read_csv(MATCH_FILE)
team_form = pd.read_csv(TEAM_FORM_FILE)
venue_features = pd.read_csv(VENUE_FILE)
team_innings = pd.read_csv(TEAM_INNINGS_FILE)

print("clean_matches shape:", matches.shape)
print("team_form_features shape:", team_form.shape)
print("venue_features shape:", venue_features.shape)
print("clean_team_innings shape:", team_innings.shape)

# ---------------------------------------
# Date Handling
# ---------------------------------------

matches["match_date"] = pd.to_datetime(matches["match_date"])
team_form["match_date"] = pd.to_datetime(team_form["match_date"])
team_innings["match_date"] = pd.to_datetime(team_innings["match_date"])

# ---------------------------------------
# Normalize Venue Names
# ---------------------------------------

matches["venue"] = matches["venue"].apply(normalize_venue)
venue_features["venue"] = venue_features["venue"].apply(normalize_venue)
team_innings["venue"] = team_innings["venue"].apply(normalize_venue)

# ---------------------------------------
# Derived Match Flags
# ---------------------------------------

print("\nCreating match-level derived fields...")

matches["impact_player_era_flag"] = (matches["season"] >= 2023).astype(int)

matches["team1_batting_first_flag"] = (
    matches["innings1_team"] == matches["team1"]
).astype(int)

matches["team2_batting_first_flag"] = (
    matches["innings1_team"] == matches["team2"]
).astype(int)

matches["team1_won_toss_flag"] = (
    matches["toss_winner"] == matches["team1"]
).astype(int)

matches["team2_won_toss_flag"] = (
    matches["toss_winner"] == matches["team2"]
).astype(int)

matches["team1_chasing_flag"] = (
    matches["innings2_team"] == matches["team1"]
).astype(int)

matches["team2_chasing_flag"] = (
    matches["innings2_team"] == matches["team2"]
).astype(int)

matches["toss_decision_bat_flag"] = (
    matches["toss_decision"].fillna("").str.lower() == "bat"
).astype(int)

matches["toss_decision_field_flag"] = (
    matches["toss_decision"].fillna("").str.lower() == "field"
).astype(int)

# ---------------------------------------
# Venue Trend Features
# prior-only venue rolling run environment
# ---------------------------------------

print("\nBuilding venue trend features...")

venue_trend = team_innings.copy()

venue_trend = venue_trend.sort_values(["venue", "match_date", "match_id"])

venue_trend["venue_runs_avg_last10"] = (
    venue_trend.groupby("venue")["runs_scored"]
    .shift(1)
    .rolling(10)
    .mean()
)

venue_trend["venue_powerplay_avg_last10"] = (
    venue_trend.groupby("venue")["powerplay_runs_scored"]
    .shift(1)
    .rolling(10)
    .mean()
)

venue_trend["venue_death_avg_last10"] = (
    venue_trend.groupby("venue")["death_overs_runs_scored"]
    .shift(1)
    .rolling(10)
    .mean()
)

venue_trend["venue_wickets_avg_last10"] = (
    venue_trend.groupby("venue")["wickets_lost"]
    .shift(1)
    .rolling(10)
    .mean()
)

venue_trend = venue_trend[[
    "match_id",
    "venue",
    "venue_runs_avg_last10",
    "venue_powerplay_avg_last10",
    "venue_death_avg_last10",
    "venue_wickets_avg_last10"
]].drop_duplicates(subset=["match_id"])

print("venue_trend shape:", venue_trend.shape)

# ---------------------------------------
# Team 1 Merge
# ---------------------------------------

print("\nMerging team1 form features...")

team1_form = team_form.copy()

team1_form = team1_form.rename(columns={
    "team": "team1",
    "team_runs_avg_last5": "team1_runs_avg_last5",
    "team_wickets_lost_avg_last5": "team1_wickets_lost_avg_last5",
    "team_powerplay_avg_last5": "team1_powerplay_avg_last5",
    "team_death_avg_last5": "team1_death_avg_last5",
    "team_runs_conceded_avg_last5": "team1_runs_conceded_avg_last5"
})

team1_keep = [
    "match_id",
    "team1",
    "team1_runs_avg_last5",
    "team1_wickets_lost_avg_last5",
    "team1_powerplay_avg_last5",
    "team1_death_avg_last5",
    "team1_runs_conceded_avg_last5"
]

matches = matches.merge(
    team1_form[team1_keep],
    on=["match_id", "team1"],
    how="left"
)

print("After team1 merge:", matches.shape)

# ---------------------------------------
# Team 2 Merge
# ---------------------------------------

print("\nMerging team2 form features...")

team2_form = team_form.copy()

team2_form = team2_form.rename(columns={
    "team": "team2",
    "team_runs_avg_last5": "team2_runs_avg_last5",
    "team_wickets_lost_avg_last5": "team2_wickets_lost_avg_last5",
    "team_powerplay_avg_last5": "team2_powerplay_avg_last5",
    "team_death_avg_last5": "team2_death_avg_last5",
    "team_runs_conceded_avg_last5": "team2_runs_conceded_avg_last5"
})

team2_keep = [
    "match_id",
    "team2",
    "team2_runs_avg_last5",
    "team2_wickets_lost_avg_last5",
    "team2_powerplay_avg_last5",
    "team2_death_avg_last5",
    "team2_runs_conceded_avg_last5"
]

matches = matches.merge(
    team2_form[team2_keep],
    on=["match_id", "team2"],
    how="left"
)

print("After team2 merge:", matches.shape)

# ---------------------------------------
# Venue Static Merge
# ---------------------------------------

print("\nMerging static venue features...")

matches = matches.merge(
    venue_features,
    on="venue",
    how="left"
)

print("After static venue merge:", matches.shape)

# ---------------------------------------
# Venue Trend Merge
# ---------------------------------------

print("\nMerging venue trend features...")

matches = matches.merge(
    venue_trend,
    on=["match_id", "venue"],
    how="left"
)

print("After venue trend merge:", matches.shape)

# ---------------------------------------
# Drop rows without usable history
# ---------------------------------------

print("\nDropping rows without usable history...")

before = len(matches)

required_cols = [
    "team1_runs_avg_last5",
    "team2_runs_avg_last5",
    "venue_avg_runs"
]

matches = matches.dropna(subset=required_cols)

after = len(matches)

print("Rows removed:", before - after)
print("Rows retained:", after)

# ---------------------------------------
# Fill Missing Venue Trend Values
# ---------------------------------------

print("\nFilling missing venue trend values...")

matches["venue_runs_avg_last10"] = matches[
    "venue_runs_avg_last10"
].fillna(matches["venue_avg_runs"])

matches["venue_powerplay_avg_last10"] = matches[
    "venue_powerplay_avg_last10"
].fillna(matches["venue_avg_powerplay"])

matches["venue_death_avg_last10"] = matches[
    "venue_death_avg_last10"
].fillna(matches["venue_avg_death"])

matches["venue_wickets_avg_last10"] = matches[
    "venue_wickets_avg_last10"
].fillna(matches["venue_avg_wickets"])

# ---------------------------------------
# Final columns
# ---------------------------------------

final_columns = [
    "match_id",
    "season",
    "match_date",
    "venue",
    "city",
    "team1",
    "team2",
    "toss_winner",
    "toss_decision",
    "innings1_team",
    "innings2_team",
    "innings1_runs",
    "innings2_runs",
    "match_total_runs",

    "impact_player_era_flag",
    "team1_batting_first_flag",
    "team2_batting_first_flag",
    "team1_won_toss_flag",
    "team2_won_toss_flag",
    "team1_chasing_flag",
    "team2_chasing_flag",
    "toss_decision_bat_flag",
    "toss_decision_field_flag",

    "team1_runs_avg_last5",
    "team1_wickets_lost_avg_last5",
    "team1_powerplay_avg_last5",
    "team1_death_avg_last5",
    "team1_runs_conceded_avg_last5",

    "team2_runs_avg_last5",
    "team2_wickets_lost_avg_last5",
    "team2_powerplay_avg_last5",
    "team2_death_avg_last5",
    "team2_runs_conceded_avg_last5",

    "venue_avg_runs",
    "venue_avg_powerplay",
    "venue_avg_death",
    "venue_avg_wickets",

    "venue_runs_avg_last10",
    "venue_powerplay_avg_last10",
    "venue_death_avg_last10",
    "venue_wickets_avg_last10"
]

matches = matches[final_columns]

# ---------------------------------------
# Final checks
# ---------------------------------------

print("\nPreview:")
print(matches.head())

print("\nFinal shape:")
print(matches.shape)

print("\nSeason range:",
      matches["season"].min(),
      "to",
      matches["season"].max())

print("\nMissing values by column:")
print(matches.isna().sum())

# ---------------------------------------
# Save
# ---------------------------------------

print("\nSaving to:", OUTPUT_FILE)

matches.to_csv(OUTPUT_FILE, index=False)

print("\nDone.")