from pathlib import Path
import pandas as pd
import pickle

# ---------------------------------------
# Paths
# ---------------------------------------

PROJECT_DIR = Path("/Users/alay/Desktop/ipl_total_predictior")

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
FEATURE_DIR = PROJECT_DIR / "data" / "features"
MODEL_DIR = PROJECT_DIR / "models"

MATCHES_FILE = PROCESSED_DIR / "clean_matches.csv"
TEAM_FORM_FILE = FEATURE_DIR / "team_form_features.csv"
VENUE_FILE = FEATURE_DIR / "venue_features.csv"
XI_FILE = FEATURE_DIR / "xi_strength_features.csv"

MODEL_FILE = MODEL_DIR / "catboost_time_split_model.pkl"

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


def get_latest_team_form(team_form_df, team_name):
    team_df = team_form_df[team_form_df["team"] == team_name].copy()

    if team_df.empty:
        raise ValueError(f"No team form history found for: {team_name}")

    team_df = team_df.sort_values(["match_date", "match_id"])
    return team_df.iloc[-1]


def get_venue_features(venue_df, venue_name):
    venue_name = normalize_venue(venue_name)

    row = venue_df[venue_df["venue"] == venue_name].copy()

    if row.empty:
        raise ValueError(f"No venue features found for: {venue_name}")

    return row.iloc[0]


def get_latest_xi_strength(xi_df, team_name):
    team_df = xi_df[xi_df["team"] == team_name].copy()

    if team_df.empty:
        raise ValueError(f"No XI strength history found for: {team_name}")

    team_df = team_df.sort_values("match_id")
    return team_df.iloc[-1]


# ---------------------------------------
# Load Data + Model
# ---------------------------------------

print("Loading model and feature tables...")

team_form = pd.read_csv(TEAM_FORM_FILE)
venue_features = pd.read_csv(VENUE_FILE)
xi_features = pd.read_csv(XI_FILE)

team_form["match_date"] = pd.to_datetime(team_form["match_date"])

venue_features["venue"] = venue_features["venue"].apply(normalize_venue)

with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

print("Loaded successfully.")

# ---------------------------------------
# Manual Input
# ---------------------------------------

print("\nEnter match details...\n")

season = int(input("Season (e.g. 2026): ").strip())
team1 = input("Team 1: ").strip()
team2 = input("Team 2: ").strip()
venue = input("Venue: ").strip()
toss_winner = input("Toss winner: ").strip()
toss_decision = input("Toss decision (bat/field): ").strip().lower()

venue = normalize_venue(venue)

# ---------------------------------------
# Derived Match Flags
# ---------------------------------------

team1_won_toss_flag = int(toss_winner == team1)
team1_batting_first_flag = int(
    (team1_won_toss_flag == 1 and toss_decision == "bat") or
    (team1_won_toss_flag == 0 and toss_decision == "field")
)

impact_player_era_flag = int(season >= 2023)
toss_decision_bat_flag = int(toss_decision == "bat")

# ---------------------------------------
# Fetch Latest Historical Features
# ---------------------------------------

team1_form = get_latest_team_form(team_form, team1)
team2_form = get_latest_team_form(team_form, team2)

venue_row = get_venue_features(venue_features, venue)

team1_xi = get_latest_xi_strength(xi_features, team1)
team2_xi = get_latest_xi_strength(xi_features, team2)

# ---------------------------------------
# Build Prediction Row
# ---------------------------------------

prediction_row = pd.DataFrame([{
    "team1": team1,
    "team2": team2,
    "venue": venue,
    "toss_winner": toss_winner,
    "toss_decision": toss_decision,

    "season": season,
    "impact_player_era_flag": impact_player_era_flag,

    "team1_batting_first_flag": team1_batting_first_flag,
    "team1_won_toss_flag": team1_won_toss_flag,
    "toss_decision_bat_flag": toss_decision_bat_flag,

    "team1_runs_avg_last5": team1_form["team_runs_avg_last5"],
    "team1_wickets_lost_avg_last5": team1_form["team_wickets_lost_avg_last5"],
    "team1_powerplay_avg_last5": team1_form["team_powerplay_avg_last5"],
    "team1_death_avg_last5": team1_form["team_death_avg_last5"],
    "team1_runs_conceded_avg_last5": team1_form["team_runs_conceded_avg_last5"],

    "team2_runs_avg_last5": team2_form["team_runs_avg_last5"],
    "team2_wickets_lost_avg_last5": team2_form["team_wickets_lost_avg_last5"],
    "team2_powerplay_avg_last5": team2_form["team_powerplay_avg_last5"],
    "team2_death_avg_last5": team2_form["team_death_avg_last5"],
    "team2_runs_conceded_avg_last5": team2_form["team_runs_conceded_avg_last5"],

    "venue_avg_runs": venue_row["venue_avg_runs"],
    "venue_avg_powerplay": venue_row["venue_avg_powerplay"],
    "venue_avg_death": venue_row["venue_avg_death"],
    "venue_avg_wickets": venue_row["venue_avg_wickets"],

    # fallback: use static venue values since live prior-to-match venue trend table
    # is not yet separately built for manual inference
    "venue_runs_avg_last10": venue_row["venue_avg_runs"],
    "venue_powerplay_avg_last10": venue_row["venue_avg_powerplay"],
    "venue_death_avg_last10": venue_row["venue_avg_death"],
    "venue_wickets_avg_last10": venue_row["venue_avg_wickets"],

    "team1_xi_bat_runs_avg_last5": team1_xi["bat_runs_avg_last5"],
    "team1_xi_bat_balls_avg_last5": team1_xi["bat_balls_avg_last5"],
    "team1_xi_bat_fours_avg_last5": team1_xi["bat_fours_avg_last5"],
    "team1_xi_bat_sixes_avg_last5": team1_xi["bat_sixes_avg_last5"],
    "team1_xi_bowl_wickets_avg_last5": team1_xi["bowl_wickets_avg_last5"],
    "team1_xi_bowl_runs_conceded_avg_last5": team1_xi["bowl_runs_conceded_avg_last5"],
    "team1_xi_bowl_dotballs_avg_last5": team1_xi["bowl_dotballs_avg_last5"],
    "team1_xi_player_count": team1_xi["xi_player_count"],

    "team2_xi_bat_runs_avg_last5": team2_xi["bat_runs_avg_last5"],
    "team2_xi_bat_balls_avg_last5": team2_xi["bat_balls_avg_last5"],
    "team2_xi_bat_fours_avg_last5": team2_xi["bat_fours_avg_last5"],
    "team2_xi_bat_sixes_avg_last5": team2_xi["bat_sixes_avg_last5"],
    "team2_xi_bowl_wickets_avg_last5": team2_xi["bowl_wickets_avg_last5"],
    "team2_xi_bowl_runs_conceded_avg_last5": team2_xi["bowl_runs_conceded_avg_last5"],
    "team2_xi_bowl_dotballs_avg_last5": team2_xi["bowl_dotballs_avg_last5"],
    "team2_xi_player_count": team2_xi["xi_player_count"],
}])

# Ensure categoricals are strings
for col in ["team1", "team2", "venue", "toss_winner", "toss_decision"]:
    prediction_row[col] = prediction_row[col].astype(str)

# ---------------------------------------
# Predict
# ---------------------------------------

predicted_total = model.predict(prediction_row)[0]

print("\n---------------------------------------")
print("Predicted Match Total Runs:", round(predicted_total, 2))
print("---------------------------------------")