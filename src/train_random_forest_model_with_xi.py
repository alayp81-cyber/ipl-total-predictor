from pathlib import Path
import pandas as pd
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# ---------------------------------------
# Paths
# ---------------------------------------

PROJECT_DIR = Path("/Users/alay/Desktop/ipl_total_predictior")

FEATURE_DIR = PROJECT_DIR / "data" / "features"
MODEL_DIR = PROJECT_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = FEATURE_DIR / "match_feature_view_with_xi.csv"
MODEL_FILE = MODEL_DIR / "random_forest_model_with_xi.pkl"

# ---------------------------------------
# Load
# ---------------------------------------

print("Loading modeling dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset shape:", df.shape)

# ---------------------------------------
# Select Features
# ---------------------------------------

FEATURE_COLUMNS = [
    "season",
    "impact_player_era_flag",

    "team1_batting_first_flag",
    "team1_won_toss_flag",
    "toss_decision_bat_flag",

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
    "venue_wickets_avg_last10",

    "team1_xi_bat_runs_avg_last5",
    "team1_xi_bat_balls_avg_last5",
    "team1_xi_bat_fours_avg_last5",
    "team1_xi_bat_sixes_avg_last5",
    "team1_xi_bowl_wickets_avg_last5",
    "team1_xi_bowl_runs_conceded_avg_last5",
    "team1_xi_bowl_dotballs_avg_last5",
    "team1_xi_player_count",

    "team2_xi_bat_runs_avg_last5",
    "team2_xi_bat_balls_avg_last5",
    "team2_xi_bat_fours_avg_last5",
    "team2_xi_bat_sixes_avg_last5",
    "team2_xi_bowl_wickets_avg_last5",
    "team2_xi_bowl_runs_conceded_avg_last5",
    "team2_xi_bowl_dotballs_avg_last5",
    "team2_xi_player_count"
]

TARGET_COLUMN = "match_total_runs"

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

# ---------------------------------------
# Train/Test Split
# ---------------------------------------

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# ---------------------------------------
# Train Model
# ---------------------------------------

print("\nTraining Random Forest model with XI features...")

model = RandomForestRegressor(
    n_estimators=500,
    max_depth=10,
    min_samples_split=8,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ---------------------------------------
# Predict
# ---------------------------------------

print("\nMaking predictions...")

y_pred = model.predict(X_test)

# ---------------------------------------
# Evaluate
# ---------------------------------------

mae = mean_absolute_error(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Absolute Error:", round(mae, 2))

# ---------------------------------------
# Feature Importance
# ---------------------------------------

importance_df = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\nTop 20 Feature Importances:")
print(importance_df.head(20))

# ---------------------------------------
# Save Model
# ---------------------------------------

with open(MODEL_FILE, "wb") as f:
    pickle.dump(model, f)

print("\nModel saved to:", MODEL_FILE)

print("\nDone.")