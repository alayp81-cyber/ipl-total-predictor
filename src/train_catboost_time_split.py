from pathlib import Path
import pandas as pd
import pickle

from sklearn.metrics import mean_absolute_error

from catboost import CatBoostRegressor

# ---------------------------------------
# Paths
# ---------------------------------------

PROJECT_DIR = Path("/Users/alay/Desktop/ipl_total_predictior")

FEATURE_DIR = PROJECT_DIR / "data" / "features"
MODEL_DIR = PROJECT_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = FEATURE_DIR / "match_feature_view_with_xi.csv"
MODEL_FILE = MODEL_DIR / "catboost_time_split_model.pkl"

# ---------------------------------------
# Load
# ---------------------------------------

print("Loading modeling dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset shape:", df.shape)

# ---------------------------------------
# Time-based Split
# ---------------------------------------

print("\nCreating time-based split...")

train_df = df[df["season"] <= 2023].copy()
test_df = df[df["season"] >= 2024].copy()

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# ---------------------------------------
# Feature Selection
# ---------------------------------------

FEATURE_COLUMNS = [
    # categorical
    "team1",
    "team2",
    "venue",
    "toss_winner",
    "toss_decision",

    # structured numeric
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
    "team2_xi_player_count",
]

TARGET_COLUMN = "match_total_runs"

CATEGORICAL_COLUMNS = [
    "team1",
    "team2",
    "venue",
    "toss_winner",
    "toss_decision",
]

X_train = train_df[FEATURE_COLUMNS].copy()
y_train = train_df[TARGET_COLUMN].copy()

X_test = test_df[FEATURE_COLUMNS].copy()
y_test = test_df[TARGET_COLUMN].copy()

# Ensure categorical columns are strings
for col in CATEGORICAL_COLUMNS:
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)

cat_feature_indices = [X_train.columns.get_loc(col) for col in CATEGORICAL_COLUMNS]

# ---------------------------------------
# Train Model
# ---------------------------------------

print("\nTraining CatBoost model...")

model = CatBoostRegressor(
    iterations=800,
    depth=6,
    learning_rate=0.03,
    loss_function="MAE",
    eval_metric="MAE",
    random_seed=42,
    verbose=100
)

model.fit(
    X_train,
    y_train,
    cat_features=cat_feature_indices,
    eval_set=(X_test, y_test),
    use_best_model=True
)

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
    "importance": model.get_feature_importance()
}).sort_values("importance", ascending=False)

print("\nTop 25 Feature Importances:")
print(importance_df.head(25))

# ---------------------------------------
# Save Model
# ---------------------------------------

with open(MODEL_FILE, "wb") as f:
    pickle.dump(model, f)

print("\nModel saved to:", MODEL_FILE)

print("\nDone.")