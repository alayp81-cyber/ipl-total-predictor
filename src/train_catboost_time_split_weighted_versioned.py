from pathlib import Path
import pandas as pd
import pickle
import re

from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor

# ---------------------------------------
# Paths
# ---------------------------------------

PROJECT_DIR = Path("/Users/alay/Desktop/ipl_total_predictior")

FEATURE_DIR = PROJECT_DIR / "data" / "features"
MODEL_DIR = PROJECT_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = FEATURE_DIR / "match_feature_view_with_xi_weighted.csv"

LATEST_MODEL_FILE = MODEL_DIR / "latest_model.pkl"

# ---------------------------------------
# Version Finder
# ---------------------------------------

def get_next_model_version():

    existing_models = list(
        MODEL_DIR.glob("catboost_model_v*.pkl")
    )

    version_numbers = []

    for model_path in existing_models:

        match = re.search(
            r"catboost_model_v(\d+)\.pkl",
            model_path.name
        )

        if match:
            version_numbers.append(
                int(match.group(1))
            )

    if len(version_numbers) == 0:
        return 1

    return max(version_numbers) + 1


# ---------------------------------------
# Load Dataset
# ---------------------------------------

print("Loading weighted dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset shape:", df.shape)

# ---------------------------------------
# Time Split
# ---------------------------------------

train_df = df[df["season"] <= 2023]
test_df = df[df["season"] >= 2024]

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# ---------------------------------------
# Feature Setup
# ---------------------------------------

FEATURE_COLUMNS = [
    "team1",
    "team2",
    "venue",
    "toss_winner",
    "toss_decision",
    "season",
    "impact_player_era_flag",
    "team1_batting_first_flag",
    "team1_won_toss_flag",
    "toss_decision_bat_flag",

    "team1_runs_avg_last5",
    "team1_powerplay_avg_last5",
    "team1_death_avg_last5",
    "team1_runs_conceded_avg_last5",

    "team2_runs_avg_last5",
    "team2_powerplay_avg_last5",
    "team2_death_avg_last5",
    "team2_runs_conceded_avg_last5",

    "venue_avg_runs",
    "venue_avg_powerplay",
    "venue_avg_death",

    "venue_runs_avg_last10",
    "venue_powerplay_avg_last10",
    "venue_death_avg_last10",

    "team1_xi_bat_runs_avg_last5",
    "team1_xi_bat_sixes_avg_last5",
    "team1_xi_bowl_wickets_avg_last5",

    "team2_xi_bat_runs_avg_last5",
    "team2_xi_bat_sixes_avg_last5",
    "team2_xi_bowl_wickets_avg_last5",
]

TARGET_COLUMN = "match_total_runs"
WEIGHT_COLUMN = "recency_weight"

CATEGORICAL_COLUMNS = [
    "team1",
    "team2",
    "venue",
    "toss_winner",
    "toss_decision"
]

X_train = train_df[FEATURE_COLUMNS].copy()
y_train = train_df[TARGET_COLUMN].copy()
w_train = train_df[WEIGHT_COLUMN].copy()

X_test = test_df[FEATURE_COLUMNS].copy()
y_test = test_df[TARGET_COLUMN].copy()

for col in CATEGORICAL_COLUMNS:
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)

cat_feature_indices = [
    X_train.columns.get_loc(col)
    for col in CATEGORICAL_COLUMNS
]

# ---------------------------------------
# Train Model
# ---------------------------------------

print("\nTraining model...")

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
    sample_weight=w_train,
    cat_features=cat_feature_indices,
    eval_set=(X_test, y_test),
    use_best_model=True
)

# ---------------------------------------
# Evaluate
# ---------------------------------------

preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)

print("\nModel MAE:", round(mae, 2))

# ---------------------------------------
# Save Versioned Model
# ---------------------------------------

version = get_next_model_version()

versioned_file = MODEL_DIR / f"catboost_model_v{version}.pkl"

with open(versioned_file, "wb") as f:
    pickle.dump(model, f)

print("Saved version:", versioned_file)

# ---------------------------------------
# Update latest_model.pkl
# ---------------------------------------

with open(LATEST_MODEL_FILE, "wb") as f:
    pickle.dump(model, f)

print("Updated latest_model.pkl")

print("\nDone.")