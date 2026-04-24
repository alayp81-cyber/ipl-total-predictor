from pathlib import Path
import json

# ---------------------------------------
# Project Paths
# ---------------------------------------

PROJECT_DIR = Path("/Users/alay/Desktop/ipl_total_predictior")

JSON_DIR = PROJECT_DIR / "data" / "raw" / "cricsheet_json"

STATE_DIR = PROJECT_DIR / "models"
STATE_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE = STATE_DIR / "training_state.json"

RETRAIN_THRESHOLD = 10

# ---------------------------------------
# Count JSON Matches
# ---------------------------------------

def count_json_matches():

    json_files = list(JSON_DIR.glob("*.json"))

    return len(json_files)


# ---------------------------------------
# Load Previous State
# ---------------------------------------

def load_previous_count():

    if not STATE_FILE.exists():

        return 0

    with open(STATE_FILE, "r") as f:

        state = json.load(f)

    return state.get("last_match_count", 0)


# ---------------------------------------
# Save New State
# ---------------------------------------

def save_current_count(count):

    state = {
        "last_match_count": count
    }

    with open(STATE_FILE, "w") as f:

        json.dump(state, f, indent=4)


# ---------------------------------------
# Decide Whether to Retrain
# ---------------------------------------

def main():

    current_count = count_json_matches()

    previous_count = load_previous_count()

    new_matches = current_count - previous_count

    print("\nMatch Count Status")
    print("-------------------")

    print("Previous match count:", previous_count)
    print("Current match count:", current_count)
    print("New matches added:", new_matches)

    if new_matches >= RETRAIN_THRESHOLD:

        print("\nRetraining required.")

        save_current_count(current_count)

        exit(1)   # signal retrain needed

    else:

        print("\nRetraining NOT required.")

        exit(0)   # signal skip


# ---------------------------------------

if __name__ == "__main__":

    main()