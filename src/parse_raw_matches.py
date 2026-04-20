from pathlib import Path
import json
import pandas as pd

# ---------------------------------------
# Project Paths
# ---------------------------------------

PROJECT_DIR = Path("/Users/alay/Desktop/ipl_total_predictior")

RAW_JSON_DIR = PROJECT_DIR / "data" / "raw" / "cricsheet_json"
OUTPUT_DIR = PROJECT_DIR / "data" / "interim"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "raw_matches.csv"

# ---------------------------------------
# Helper Functions
# ---------------------------------------

def extract_match_id(file_path):
    return file_path.stem


def extract_season(date_str):
    return int(str(date_str)[:4]) if date_str else None


def safe_get(d, key, default=None):
    if isinstance(d, dict):
        return d.get(key, default)
    return default


# ---------------------------------------
# Main Parsing Function
# ---------------------------------------

def parse_raw_matches():
    json_files = sorted(RAW_JSON_DIR.glob("*.json"))

    print(f"Found {len(json_files)} match files")

    rows = []

    for file in json_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                match_json = json.load(f)

            info = match_json.get("info", {})

            match_id = extract_match_id(file)

            dates = safe_get(info, "dates", [])
            match_date = dates[0] if dates else None
            season = extract_season(match_date)

            venue = safe_get(info, "venue")
            city = safe_get(info, "city")

            teams = safe_get(info, "teams", [])
            team1 = teams[0] if len(teams) > 0 else None
            team2 = teams[1] if len(teams) > 1 else None

            toss_info = safe_get(info, "toss", {})
            toss_winner = safe_get(toss_info, "winner")
            toss_decision = safe_get(toss_info, "decision")

            outcome = safe_get(info, "outcome", {})
            winner = safe_get(outcome, "winner")

            # ---------------------------------------
            # Extract innings totals
            # ---------------------------------------

            innings = match_json.get("innings", [])

            innings1_runs = None
            innings2_runs = None

            innings1_team = None
            innings2_team = None

            if len(innings) >= 1:
                inn1 = innings[0]
                innings1_team = inn1.get("team")
                overs1 = inn1.get("overs", [])

                innings1_runs = sum(
                    delivery["runs"]["total"]
                    for over in overs1
                    for delivery in over.get("deliveries", [])
                )

            if len(innings) >= 2:
                inn2 = innings[1]
                innings2_team = inn2.get("team")
                overs2 = inn2.get("overs", [])

                innings2_runs = sum(
                    delivery["runs"]["total"]
                    for over in overs2
                    for delivery in over.get("deliveries", [])
                )

            if innings1_runs is not None and innings2_runs is not None:
                match_total_runs = innings1_runs + innings2_runs
            else:
                match_total_runs = None

            completed_flag = (
                innings1_runs is not None and
                innings2_runs is not None
            )

            super_over_flag = len(innings) > 2

            row = {
                "match_id": match_id,
                "season": season,
                "match_date": match_date,
                "venue": venue,
                "city": city,
                "team1": team1,
                "team2": team2,
                "toss_winner": toss_winner,
                "toss_decision": toss_decision,
                "innings1_team": innings1_team,
                "innings2_team": innings2_team,
                "innings1_runs": innings1_runs,
                "innings2_runs": innings2_runs,
                "match_total_runs": match_total_runs,
                "winner": winner,
                "completed_flag": completed_flag,
                "super_over_flag": super_over_flag,
            }

            rows.append(row)

        except Exception as e:
            print(f"Error parsing {file.name}: {e}")

    df = pd.DataFrame(rows)

    print("\nPreview:")
    print(df.head())

    print("\nShape:")
    print(df.shape)

    print("\nSaving to:", OUTPUT_FILE)

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nDone.")


# ---------------------------------------

if __name__ == "__main__":
    parse_raw_matches()