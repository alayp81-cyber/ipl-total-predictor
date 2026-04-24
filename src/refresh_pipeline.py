from pathlib import Path
import subprocess
import sys

# ---------------------------------------
# Project Paths
# ---------------------------------------

PROJECT_DIR = Path("/Users/alay/Desktop/ipl_total_predictior")
SRC_DIR = PROJECT_DIR / "src"

# ---------------------------------------
# YOUR ACTUAL PIPELINE ORDER
# ---------------------------------------

PIPELINE_STEPS = [

    # Raw Parsing
    "parse_raw_matches.py",

    # Clean Tables
    "build_clean_matches.py",
    "build_clean_team_innings.py",
    "build_clean_player_match_stats.py",
    "build_clean_playing_xi.py",

    # Validation
    "validate_raw_tables.py",

    # Feature Engineering
    "build_team_form_features.py",
    "build_venue_features.py",
    "build_xi_strength_features.py",

    # Final Feature Views
    "build_match_feature_view.py",
    "build_match_feature_view_with_xi.py",

]

# ---------------------------------------
# Runner
# ---------------------------------------

def run_step(script_name):

    script_path = SRC_DIR / script_name

    if not script_path.exists():
        raise FileNotFoundError(
            f"Missing script: {script_path}"
        )

    print("\n" + "=" * 60)
    print(f"Running: {script_name}")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(SRC_DIR)
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Pipeline failed at: {script_name}"
        )

    print(f"Completed: {script_name}")


def main():

    print("\nStarting IPL Rolling Refresh Pipeline...")

    for script_name in PIPELINE_STEPS:

        run_step(script_name)

    print("\n" + "=" * 60)
    print("Pipeline refresh completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()