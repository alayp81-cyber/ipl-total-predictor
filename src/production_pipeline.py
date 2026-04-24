import subprocess
from pathlib import Path
import sys

# ---------------------------------------
# Paths
# ---------------------------------------

PROJECT_DIR = Path("/Users/alay/Desktop/ipl_total_predictior")
SRC_DIR = PROJECT_DIR / "src"

REFRESH_SCRIPT = SRC_DIR / "refresh_pipeline.py"
CHECK_SCRIPT = SRC_DIR / "check_retrain_needed.py"
TRAIN_SCRIPT = SRC_DIR / "train_catboost_time_split_weighted_versioned.py"

# ---------------------------------------
# Helper Function
# ---------------------------------------

def run_script(script_path):
    print("\nRunning:", script_path.name)

    result = subprocess.run(
        ["python3", str(script_path)],
        capture_output=True,
        text=True
    )

    print(result.stdout)

    if result.returncode not in [0, 1]:
        print("Error running:", script_path.name)
        print(result.stderr)
        sys.exit(1)

    return result.returncode

# ---------------------------------------
# Main Pipeline
# ---------------------------------------

def main():
    print("\n=====================================")
    print(" IPL Production Pipeline Starting ")
    print("=====================================")

    # Step 1 — Refresh Features
    run_script(REFRESH_SCRIPT)

    # Step 2 — Check Retrain Requirement
    retrain_signal = run_script(CHECK_SCRIPT)

    # Step 3 — Retrain If Needed
    if retrain_signal == 1:
        print("\nRetraining model with versioning...")
        run_script(TRAIN_SCRIPT)
        print("\nModel retraining completed.")
    else:
        print("\nSkipping retraining.")

    print("\n=====================================")
    print(" Pipeline Completed Successfully ")
    print("=====================================")

# ---------------------------------------

if __name__ == "__main__":
    main()