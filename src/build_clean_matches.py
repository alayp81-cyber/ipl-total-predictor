from pathlib import Path
import pandas as pd

# ---------------------------------------
# Paths
# ---------------------------------------

PROJECT_DIR = Path("/Users/alay/Desktop/ipl_total_predictior")

INTERIM_DIR = PROJECT_DIR / "data" / "interim"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = INTERIM_DIR / "raw_matches.csv"
OUTPUT_FILE = PROCESSED_DIR / "clean_matches.csv"

# ---------------------------------------
# Load
# ---------------------------------------

print("Loading raw matches...")

df = pd.read_csv(INPUT_FILE)

print("Original shape:", df.shape)

# ---------------------------------------
# Filtering Logic
# ---------------------------------------

print("\nFiltering completed matches...")

df = df[
    df["completed_flag"] == True
]

print("After completion filter:", df.shape)

print("\nFiltering seasons >= 2018...")

df = df[
    df["season"] >= 2018
]

print("After season filter:", df.shape)

print("\nRemoving super-over-only records...")

df = df[
    df["super_over_flag"] == False
]

print("After super-over filter:", df.shape)

# ---------------------------------------
# Final Checks
# ---------------------------------------

print("\nFinal season range:",
      df["season"].min(),
      "to",
      df["season"].max())

print("Total matches retained:", len(df))

# ---------------------------------------
# Save
# ---------------------------------------

print("\nSaving to:", OUTPUT_FILE)

df.to_csv(OUTPUT_FILE, index=False)

print("\nDone.")