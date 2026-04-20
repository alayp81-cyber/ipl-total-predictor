from pathlib import Path
import pandas as pd

# ---------------------------------------
# Paths
# ---------------------------------------

PROJECT_DIR = Path("/Users/alay/Desktop/ipl_total_predictior")

INTERIM_DIR = PROJECT_DIR / "data" / "interim"

MATCHES_FILE = INTERIM_DIR / "raw_matches.csv"
PLAYING_XI_FILE = INTERIM_DIR / "raw_playing_xi.csv"
TEAM_INNINGS_FILE = INTERIM_DIR / "raw_team_innings.csv"
PLAYER_STATS_FILE = INTERIM_DIR / "raw_player_match_stats.csv"

# ---------------------------------------
# Load Data
# ---------------------------------------

print("Loading raw tables...\n")

matches = pd.read_csv(MATCHES_FILE)
playing_xi = pd.read_csv(PLAYING_XI_FILE)
team_innings = pd.read_csv(TEAM_INNINGS_FILE)
player_stats = pd.read_csv(PLAYER_STATS_FILE)

# ---------------------------------------
# Basic Shapes
# ---------------------------------------

print("=== Table Shapes ===")

print("raw_matches:", matches.shape)
print("raw_playing_xi:", playing_xi.shape)
print("raw_team_innings:", team_innings.shape)
print("raw_player_match_stats:", player_stats.shape)

# Expected rough sizes
print("\n=== Expected Size Checks ===")

print("Matches:", len(matches))
print("Playing XI rows per match:",
      round(len(playing_xi) / len(matches), 2))

print("Team innings rows per match:",
      round(len(team_innings) / len(matches), 2))

print("Player stats rows per match:",
      round(len(player_stats) / len(matches), 2))

# ---------------------------------------
# Missing Value Checks
# ---------------------------------------

print("\n=== Missing Value Checks ===")

print("\nMatches missing totals:",
      matches["match_total_runs"].isna().sum())

print("Team innings missing runs:",
      team_innings["runs_scored"].isna().sum())

print("Player stats missing teams:",
      player_stats["team"].isna().sum())

# ---------------------------------------
# Logical Integrity Checks
# ---------------------------------------

print("\n=== Logical Integrity Checks ===")

# Each match should have exactly 2 innings
innings_per_match = (
    team_innings.groupby("match_id")
    .size()
)

bad_matches = innings_per_match[
    innings_per_match != 2
]

print("Matches without exactly 2 innings:",
      len(bad_matches))

# Each match should have ~22 players
players_per_match = (
    playing_xi.groupby("match_id")
    .size()
)

weird_player_counts = players_per_match[
    (players_per_match < 20) |
    (players_per_match > 26)
]

print("Matches with abnormal XI counts:",
      len(weird_player_counts))

# ---------------------------------------
# Range Checks
# ---------------------------------------

print("\n=== Range Checks ===")

print("Season range:",
      matches["season"].min(),
      "to",
      matches["season"].max())

print("Runs scored max:",
      team_innings["runs_scored"].max())

print("Runs scored min:",
      team_innings["runs_scored"].min())

print("Wickets lost max:",
      team_innings["wickets_lost"].max())

print("Wickets lost min:",
      team_innings["wickets_lost"].min())

print("\nValidation Complete.")