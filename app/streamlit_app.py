import streamlit as st
import pickle
import pandas as pd
import numpy as np
import json
from pathlib import Path

# ---------------------------------------
# Project Paths
# ---------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "latest_model.pkl"
FEATURE_DATA_PATH = BASE_DIR / "data" / "features" / "match_feature_view_with_xi.csv"

GLOBAL_STATS_PATH = BASE_DIR / "models" / "residual_range_stats.json"
VENUE_STATS_PATH = BASE_DIR / "models" / "venue_residual_range_stats.json"

# ---------------------------------------
# Page Config
# ---------------------------------------

st.set_page_config(
    page_title="IPL Total Predictor",
    layout="centered"
)

# ---------------------------------------
# Theme Styling
# ---------------------------------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #0b1f4d 0%, #102b6a 100%);
}

h1, h2, h3 {
    color: white !important;
}

label {
    color: white !important;
    font-weight: 600;
}

.stButton > button {
    background-color: #ff8c00;
    color: white;
    border-radius: 10px;
}

div[data-testid="stMetric"] {
    background-color: rgba(255,255,255,0.08);
    padding: 15px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------
# Load Model
# ---------------------------------------

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

# ---------------------------------------
# Load Data
# ---------------------------------------

@st.cache_data
def load_feature_data():
    return pd.read_csv(FEATURE_DATA_PATH)

# ---------------------------------------
# Load Global Stats
# ---------------------------------------

@st.cache_data
def load_global_stats():
    if GLOBAL_STATS_PATH.exists():
        with open(GLOBAL_STATS_PATH, "r") as f:
            return json.load(f)
    return {"range_half_width": 25}

# ---------------------------------------
# Load Venue Stats
# ---------------------------------------

@st.cache_data
def load_venue_stats():
    if VENUE_STATS_PATH.exists():
        with open(VENUE_STATS_PATH, "r") as f:
            return json.load(f)
    return {}

model = load_model()
feature_df = load_feature_data()
global_stats = load_global_stats()
venue_stats = load_venue_stats()

# ---------------------------------------
# Dropdown Options
# ---------------------------------------

TEAMS = sorted(feature_df["team1"].unique())
VENUES = sorted(feature_df["venue"].unique())
TOSS_DECISIONS = ["bat", "field"]

# ---------------------------------------
# Range Logic (Venue-aware)
# ---------------------------------------

def get_prediction_range(model, X_input, venue):

    prediction = float(model.predict(X_input)[0])

    # Venue-specific width
    if venue in venue_stats:

        half_width = venue_stats[venue]["range_half_width"]

    else:

        half_width = global_stats.get(
            "range_half_width",
            25
        )

    low = prediction - half_width
    high = prediction + half_width

    return prediction, low, high, half_width

# ---------------------------------------
# UI
# ---------------------------------------

st.title("🏏 IPL Match Total Predictor")

st.success("✅ Venue-aware prediction ranges active")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:

    season = st.number_input(
        "Season",
        min_value=2018,
        max_value=2035,
        value=2026
    )

    team1 = st.selectbox("Team 1", TEAMS)

    venue = st.selectbox("Venue", VENUES)

with col2:

    team2 = st.selectbox("Team 2", TEAMS)

    toss_winner = st.selectbox("Toss Winner", TEAMS)

    toss_decision = st.selectbox(
        "Toss Decision",
        TOSS_DECISIONS
    )

st.markdown("---")

# ---------------------------------------
# Prediction
# ---------------------------------------

if st.button("Predict Match Total"):

    template = feature_df[
        (feature_df["team1"] == team1) &
        (feature_df["team2"] == team2) &
        (feature_df["venue"] == venue)
    ]

    if template.empty:
        template = feature_df.sample(1)

    template = template.iloc[0].copy()

    template["season"] = season
    template["team1"] = team1
    template["team2"] = team2
    template["venue"] = venue
    template["toss_winner"] = toss_winner
    template["toss_decision"] = toss_decision

    template["toss_decision_bat_flag"] = int(
        toss_decision == "bat"
    )

    if "match_total_runs" in template.index:
        template = template.drop("match_total_runs")

    if "recency_weight" in template.index:
        template = template.drop("recency_weight")

    X_input = pd.DataFrame([template])

    model_features = model.feature_names_

    X_input = X_input[model_features]

    prediction, low, high, width = get_prediction_range(
        model,
        X_input,
        venue
    )

    prediction = round(prediction)
    low = round(low)
    high = round(high)

    st.markdown("## 🎯 Predicted Total")

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Expected Runs", prediction)

    with c2:
        st.metric(
            "Predicted Range",
            f"{low}–{high}"
        )

    st.caption(
        f"Range width based on venue volatility: ±{width}"
    )