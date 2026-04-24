import streamlit as st
import pickle
import pandas as pd
import numpy as np
import json
from pathlib import Path

# ---------------------------------------
# Project Root Detection
# ---------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "latest_model.pkl"
FEATURE_DATA_PATH = BASE_DIR / "data" / "features" / "match_feature_view_with_xi.csv"
RESIDUAL_STATS_PATH = BASE_DIR / "models" / "residual_range_stats.json"

# ---------------------------------------
# Page Config
# ---------------------------------------

st.set_page_config(
    page_title="IPL Total Predictor",
    layout="centered"
)

# ---------------------------------------
# IPL Vibrant Theme Styling
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
    border: none;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #e57c00;
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
    if not MODEL_PATH.exists():
        st.error(f"Model file not found at: {MODEL_PATH}")
        st.stop()

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    return model

# ---------------------------------------
# Load Data
# ---------------------------------------

@st.cache_data
def load_feature_data():
    if not FEATURE_DATA_PATH.exists():
        st.error(f"Feature data not found at: {FEATURE_DATA_PATH}")
        st.stop()

    return pd.read_csv(FEATURE_DATA_PATH)

# ---------------------------------------
# Load Residual Stats
# ---------------------------------------

@st.cache_data
def load_residual_stats():
    if not RESIDUAL_STATS_PATH.exists():
        return {
            "range_half_width": 25,
            "mae": None,
            "p75_abs_error": None,
            "test_rows": None
        }

    with open(RESIDUAL_STATS_PATH, "r") as f:
        return json.load(f)

model = load_model()
feature_df = load_feature_data()
residual_stats = load_residual_stats()

# ---------------------------------------
# Dropdown Options
# ---------------------------------------

TEAMS = sorted(feature_df["team1"].unique())
VENUES = sorted(feature_df["venue"].unique())
TOSS_DECISIONS = ["bat", "field"]

# ---------------------------------------
# Team Colors
# ---------------------------------------

TEAM_COLORS = {
    "Chennai Super Kings": "#fdb913",
    "Delhi Capitals": "#17449b",
    "Gujarat Titans": "#1c1c1c",
    "Kolkata Knight Riders": "#3a225d",
    "Lucknow Super Giants": "#00aaff",
    "Mumbai Indians": "#004ba0",
    "Punjab Kings": "#d71920",
    "Rajasthan Royals": "#ea1a85",
    "Royal Challengers Bengaluru": "#ec1c24",
    "Sunrisers Hyderabad": "#ff822a",
}

# ---------------------------------------
# Prediction Range Helper
# ---------------------------------------

def get_prediction_range(model, X_input, residual_stats):
    prediction = float(model.predict(X_input)[0])

    half_width = residual_stats.get("range_half_width", 25)

    low = prediction - half_width
    high = prediction + half_width

    return prediction, low, high

# ---------------------------------------
# Title
# ---------------------------------------

st.title("🏏 IPL Match Total Predictor")

st.success("✅ Latest model, data and residual range stats loaded")

st.markdown("---")

# ---------------------------------------
# Inputs
# ---------------------------------------

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

    team2 = st.selectbox(
        "Team 2",
        TEAMS,
        index=1
    )

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

    if team1 == team2:
        st.error("Teams must be different.")
        st.stop()

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

    template["toss_decision_bat_flag"] = int(toss_decision == "bat")

    if "toss_decision_field_flag" in template.index:
        template["toss_decision_field_flag"] = int(toss_decision == "field")

    if "match_total_runs" in template.index:
        template = template.drop("match_total_runs")

    if "recency_weight" in template.index:
        template = template.drop("recency_weight")

    X_input = pd.DataFrame([template])

    model_features = model.feature_names_
    X_input = X_input[model_features]

    prediction, low, high = get_prediction_range(
        model,
        X_input,
        residual_stats
    )

    prediction = round(prediction)
    low = round(low)
    high = round(high)

    # ---------------------------------------
    # Match Banner
    # ---------------------------------------

    team1_color = TEAM_COLORS.get(team1, "#ffffff")
    team2_color = TEAM_COLORS.get(team2, "#ffffff")

    st.markdown("## 📊 Match Summary")

    st.markdown(
        f"""
        <div style="
            padding:15px;
            border-radius:12px;
            background: linear-gradient(
                90deg,
                {team1_color} 0%,
                {team2_color} 100%
            );
            text-align:center;
            font-weight:700;
            font-size:20px;
            color:white;
        ">
            {team1} vs {team2}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write(f"Venue: **{venue}**")
    st.write(f"Toss Winner: **{toss_winner}**")
    st.write(f"Toss Decision: **{toss_decision.upper()}**")

    st.markdown("---")

    st.markdown("## 🎯 Predicted Total")

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Expected Runs", f"{prediction}")

    with c2:
        st.metric("Predicted Range", f"{low}–{high}")

    st.caption(
        f"Range uses historical model error band. "
        f"MAE: {residual_stats.get('mae', 'N/A')}, "
        f"75th percentile error: {residual_stats.get('p75_abs_error', 'N/A')}."
    )