import streamlit as st
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------
# Project Root Detection
# ---------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "latest_model.pkl"
FEATURE_DATA_PATH = BASE_DIR / "data" / "features" / "match_feature_view_with_xi.csv"

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

model = load_model()
feature_df = load_feature_data()

# ---------------------------------------
# Dropdown Options
# ---------------------------------------

TEAMS = sorted(feature_df["team1"].unique())
VENUES = sorted(feature_df["venue"].unique())
TOSS_DECISIONS = ["bat", "field"]

# ---------------------------------------
# Prediction Range Helper
# ---------------------------------------

def get_prediction_range(model, X_input):

    prediction = float(model.predict(X_input)[0])

    # Conservative fallback range
    return prediction, prediction - 25, prediction + 25

# ---------------------------------------
# Title
# ---------------------------------------

st.title("🏏 IPL Match Total Predictor")

st.success("✅ Latest model and data loaded")

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

    if "match_total_runs" in template.index:
        template = template.drop("match_total_runs")

    if "recency_weight" in template.index:
        template = template.drop("recency_weight")

    X_input = pd.DataFrame([template])

    model_features = model.feature_names_

    X_input = X_input[model_features]

    prediction, low, high = get_prediction_range(model, X_input)

    prediction = round(prediction)
    low = round(low)
    high = round(high)

    st.markdown("## 🎯 Predicted Total")

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Expected Runs", f"{prediction}")

    with c2:
        st.metric("Predicted Range", f"{low}–{high}")