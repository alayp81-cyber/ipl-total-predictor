import streamlit as st
import pickle
import pandas as pd
import numpy as np

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

/* Background */
.stApp {
    background: linear-gradient(180deg, #0b1f4d 0%, #102b6a 100%);
}

/* Titles */
h1, h2, h3 {
    color: white !important;
}

/* Labels */
label {
    color: white !important;
    font-weight: 600;
}

/* Buttons */
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

/* Metric Box */
div[data-testid="stMetric"] {
    background-color: rgba(255,255,255,0.08);
    padding: 15px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------
# RELATIVE PATHS (Cloud Safe)
# ---------------------------------------

MODEL_PATH = "models/random_forest_model_with_xi.pkl"
FEATURE_DATA_PATH = "data/features/match_feature_view_with_xi.csv"

# ---------------------------------------
# Load Model
# ---------------------------------------

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

# ---------------------------------------
# Load Data
# ---------------------------------------

@st.cache_data
def load_feature_data():
    df = pd.read_csv(FEATURE_DATA_PATH)
    return df

model = load_model()
feature_df = load_feature_data()

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
# Range Calculation
# ---------------------------------------

def get_prediction_range(rf_model, X_input):

    x_array = X_input.to_numpy()

    tree_preds = np.array([
        tree.predict(x_array)[0]
        for tree in rf_model.estimators_
    ])

    mean_pred = float(tree_preds.mean())
    std_pred = float(tree_preds.std())

    low = mean_pred - std_pred
    high = mean_pred + std_pred

    return mean_pred, low, high

# ---------------------------------------
# Title
# ---------------------------------------

st.title("🏏 IPL Match Total Predictor")

st.success("✅ Model and data loaded")

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

    toss_winner = st.selectbox(
        "Toss Winner",
        TEAMS
    )

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

    template["toss_decision_bat_flag"] = int(toss_decision == "bat")
    template["toss_decision_field_flag"] = int(toss_decision == "field")

    if "match_total_runs" in template.index:
        template = template.drop("match_total_runs")

    X_input = pd.DataFrame([template])

    model_features = model.feature_names_in_
    X_input = X_input[model_features]

    prediction, low, high = get_prediction_range(
        model,
        X_input
    )

    prediction = round(prediction)
    low = round(low)
    high = round(high)

    # ---------------------------------------
    # Banner
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
        st.metric(
            "Expected Runs",
            f"{prediction}"
        )

    with c2:
        st.metric(
            "Predicted Range",
            f"{low}–{high}"
        )