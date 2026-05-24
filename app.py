import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import sys
from pathlib import Path

# Add project root to path for imports (allow importing as package)
sys.path.insert(0, str(Path(__file__).parent))

from src.logger import setup_logging
# initialize logging as early as possible
setup_logging()

from src.pipeline.prediction_pipeline import PredictionPipeline
from src.utils import LOCATIONS

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(
    page_title="Fraud Risk Dashboard",
    layout="wide"
)

st.title("💳 Credit Card Fraud Risk Dashboard")
st.caption("Risk prioritization system for fraud analysts")

# -------------------------------
# LOAD PREDICTION PIPELINE
# -------------------------------
@st.cache_resource
def load_prediction_pipeline():
    """Load and cache the prediction pipeline"""
    return PredictionPipeline()

prediction_pipeline = load_prediction_pipeline()

# -------------------------------
# SIDEBAR INPUTS
# -------------------------------
if "transaction_time" not in st.session_state:
    st.session_state.transaction_time = datetime.datetime.now().time()

st.sidebar.header("🧾 Transaction Details")

amount = st.sidebar.number_input(
    "Transaction Amount",
    1.0,
    1000000.0,
    2500.0
)

transaction_type = st.sidebar.selectbox(
    "Transaction Type",
    ["purchase", "refund"]
)

location = st.sidebar.selectbox(
    "Location",
    LOCATIONS
)


transaction_time = st.sidebar.time_input(
    "Transaction Time",
    value=st.session_state.transaction_time,
    step=60
)

st.session_state.transaction_time = transaction_time

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
now = datetime.datetime.now()

hour = transaction_time.hour
day = now.day
weekday = now.weekday()
is_weekend = int(weekday in [5, 6])
is_night = int(hour >= 0 and hour <= 5)
high_amount = int(amount > 1000)
# Prepare transaction data for prediction
transaction_data = {
    "Amount": amount,
    "TransactionType": transaction_type,
    "Location": location,
    "hour": hour,
    "day": day,
    "weekday": weekday,
    "is_weekend": is_weekend,
    "is_night": is_night,
    "high_amount": high_amount
}

# -------------------------------
# PREDICT RISK
# -------------------------------
prediction = prediction_pipeline.predict_risk(transaction_data)

risk_score = prediction["risk_score"]
percentile = prediction["percentile"]
risk_level = prediction["risk_level"]

# -------------------------------
# MAIN DASHBOARD
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    st.metric("Risk Percentile", f"{percentile:.2f}%")
    st.metric("Fraud Risk Score", f"{risk_score:.5f}")
    st.markdown(f"### Risk Level: :{risk_level['color']}[{risk_level['label']}]")

with col2:
    st.markdown("### 📊 Risk Distribution (Training Data)")
    fig, ax = plt.subplots()
    ax.hist(prediction_pipeline.reference_scores, bins=50)
    ax.axvline(risk_score, linestyle="--", linewidth=2)
    ax.set_xlabel("Risk Score")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

# -------------------------------
# EXPLANATION
# -------------------------------
with st.expander("ℹ️ How to interpret this"):
    st.write("""
    - This system **does NOT auto-block transactions**
    - It **prioritizes transactions** for fraud review
    - Analysts investigate **top-risk percentiles**
    - Low probability does NOT mean safe — it means *lower relative risk*
    """)

# -------------------------------
# FOOTER
# -------------------------------
st.caption("⚠️ This is a decision-support system, not an automated fraud blocker.")
