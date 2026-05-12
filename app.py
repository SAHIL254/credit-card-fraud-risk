import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

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
st.sidebar.header("🧾 Transaction Details")

amount = st.sidebar.number_input("Transaction Amount", 1.0, 1000000.0, 2500.0)
merchant_id = st.sidebar.number_input("Merchant ID", 1, 100000, 500)

transaction_type = st.sidebar.selectbox(
    "Transaction Type",
    ["purchase", "refund"]
)

location = st.sidebar.selectbox(
    "Location",
    LOCATIONS
)

transaction_time = st.sidebar.time_input("Transaction Time")

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
now = datetime.datetime.now()

hour = transaction_time.hour
day = now.day
weekday = now.weekday()
is_weekend = int(weekday in [5, 6])

# Prepare transaction data for prediction
transaction_data = {
    "amount": amount,
    "merchant_id": merchant_id,
    "transaction_type": transaction_type,
    "location": location,
    "transaction_time": transaction_time
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
    ax.axvline(risk_score, linestyle="--")
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
