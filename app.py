import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import datetime

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
# LOAD ARTIFACTS
# -------------------------------
model = joblib.load("artifacts/fraud_rf_model.pkl")
preprocessor = joblib.load("artifacts/preprocessor.pkl")
feature_columns = joblib.load("artifacts/feature_columns.pkl")
reference_scores = np.load("artifacts/reference_scores.npy")

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
    [
        "New York", "Los Angeles", "Houston", "Dallas",
        "Phoenix", "Philadelphia", "San Antonio",
        "San Diego", "San Jose"
    ]
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

# Base input row
input_dict = {
    "Amount": amount,
    "MerchantID": merchant_id,
    "hour": hour,
    "day": day,
    "weekday": weekday,
    "is_weekend": is_weekend
}

# Initialize ALL expected columns with 0
input_df = pd.DataFrame(
    np.zeros((1, len(feature_columns))),
    columns=feature_columns
)

# Fill numeric values
for col in input_dict:
    input_df[col] = input_dict[col]

# One-hot TransactionType
if transaction_type == "refund":
    input_df["TransactionType_refund"] = 1

# One-hot Location
loc_col = f"Location_{location}"
if loc_col in input_df.columns:
    input_df[loc_col] = 1

# -------------------------------
# TRANSFORM & SCORE
# -------------------------------
X_processed = preprocessor.transform(input_df)
risk_score = model.predict_proba(X_processed)[0][1]

# Percentile calculation
percentile = (reference_scores < risk_score).mean() * 100

# -------------------------------
# RISK LABEL
# -------------------------------
if percentile >= 98.5:
    risk_label = "🔴 HIGH RISK"
    color = "red"
elif percentile >= 95:
    risk_label = "🟠 MEDIUM RISK"
    color = "orange"
else:
    risk_label = "🟢 LOW RISK"
    color = "green"

# -------------------------------
# MAIN DASHBOARD
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    st.metric("Risk Percentile", f"{percentile:.2f}%")
    st.metric("Fraud Risk Score", f"{risk_score:.5f}")
    st.markdown(f"### Risk Level: :{color}[{risk_label}]")

with col2:
    st.markdown("### 📊 Risk Distribution (Training Data)")
    fig, ax = plt.subplots()
    ax.hist(reference_scores, bins=50)
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
