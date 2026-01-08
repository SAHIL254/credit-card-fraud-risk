# 🟥 Credit Card Fraud Risk Prioritization System

A Machine Learning–based **fraud risk scoring and prioritization system** designed for financial institutions to identify and rank suspicious credit card transactions for manual review.
Instead of binary fraud prediction, the system focuses on **risk-based decision making**, which better reflects real-world banking operations.

---

## 🔥 Problem Statement

Credit card transaction data is **highly imbalanced**, with fraud cases typically below 1%.
In such scenarios:

* Accuracy becomes misleading
* Binary predictions fail to capture business risk
* Fraud teams need **prioritized alerts**, not yes/no decisions

This project addresses the problem by **ranking transactions based on fraud risk** so analysts can focus on the most suspicious cases first.

---

## 🎯 Project Objectives

* Build a fraud **risk scoring model** using Machine Learning
* Handle extreme class imbalance appropriately
* Avoid misleading accuracy-based evaluation
* Introduce **percentile-based risk prioritization**
* Provide an interactive **Streamlit dashboard** for fraud analysts
* Align model outputs with **real-world banking workflows**

---

## 🧠 Solution Approach

Instead of predicting *Fraud / Not Fraud* directly:

* The model outputs a **fraud probability score**
* Transactions are ranked by risk
* Risk is expressed as a **percentile**
* Analysts review the **top X% highest-risk transactions**

This mirrors **production-level fraud detection systems**.

---

## 🧰 Technologies Used

**Data & Analysis**: Pandas, NumPy, Matplotlib, Seaborn
**Machine Learning**: Random Forest (Primary), Logistic Regression (Baseline), Isolation Forest (Anomaly Detection)
**Preprocessing**: Feature engineering, One-Hot Encoding, StandardScaler, ColumnTransformer
**Evaluation Metrics**: ROC-AUC, PR-AUC, Fraud Capture @ K%, Percentile Thresholds
**Deployment**: Streamlit, Joblib (model persistence)

---

## 📊 Model Evaluation Strategy

Traditional metrics like accuracy were **intentionally avoided**.

| Metric                | Reason                            |
| --------------------- | --------------------------------- |
| ROC-AUC               | Measures ranking ability          |
| PR-AUC                | Focuses on minority (fraud) class |
| Fraud Capture @ K%    | Business-aligned metric           |
| Percentile Thresholds | Operational decision-making       |

---

## 🧮 Risk Prioritization Logic

* Model predicts **fraud probability**

* Probability is converted into a **risk percentile**

* Risk levels:

  * 🟢 Low Risk
  * 🟠 Medium Risk
  * 🔴 High Risk

* Analysts investigate **top-risk percentiles only**

---

## 🖥 Streamlit Dashboard Features

* Transaction input form
* Real-time fraud risk score
* Risk percentile calculation
* Visual risk indicator (Low / Medium / High)
* Business-friendly UI for fraud analysts

---

## 🖥 Streamlit Dashboard Preview

### Transaction Risk Evaluation Example

* **Input**: Transaction amount, merchant, location, time
* **Output**:

  * Fraud Risk Score
  * Risk Percentile
  * Risk Level (Low / Medium / High)

📸 **Screenshots**:

### 🔴 High Risk Example

![High Risk Example](screenshots/high_risk.png)

### 🟠 Medium Risk Example

![Medium Risk Example](screenshots/medium_risk.png)

### 🟢 Low Risk Example

![Low Risk Example](screenshots/low_risk.png)


> **Note:** Model artifact files (`.pkl`, `.npy`) are **not included in this repo** due to GitHub file size limits. You can generate them by running the notebook locally (`Credit_Card.ipynb`).

---

## 📁 Project Structure

```
CREDIT CARD FRAUD DETECTION/
├── app.py                  # Streamlit application
├── Credit_Card.ipynb       # Full ML notebook
├── credit_card_fraud_dataset.csv
├── artifacts/              # Generated locally, not tracked in Git
│   ├── fraud_rf_model.pkl
│   ├── preprocessor.pkl
│   ├── reference_scores.npy
│   └── feature_columns.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ▶ How to Run the Project

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Generate model artifacts locally

Run the notebook:

```bash
Credit_Card.ipynb
```

### 3️⃣ Run Streamlit app

```bash
streamlit run app.py
```

---

## 🧠 Business Interpretation

* **High Risk** → Immediate investigation
* **Medium Risk** → Monitor or review
* **Low Risk** → Auto-approve

> System **reduces false alarms** and improves fraud team efficiency.

---

## ⚠️ Key Learnings

* High accuracy ≠ good fraud model
* Imbalanced data requires ranking, not classification
* Percentile-based decisions reflect real banking systems
* ML success depends on **business alignment**

---

## 🚀 Future Improvements

* SHAP explainability dashboard
* Real-time transaction streaming
* Model monitoring & drift detection
* REST API deployment
* Cloud deployment (AWS/GCP)

---

## 👨‍💻 Author

**Sahil Dervankar**
B.Tech CSE (AI/ML)

Aspiring Machine Learning Engineer

---

