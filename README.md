# FinanSafe — Banking Intelligence System

A production-structured financial risk management web application built with Flask and scikit-learn. FinanSafe integrates three independently trained machine learning models to perform real-time credit risk assessment, customer behavioral profiling, customer segmentation, and transaction-level fraud detection — all served through a clean, multi-page web interface.

---

## Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Machine Learning Models](#machine-learning-models)
- [Behavioral Risk Engine](#behavioral-risk-engine)
- [Loan Recommendation Logic](#loan-recommendation-logic)
- [Application Pages and Routes](#application-pages-and-routes)
- [Data Storage](#data-storage)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [Running the Application](#running-the-application)
- [Dependencies](#dependencies)
- [Known Limitations](#known-limitations)
- [Future Improvements](#future-improvements)

---

## Project Overview

FinanSafe is designed to simulate the risk intelligence layer of a modern banking or fintech platform. It addresses two distinct problem domains:

**Customer-Level Risk Assessment** — Given a customer's financial profile (income, debt ratios, payment history, loan amount), the system produces a behavioral risk score, a machine learning-based default probability, a customer segment classification, and a final loan recommendation of APPROVED, REVIEW, or REJECTED.

**Transaction-Level Fraud Detection** — Given a single credit card transaction (represented by PCA-transformed features as used in real banking systems), the system predicts whether the transaction is fraudulent and returns a risk score and verdict in real time.

All three models are pre-trained offline using real public datasets and loaded at server startup via pickle. No model training occurs at runtime.

---

## System Architecture

```
User (Browser)
      |
      v
Flask Web Server (app.py)
      |
      |---> /               Customer Assessment Page
      |---> /analyze        Runs Default Model + Segmentation Model + Behavioral Engine
      |---> /fraud          Fraud Detection Page
      |---> /fraud_predict  Runs Fraud Detection Model on real transaction
      |---> /dashboard      Reads assessments.csv, displays all records + stats
      |
      |---> models/
      |         credit_default_model.pkl
      |         customer_segmentation_model.pkl
      |         fraud_detection_model.pkl
      |
      |---> assessments.csv     (runtime storage, appended on each assessment)
      |---> creditcard.csv      (fraud dataset, loaded once at startup)
```

The frontend is rendered entirely server-side as Python f-strings injected into a base HTML template. There is no JavaScript framework, no frontend build step, and no external CSS library.

---

## Machine Learning Models

### 1. Credit Default Prediction Model

**File:** `models/credit_default_model.pkl`

**Purpose:** Predicts the probability that a loan applicant will default on their debt obligations within two years.

**Training Dataset:** Give Me Some Credit — Kaggle (150,000 records)

**Performance:**
- ROC-AUC: 0.82
- Overall Accuracy: 82%

**Input Features (exact column names required):**

| Feature | Description |
|---|---|
| `RevolvingUtilizationOfUnsecuredLines` | Revolving credit utilization ratio (0 to 1) |
| `age` | Age of the borrower in years |
| `NumberOfTime30-59DaysPastDueNotWorse` | Count of 30-59 day late payments |
| `DebtRatio` | Monthly debt payments divided by monthly income |
| `MonthlyIncome` | Monthly income in dollars |
| `NumberOfOpenCreditLinesAndLoans` | Total open credit lines |
| `NumberOfTimes90DaysLate` | Count of 90+ day late payments |
| `NumberRealEstateLoansOrLines` | Number of real estate loans held |
| `NumberOfTime60-89DaysPastDueNotWorse` | Count of 60-89 day late payments |
| `NumberOfDependents` | Number of financial dependents |

**Output:** `predict_proba()[0][1]` — probability of default (0.0 to 1.0)

**Risk Thresholds:**

| Score | Label |
|---|---|
| Below 30% | LOW RISK |
| 30% to 59% | MEDIUM RISK |
| 60% and above | HIGH RISK |

---

### 2. Customer Segmentation Model

**File:** `models/customer_segmentation_model.pkl`

**Purpose:** Classifies a customer into one of five behavioral-financial segments using unsupervised clustering. The segment is used as an additional signal in the loan recommendation engine.

**Algorithm:** K-Means Clustering

**Silhouette Score:** 0.55

**Input Features:**

| Feature | Preprocessing |
|---|---|
| Annual Income | Divided by 1000 (passed in thousands) |
| Spending Score | Raw value (1 to 100) |

**Input shape:** `numpy.array([[annual_income_k, spending_score]])` — shape `(1, 2)`

**Output Segments:**

| Cluster Index | Segment Label |
|---|---|
| 0 | Budget Customer |
| 1 | Premium Customer |
| 2 | Impulsive Buyer |
| 3 | Careful Spender |
| 4 | Middle Customer |

Segments 1 (Premium) and 3 (Careful Spender) are treated as lower-risk profiles in the loan recommendation logic.

---

### 3. Fraud Detection Model

**File:** `models/fraud_detection_model.pkl`

**Purpose:** Classifies individual credit card transactions as fraudulent or legitimate.

**Training Dataset:** Credit Card Fraud Detection — ULB Machine Learning Group, Kaggle (284,807 transactions)

**Performance:**

| Metric | Value |
|---|---|
| ROC-AUC | 0.99 |
| Fraud Recall | 85% |
| Overall Accuracy | 99% |
| Fraud Rate in Dataset | 0.17% |

**Input Features:** V1 through V28 (PCA-transformed by the issuing bank to protect cardholder privacy), plus `Amount_Scaled` and `Time_Scaled` (StandardScaler applied at load time). The original `Amount` and `Time` columns are dropped after scaling.

**Preprocessing applied at startup:**

```python
scaler = StandardScaler()
df_fraud['Amount_Scaled'] = scaler.fit_transform(df_fraud[['Amount']])
df_fraud['Time_Scaled']   = scaler.fit_transform(df_fraud[['Time']])
df_fraud.drop(columns=['Amount', 'Time'], inplace=True)
```

**Output:**
- `predict_proba()[0][1]` — fraud probability (displayed as risk score percentage)
- `predict()[0]` — binary prediction (1 = Fraud, 0 = Legitimate)

**Risk Thresholds:**

| Score | Color |
|---|---|
| Below 30% | Green |
| 30% to 59% | Orange |
| 60% and above | Red |

---

## Behavioral Risk Engine

The behavioral risk score is a rule-based scoring system that runs independently of the ML models. It quantifies a customer's financial stress signals using weighted rules applied to their input data.

**Scoring Rules:**

| Condition | Points Added |
|---|---|
| Debt ratio above 0.8 | +30 |
| Debt ratio 0.5 to 0.8 | +15 |
| Debt ratio 0.3 to 0.5 | +5 |
| Utilization above 0.8 | +25 |
| Utilization 0.5 to 0.8 | +12 |
| Utilization 0.3 to 0.5 | +5 |
| Each 30-59 day late payment | +8 |
| Each 60-89 day late payment | +12 |
| Each 90+ day late payment | +20 |
| More than 4 dependents | +10 |
| 3 to 4 dependents | +5 |
| Loan-to-income ratio above 5x | +30 |
| Loan-to-income ratio 3x to 5x | +15 |
| Loan-to-income ratio 1x to 3x | +5 |

Maximum score is capped at 100.

**Output Thresholds:**

| Score | Label | Color |
|---|---|---|
| Below 30 | LOW RISK | Green (#28a745) |
| 30 to 59 | MEDIUM RISK | Orange (#fd7e14) |
| 60 and above | HIGH RISK | Red (#dc3545) |

---

## Loan Recommendation Logic

The final loan decision combines the ML default probability, the behavioral risk score, the customer segment, and the loan-to-income ratio. Rules are evaluated in priority order (first match wins):

| Condition | Decision |
|---|---|
| Loan-to-income ratio exceeds 5x | REJECTED |
| Default probability above 70% | REJECTED |
| Behavioral score >= 60 AND default probability above 40% | REJECTED |
| Default < 20% AND behavioral < 30 AND segment is Premium or Careful AND LTI <= 3x | APPROVED |
| Default < 30% AND behavioral < 30 AND LTI <= 3x | APPROVED |
| Default < 40% AND behavioral < 40 AND segment is Premium, Careful, or Middle AND LTI <= 2x | APPROVED |
| Default < 50% AND behavioral < 60 | REVIEW |
| All other combinations | REJECTED |

---

## Application Pages and Routes

| Route | Method | Description |
|---|---|---|
| `/` | GET | Customer Assessment form page |
| `/analyze` | POST | Processes form, runs all models, appends to CSV, returns results |
| `/fraud` | GET | Fraud Detection page with dataset availability check |
| `/fraud_predict` | POST | Selects real transaction from dataset, runs fraud model, returns verdict |
| `/dashboard` | GET | Reads assessments.csv, displays full assessment history with statistics |

---

## Data Storage

All completed assessments are appended to `assessments.csv` in the project root. The file is created automatically on first run if it does not exist.

**CSV Columns (21 total):**

```
timestamp, name, age, income, annual_income, spending_score,
debt_ratio, utilization, dependents, credit_lines, real_estate_loans,
late_30_59, late_60_89, late_90, loan_amount,
behavioral_score, behavioral_label, default_score, default_label,
segment, recommendation
```

This file is gitignored and exists only at runtime on the local machine.

---

## Project Structure

```
FinanSafe/
|
|-- app.py                              Main Flask application
|-- assessments.csv                     Runtime assessment log (auto-created, gitignored)
|-- creditcard.csv                      Fraud detection dataset (gitignored)
|-- .gitignore                          Excludes models, datasets, and CSV log
|
|-- models/
|       credit_default_model.pkl        Trained default prediction model (gitignored)
|       customer_segmentation_model.pkl Trained K-Means segmentation model (gitignored)
|       fraud_detection_model.pkl       Trained fraud detection model (gitignored)
```

Note: The models and datasets are excluded from version control due to file size. They must be trained or sourced separately and placed in the correct paths before running the application.

---

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- pip
- Virtual environment tool (recommended)

### Step 1 — Clone the repository

```bash
git clone https://github.com/nikh240103026-debug/finansafe-banking-intelligence-system.git
cd finansafe-banking-intelligence-system
```

### Step 2 — Create and activate a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install flask pandas numpy scikit-learn
```

### Step 4 — Add the trained models

Place the following pickle files in a `models/` directory at the project root:

```
models/credit_default_model.pkl
models/customer_segmentation_model.pkl
models/fraud_detection_model.pkl
```

These models must be trained using the datasets and feature schemas described in the Machine Learning Models section above. The exact input column names must match what is documented, as the models are loaded and called directly without any adapter layer.

### Step 5 — Add the fraud dataset

Download `creditcard.csv` from the ULB Credit Card Fraud Detection dataset on Kaggle and place it in the project root:

```
finansafe-banking-intelligence-system/
    creditcard.csv
```

Without this file, the Fraud Detection page will display a warning and the transaction simulation buttons will not appear. The Customer Assessment and Dashboard pages are not affected.

---

## Running the Application

Always run the application from inside the project directory to ensure all relative file paths resolve correctly:

```bash
cd finansafe-banking-intelligence-system
python app.py
```

Then open your browser and navigate to:

```
http://127.0.0.1:5000
```

If you encounter file-not-found errors for models or the dataset, add the following near the top of `app.py` to confirm the working directory at runtime:

```python
import os
print("Working directory:", os.getcwd())
print("creditcard.csv found:", os.path.exists('creditcard.csv'))
```

---

## Dependencies

| Package | Purpose |
|---|---|
| Flask | Web framework and routing |
| pandas | Data loading, CSV handling, DataFrame construction |
| numpy | Array construction for model input |
| scikit-learn | Model loading (pickle), StandardScaler for fraud preprocessing |
| pickle | Deserializing trained model files (standard library) |
| csv | Writing assessment records (standard library) |
| os | File path handling (standard library) |
| random | Transaction sampling for fraud simulation (standard library) |
| datetime | Timestamping assessment records (standard library) |

---

## Known Limitations

- No user authentication or session isolation. All assessments are written to a single shared CSV file.
- Storage is file-based (CSV). There is no database, so concurrent writes from multiple users may cause data corruption.
- The fraud model requires `creditcard.csv` to be present at startup. If it is missing, fraud simulation is entirely disabled.
- The frontend is rendered as inline Python strings. There are no separate HTML template files, making UI changes more difficult to maintain.
- Models are not versioned or retrained on new data. Any data drift over time will reduce prediction quality.
- There are no API endpoints. All interactions are form POST requests returning full HTML pages.
- The application has no mobile-responsive layout.
- Model files are gitignored, meaning the repository cannot be run standalone without separately sourcing or retraining the models.

---

## Future Improvements

- Migrate storage from CSV to a relational database (PostgreSQL or SQLite) to support concurrent access and proper querying
- Add user authentication with role-based access (underwriter, analyst, admin)
- Expose REST API endpoints so the risk engine can be consumed by external applications
- Separate HTML templates into Jinja2 `.html` files for maintainability
- Add model versioning and a retraining pipeline triggered by new assessment data
- Implement mobile-responsive design
- Add model explainability output (SHAP values) to show which features drove each decision
- Deploy to a cloud platform (Render, Railway, or AWS) with environment variable configuration for file paths
- Include the model training notebooks in the repository so the project can be fully reproduced from source data

---

## Datasets Used

**Give Me Some Credit** — Kaggle competition dataset for credit default prediction.
Source: https://www.kaggle.com/c/GiveMeSomeCredit

**Credit Card Fraud Detection** — Real anonymized European cardholder transactions provided by the ULB Machine Learning Group.
Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

**Mall Customer Segmentation Data** — Customer income and spending score dataset used for K-Means clustering.
Source: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

---

## License

This project is intended for educational and portfolio purposes. The datasets used are publicly available on Kaggle under their respective licenses. The trained model files are not distributed with this repository due to the large file size that is not accepted by github.

---

## Author

Nikhil Raj from IIIT MANIPUR - B.TECH, Computer Science & Engineering with specialization in AI and Data Science, currently working on FinTech Data Science projects to solve the real world problems exist in Financial world.
