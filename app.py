from flask import Flask, request, redirect, url_for
import pickle
import pandas as pd
import numpy as np
import csv
import os
import random
import html
import urllib.request
from datetime import datetime
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'finansafe2026')

# -----------------------------------------------------------------------------
# MODEL LOADING
# -----------------------------------------------------------------------------
# The credit-default model is too large for a normal GitHub repository. When
# running on Render, it can therefore be downloaded from the GitHub Release
# asset through CREDIT_MODEL_URL.

fraud_model = None
default_model = None
segment_model = None
MODEL_ERROR = None

try:
    with open('models/fraud_detection_model.pkl', 'rb') as f:
        fraud_model = pickle.load(f)

    credit_model_path = 'models/credit_default_model.pkl'

    if not os.path.exists(credit_model_path):
        credit_model_url = os.environ.get('CREDIT_MODEL_URL')

        if credit_model_url:
            print('Downloading credit default model...')
            os.makedirs('models', exist_ok=True)
            urllib.request.urlretrieve(credit_model_url, credit_model_path)
            print('Credit default model downloaded successfully.')
        else:
            print('CREDIT_MODEL_URL environment variable not found.')

    with open(credit_model_path, 'rb') as f:
        default_model = pickle.load(f)

    with open('models/customer_segmentation_model.pkl', 'rb') as f:
        segment_model = pickle.load(f)

except Exception as e:
    MODEL_ERROR = str(e)
    print(f'Warning: Model loading error: {e}')


segment_names = {
    0: 'Budget Customer',
    1: 'Premium Customer',
    2: 'Impulsive Buyer',
    3: 'Careful Spender',
    4: 'Middle Customer'
}


# -----------------------------------------------------------------------------
# ASSESSMENT STORAGE
# -----------------------------------------------------------------------------
DATA_FILE = 'assessments.csv'

ASSESSMENT_HEADERS = [
    'timestamp', 'name', 'age', 'income', 'annual_income',
    'spending_score', 'debt_ratio', 'utilization', 'dependents',
    'credit_lines', 'real_estate_loans', 'late_30_59',
    'late_60_89', 'late_90', 'loan_amount',
    'behavioral_score', 'behavioral_label',
    'default_score', 'default_label', 'segment', 'recommendation'
]


if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(ASSESSMENT_HEADERS)


# -----------------------------------------------------------------------------
# FRAUD DATASET
# -----------------------------------------------------------------------------
df_fraud = None
fraud_samples = None
legit_samples = None
fraud_amounts = None
legit_amounts = None
fraud_times = None
legit_times = None

if os.path.exists('creditcard.csv'):
    try:
        df_fraud = pd.read_csv('creditcard.csv')
        scaler = StandardScaler()

        original_amounts = df_fraud['Amount'].copy()
        original_times = df_fraud['Time'].copy()

        df_fraud['Amount_Scaled'] = scaler.fit_transform(df_fraud[['Amount']])
        df_fraud['Time_Scaled'] = scaler.fit_transform(df_fraud[['Time']])
        df_fraud.drop(columns=['Amount', 'Time'], inplace=True)

        fraud_samples = df_fraud[df_fraud['Class'] == 1].copy()
        legit_samples = df_fraud[df_fraud['Class'] == 0].copy()

        fraud_amounts = original_amounts[fraud_samples.index]
        legit_amounts = original_amounts[legit_samples.index]
        fraud_times = original_times[fraud_samples.index]
        legit_times = original_times[legit_samples.index]

    except Exception as e:
        print(f"MODEL LOADING ERROR: {type(e).__name__}: {e}")
        fraud_model = None
        default_model = None
        segment_model = None
        df_fraud = None


LOCATIONS = [
    'Online Purchase',
    'ATM Withdrawal',
    'POS Terminal',
    'International Transfer',
    'Mobile Payment',
    'E-commerce'
]


# -----------------------------------------------------------------------------
# BUSINESS LOGIC
# -----------------------------------------------------------------------------
def seconds_to_time(seconds):
    hours = int((seconds % 86400) / 3600)
    minutes = int((seconds % 3600) / 60)
    period = 'AM' if hours < 12 else 'PM'
    hours = hours % 12 or 12
    return f'{hours:02d}:{minutes:02d} {period}'


def calculate_behavioral_risk(
    debt_ratio,
    utilization,
    late_30_59,
    late_60_89,
    late_90,
    dependents,
    income,
    loan_amount
):
    score = 0

    if debt_ratio > 0.8:
        score += 30
    elif debt_ratio > 0.5:
        score += 15
    elif debt_ratio > 0.3:
        score += 5

    if utilization > 0.8:
        score += 25
    elif utilization > 0.5:
        score += 12
    elif utilization > 0.3:
        score += 5

    score += late_30_59 * 8
    score += late_60_89 * 12
    score += late_90 * 20

    if dependents > 4:
        score += 10
    elif dependents > 2:
        score += 5

    annual_income = income * 12

    if annual_income > 0:
        lti = loan_amount / annual_income

        if lti > 5:
            score += 30
        elif lti > 3:
            score += 15
        elif lti > 1:
            score += 5

    score = min(score, 100)

    if score >= 60:
        return score, 'HIGH RISK', '#b42318'
    elif score >= 30:
        return score, 'MEDIUM RISK', '#946200'
    else:
        return score, 'LOW RISK', '#167c5a'


def get_loan_recommendation(
    default_prob,
    behavioral_score,
    segment,
    monthly_income,
    loan_amount
):
    annual_income = monthly_income * 12
    loan_to_income = (
        loan_amount / annual_income
        if annual_income > 0
        else 999
    )

    if loan_to_income > 5:
        return (
            'REJECTED',
            f'Loan amount is {loan_to_income:.1f}x annual income. '
            'Exceeds maximum allowed ratio of 5x.',
            '#b42318'
        )

    if default_prob > 0.70:
        return (
            'REJECTED',
            'High credit default risk. Customer unlikely to repay loan.',
            '#b42318'
        )

    if behavioral_score >= 60 and default_prob > 0.40:
        return (
            'REJECTED',
            'High behavioral risk combined with elevated default probability.',
            '#b42318'
        )

    if (
        default_prob < 0.20
        and behavioral_score < 30
        and segment in [1, 3]
        and loan_to_income <= 3
    ):
        return (
            'APPROVED',
            f'Excellent profile. Low default risk with premium or careful '
            f'spender segment. Loan is {loan_to_income:.1f}x annual income.',
            '#167c5a'
        )

    if default_prob < 0.30 and behavioral_score < 30 and loan_to_income <= 3:
        return (
            'APPROVED',
            f'Good profile. Low default and behavioral risk. Loan is '
            f'{loan_to_income:.1f}x annual income.',
            '#167c5a'
        )

    if (
        default_prob < 0.40
        and behavioral_score < 40
        and segment in [1, 3, 4]
        and loan_to_income <= 2
    ):
        return (
            'APPROVED',
            'Acceptable risk profile. Loan approved with standard terms.',
            '#167c5a'
        )

    if default_prob < 0.50 and behavioral_score < 60:
        return (
            'REVIEW',
            f'Moderate risk profile. Loan is {loan_to_income:.1f}x annual '
            'income. Manual review recommended.',
            '#946200'
        )

    return (
        'REJECTED',
        'Combined risk indicators too high. Loan application rejected.',
        '#b42318'
    )


# -----------------------------------------------------------------------------
# UI BASE TEMPLATE
# -----------------------------------------------------------------------------
# The backend above intentionally follows the original application logic.
# The sections below are the redesigned frontend only.
# -----------------------------------------------------------------------------
BASE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="theme-color" content="#0b1f33">
    <title>FinanSafe — {title}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --navy: #0b1f33;
            --navy-2: #16324a;
            --green: #167c5a;
            --green-dark: #0f6247;
            --green-soft: #edf7f3;
            --red: #b42318;
            --red-soft: #fff1f0;
            --amber: #946200;
            --amber-soft: #fff8e6;
            --blue-soft: #eef4f8;
            --text: #17212b;
            --muted: #687582;
            --page: #f5f7f8;
            --surface: #ffffff;
            --soft: #f8fafb;
        }}

        * {{ box-sizing: border-box; }}

        html {{ scroll-behavior: smooth; }}

        body {{
            margin: 0;
            background: var(--page);
            color: var(--text);
            font-family: Inter, Arial, sans-serif;
            -webkit-font-smoothing: antialiased;
        }}

        a {{ text-decoration: none !important; }}

        button, input, select {{ font: inherit; }}

        .topbar {{
            background: #ffffff;
            position: sticky;
            top: 0;
            z-index: 50;
            box-shadow: 0 1px 0 rgba(11, 31, 51, 0.07);
        }}

        .topbar-inner {{
            max-width: 1180px;
            min-height: 72px;
            margin: 0 auto;
            padding: 0 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}

        .brand {{
            display: flex;
            align-items: center;
            gap: 11px;
            color: var(--navy);
            font-size: 18px;
            font-weight: 700;
            letter-spacing: -0.025em;
        }}

        .brand-mark {{
            width: 34px;
            height: 34px;
            display: grid;
            place-items: center;
            background: var(--navy);
            color: #ffffff;
        }}

        .brand-mark svg {{ width: 18px; height: 18px; }}

        .nav-links {{
            display: flex;
            align-items: center;
            gap: 28px;
            height: 72px;
        }}

        .nav-links a {{
            position: relative;
            height: 72px;
            display: flex;
            align-items: center;
            color: #71808d;
            font-size: 13px;
            font-weight: 600;
        }}

        .nav-links a:hover {{ color: var(--navy); }}

        .nav-links a.active {{ color: var(--green-dark); }}

        .nav-links a.active::after {{
            content: "";
            position: absolute;
            left: 0;
            right: 0;
            bottom: 0;
            height: 3px;
            background: var(--green);
        }}

        .page {{
            max-width: 1180px;
            margin: 0 auto;
            padding: 38px 24px 64px;
        }}

        .eyebrow {{
            margin-bottom: 9px;
            color: var(--green-dark);
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }}

        .page-title {{
            margin: 0 0 8px;
            color: var(--navy);
            font-size: 30px;
            line-height: 1.15;
            font-weight: 700;
            letter-spacing: -0.04em;
        }}

        .page-subtitle {{
            max-width: 720px;
            margin: 0;
            color: var(--muted);
            font-size: 14px;
            line-height: 1.7;
        }}

        .section-head {{ margin-bottom: 28px; }}

        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 1px;
            background: #e9eef1;
            margin-bottom: 30px;
        }}

        .metric {{
            min-height: 112px;
            padding: 22px;
            background: #ffffff;
        }}

        .metric-label {{
            color: #788692;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.11em;
            text-transform: uppercase;
            margin-bottom: 12px;
        }}

        .metric-value {{
            color: var(--navy);
            font-size: 25px;
            line-height: 1;
            font-weight: 700;
            letter-spacing: -0.03em;
        }}

        .layout {{
            display: grid;
            grid-template-columns: 360px minmax(0, 1fr);
            gap: 28px;
            align-items: start;
        }}

        .panel {{
            background: #ffffff;
            box-shadow: 0 7px 22px rgba(11, 31, 51, 0.045);
        }}

        .panel-head {{
            padding: 19px 22px;
            background: #fbfcfc;
            box-shadow: inset 0 -1px 0 #edf0f2;
        }}

        .panel-title {{
            margin: 0;
            color: var(--navy);
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }}

        .form {{ padding: 22px; }}

        .field {{ margin-bottom: 17px; }}

        .field:last-child {{ margin-bottom: 0; }}

        .field-label {{
            display: block;
            margin-bottom: 7px;
            color: #52616d;
            font-size: 11px;
            font-weight: 700;
        }}

        .input {{
            width: 100%;
            height: 43px;
            padding: 0 12px;
            border: 0;
            border-radius: 0;
            outline: 0;
            background: #f7f9fa;
            color: var(--text);
            box-shadow: inset 0 -2px 0 #dce3e7;
            transition: background .18s ease, box-shadow .18s ease;
        }}

        .input:focus {{
            background: #ffffff;
            box-shadow: inset 0 -2px 0 var(--green);
        }}

        input[type="range"] {{
            width: 100%;
            accent-color: var(--green);
        }}

        .grid-2 {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 13px;
        }}

        .grid-3 {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
        }}

        .primary-btn {{
            width: 100%;
            min-height: 48px;
            margin-top: 7px;
            border: 0;
            border-radius: 0;
            background: var(--navy);
            color: #ffffff;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: .05em;
            text-transform: uppercase;
            cursor: pointer;
            transition: background .18s ease, transform .18s ease;
        }}

        .primary-btn:hover {{
            background: var(--green-dark);
            transform: translateY(-1px);
        }}

        .empty-state {{
            min-height: 570px;
            padding: 48px;
            background: #ffffff;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
        }}

        .empty-mark {{
            width: 50px;
            height: 50px;
            margin: 0 auto 17px;
            display: grid;
            place-items: center;
            background: var(--blue-soft);
            color: #9aa8b3;
        }}

        .empty-title {{
            margin: 0 0 6px;
            color: #6d7b86;
            font-size: 14px;
            font-weight: 600;
        }}

        .empty-copy {{
            max-width: 280px;
            margin: 0 auto;
            color: #9aa6af;
            font-size: 12px;
            line-height: 1.6;
        }}

        .result-stack {{ display: grid; gap: 20px; }}

        .result-panel {{ background: #ffffff; padding: 26px; box-shadow: 0 7px 22px rgba(11, 31, 51, .045); }}

        .result-top {{
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 20px;
            margin-bottom: 24px;
        }}

        .result-name {{
            margin: 0 0 5px;
            color: var(--navy);
            font-size: 24px;
            font-weight: 700;
            letter-spacing: -.035em;
        }}

        .result-meta {{ color: #8a969f; font-size: 11px; }}

        .segment {{
            padding: 7px 10px;
            background: var(--green-soft);
            color: var(--green-dark);
            font-size: 10px;
            font-weight: 700;
            letter-spacing: .06em;
            text-transform: uppercase;
            white-space: nowrap;
        }}

        .profile-grid {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            background: #edf1f3;
            gap: 1px;
            margin-bottom: 24px;
        }}

        .profile-item {{ padding: 17px; background: #ffffff; }}

        .small-label {{
            margin-bottom: 6px;
            color: #89959e;
            font-size: 9px;
            font-weight: 700;
            letter-spacing: .11em;
            text-transform: uppercase;
        }}

        .profile-value {{ color: var(--navy); font-size: 16px; font-weight: 700; }}

        .risk-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 14px;
        }}

        .risk-box {{ padding: 18px; background: #f7f9fa; }}

        .risk-head {{
            display: flex;
            justify-content: space-between;
            align-items: end;
            gap: 12px;
            margin-bottom: 11px;
        }}

        .risk-name {{
            color: #66737d;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: .1em;
            text-transform: uppercase;
        }}

        .risk-label {{ font-size: 13px; font-weight: 700; }}

        .risk-track {{ width: 100%; height: 5px; background: #e2e7ea; overflow: hidden; }}

        .risk-fill {{ height: 100%; }}

        .risk-note {{ margin: 9px 0 0; color: #87939c; font-size: 10px; line-height: 1.5; }}

        .decision {{
            padding: 32px 28px;
            background: var(--navy);
            color: #ffffff;
            text-align: center;
        }}

        .decision-label {{
            margin-bottom: 9px;
            color: #a9b8c4;
            font-size: 9px;
            font-weight: 700;
            letter-spacing: .18em;
            text-transform: uppercase;
        }}

        .decision-value {{
            margin-bottom: 12px;
            font-size: 34px;
            line-height: 1;
            font-weight: 700;
            letter-spacing: -.04em;
        }}

        .decision-copy {{
            max-width: 590px;
            margin: 0 auto;
            color: #c7d1d8;
            font-size: 12px;
            line-height: 1.7;
        }}

        .fraud-hero {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 30px;
            padding: 32px;
            margin-bottom: 25px;
            background: var(--navy);
            color: #ffffff;
        }}

        .fraud-hero h1 {{ color: #ffffff; font-size: 25px; margin-bottom: 9px; }}
        .fraud-hero p {{ max-width: 700px; color: #c6d2dc; font-size: 12px; line-height: 1.7; margin: 0; }}

        .hero-stats {{ display: flex; flex: 0 0 auto; gap: 25px; }}
        .hero-stat {{ text-align: right; }}
        .hero-stat-value {{ font-size: 22px; font-weight: 700; }}
        .hero-stat-label {{ color: #9fb1bf; font-size: 9px; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; }}

        .fraud-layout {{
            display: grid;
            grid-template-columns: 300px minmax(0, 1fr);
            gap: 25px;
        }}

        .control-panel {{ background: #ffffff; padding: 22px; box-shadow: 0 7px 22px rgba(11,31,51,.045); }}

        .control-title {{
            margin-bottom: 18px;
            color: #6c7983;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: .13em;
            text-transform: uppercase;
        }}

        .sim-btn {{
            width: 100%;
            min-height: 43px;
            margin-bottom: 9px;
            border: 0;
            border-radius: 0;
            cursor: pointer;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: .08em;
            text-transform: uppercase;
        }}

        .sim-fraud {{ background: var(--red-soft); color: var(--red); }}
        .sim-legit {{ background: var(--green-soft); color: var(--green-dark); }}
        .sim-random {{ background: var(--blue-soft); color: var(--navy); }}
        .sim-btn:hover {{ filter: brightness(.97); }}

        .metadata {{
            margin-top: 18px;
            padding: 18px;
            background: var(--navy-2);
            color: #b9c6cf;
            font: 10px/1.9 Consolas, monospace;
        }}

        .metadata strong {{ color: #79c7a8; font-weight: 500; }}

        .fraud-result {{ background: #ffffff; padding: 28px; box-shadow: 0 7px 22px rgba(11,31,51,.045); }}

        .transaction-head {{
            display: flex;
            justify-content: space-between;
            gap: 20px;
            margin-bottom: 25px;
        }}

        .txn-id {{ margin-bottom: 6px; color: #9aa5ad; font-size: 9px; font-weight: 700; letter-spacing: .13em; }}
        .txn-amount {{ color: var(--navy); font-size: 34px; font-weight: 700; letter-spacing: -.04em; }}
        .txn-location {{ color: #596873; font-size: 12px; font-weight: 700; text-align: right; }}
        .txn-time {{ margin-top: 4px; color: #9aa5ad; font-size: 10px; text-align: right; }}

        .fraud-stats {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1px; background: #e8edef; margin-bottom: 22px; }}
        .fraud-stat {{ padding: 17px; background: #f8fafb; text-align: center; }}
        .fraud-stat-value {{ font-size: 22px; font-weight: 700; }}

        .fraud-bar {{ width: 100%; height: 6px; background: #e4e9ec; overflow: hidden; margin-bottom: 25px; }}
        .fraud-bar-fill {{ height: 100%; }}

        .verdict {{ padding: 17px 20px; text-align: center; font-size: 13px; font-weight: 700; letter-spacing: .07em; text-transform: uppercase; }}
        .verdict-fraud {{ background: var(--red-soft); color: var(--red); }}
        .verdict-legit {{ background: var(--green-soft); color: var(--green-dark); }}

        .notice {{ padding: 14px; background: var(--amber-soft); color: #6d5200; font-size: 11px; line-height: 1.6; }}
        .notice strong {{ font-weight: 700; }}

        .dashboard-grid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 1px;
            background: #e9eef1;
            margin-bottom: 30px;
        }}

        .dashboard-stat {{ padding: 22px; background: #ffffff; min-height: 112px; }}
        .dashboard-stat .metric-value {{ margin-top: 6px; }}

        .table-panel {{ background: #ffffff; box-shadow: 0 7px 22px rgba(11,31,51,.045); overflow-x: auto; }}
        table {{ width: 100%; min-width: 800px; border-collapse: collapse; }}
        th {{ padding: 15px 18px; background: #f8fafb; color: #7d8992; font-size: 9px; font-weight: 700; letter-spacing: .11em; text-transform: uppercase; text-align: left; }}
        td {{ padding: 15px 18px; color: #53616c; font-size: 11px; box-shadow: inset 0 -1px 0 #eef1f3; }}
        tr:hover td {{ background: #fbfcfc; }}

        .status {{ display: inline-block; padding: 5px 8px; font-size: 9px; font-weight: 700; letter-spacing: .07em; text-transform: uppercase; }}
        .status-APPROVED {{ background: var(--green-soft); color: var(--green-dark); }}
        .status-REJECTED {{ background: var(--red-soft); color: var(--red); }}
        .status-REVIEW {{ background: var(--amber-soft); color: var(--amber); }}

        .mobile-nav {{ display: none; }}

        @media (max-width: 900px) {{
            .layout, .fraud-layout {{ grid-template-columns: 1fr; }}
            .metric-grid, .dashboard-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
            .fraud-hero {{ align-items: flex-start; flex-direction: column; }}
            .hero-stat {{ text-align: left; }}
            .hero-stats {{ width: 100%; }}
        }}

        @media (max-width: 640px) {{
            .topbar-inner {{ min-height: 64px; padding: 0 16px; }}
            .nav-links {{ gap: 12px; height: 64px; }}
            .nav-links a {{ height: 64px; font-size: 10px; }}
            .brand {{ font-size: 16px; }}
            .brand-mark {{ width: 30px; height: 30px; }}
            .page {{ padding: 28px 16px 45px; }}
            .page-title {{ font-size: 25px; }}
            .metric-grid, .dashboard-grid {{ grid-template-columns: 1fr 1fr; }}
            .grid-2, .grid-3, .risk-grid, .profile-grid {{ grid-template-columns: 1fr; }}
            .result-top, .transaction-head {{ flex-direction: column; }}
            .txn-location, .txn-time {{ text-align: left; }}
            .fraud-hero, .result-panel, .fraud-result {{ padding: 22px; }}
            .hero-stats {{ gap: 18px; }}
        }}
    </style>
</head>
<body>
    <nav class="topbar">
        <div class="topbar-inner">
            <a href="/" class="brand">
                <span class="brand-mark">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M12 3 20 6v5c0 5.1-3.1 8.6-8 10-4.9-1.4-8-4.9-8-10V6l8-3Z"/>
                        <path d="m8.5 12 2.2 2.2 4.8-5"/>
                    </svg>
                </span>
                <span>FinanSafe</span>
            </a>
            <div class="nav-links">
                <a href="/" class="{nav_assess_cls}">Assessment</a>
                <a href="/fraud" class="{nav_fraud_cls}">Fraud Detection</a>
                <a href="/dashboard" class="{nav_dash_cls}">Dashboard</a>
            </div>
        </div>
    </nav>

    <main class="page">
        {body}
    </main>
</body>
</html>
"""


# -----------------------------------------------------------------------------
# ASSESSMENT PAGE
# -----------------------------------------------------------------------------
ASSESS_BODY = """
<section class="section-head">
    <div class="eyebrow">Credit intelligence platform</div>
    <h1 class="page-title">Customer Assessment</h1>
    <p class="page-subtitle">
        Evaluate credit behaviour, predicted default probability and customer
        segment before making a lending decision.
    </p>
</section>

<div class="metric-grid">
    <div class="metric">
        <div class="metric-label">Active Models</div>
        <div class="metric-value">3</div>
    </div>
    <div class="metric">
        <div class="metric-label">Model Precision</div>
        <div class="metric-value">92.4%</div>
    </div>
    <div class="metric">
        <div class="metric-label">AUC Score</div>
        <div class="metric-value">0.82</div>
    </div>
    <div class="metric">
        <div class="metric-label">Analysis Latency</div>
        <div class="metric-value">1.2ms</div>
    </div>
</div>

<div class="layout">
    <section class="panel">
        <div class="panel-head">
            <h2 class="panel-title">Applicant Details</h2>
        </div>

        <form action="/analyze" method="post" class="form">
            <div class="field">
                <label class="field-label" for="name">Full Name</label>
                <input id="name" type="text" name="name" class="input" placeholder="Applicant name" required>
            </div>

            <div class="grid-2">
                <div class="field">
                    <label class="field-label" for="age">Age</label>
                    <input id="age" type="number" name="age" class="input" placeholder="32" min="18" required>
                </div>
                <div class="field">
                    <label class="field-label" for="dependents">Dependents</label>
                    <input id="dependents" type="number" name="dependents" class="input" placeholder="2" min="0" value="0" required>
                </div>
            </div>

            <div class="grid-2">
                <div class="field">
                    <label class="field-label" for="income">Monthly Income</label>
                    <input id="income" type="number" name="income" class="input" placeholder="5000" min="0" step="0.01" required>
                </div>
                <div class="field">
                    <label class="field-label" for="annual_income">Annual Income</label>
                    <input id="annual_income" type="number" name="annual_income" class="input" placeholder="60000" min="0" step="0.01" required>
                </div>
            </div>

            <div class="field">
                <label class="field-label" for="spending_score">Spending Score: <span id="spendingValue">50</span></label>
                <input id="spending_score" type="range" name="spending_score" min="1" max="100" value="50" oninput="document.getElementById('spendingValue').textContent=this.value">
            </div>

            <div class="grid-2">
                <div class="field">
                    <label class="field-label" for="debt_ratio">Debt Ratio</label>
                    <input id="debt_ratio" type="number" step="0.01" name="debt_ratio" class="input" placeholder="0.45" min="0" required>
                </div>
                <div class="field">
                    <label class="field-label" for="utilization">Utilization</label>
                    <input id="utilization" type="number" step="0.01" name="utilization" class="input" placeholder="0.30" min="0" required>
                </div>
            </div>

            <div class="field">
                <label class="field-label">Payment Delinquencies</label>
                <div class="grid-3">
                    <input type="number" name="late_30_59" class="input" placeholder="30d" value="0" min="0">
                    <input type="number" name="late_60_89" class="input" placeholder="60d" value="0" min="0">
                    <input type="number" name="late_90" class="input" placeholder="90d" value="0" min="0">
                </div>
            </div>

            <div class="field">
                <label class="field-label" for="loan_amount">Requested Loan Amount</label>
                <input id="loan_amount" type="number" name="loan_amount" class="input" placeholder="50000" min="0" step="0.01" required>
            </div>

            <div class="grid-2">
                <div class="field">
                    <label class="field-label" for="credit_lines">Credit Lines</label>
                    <input id="credit_lines" type="number" name="credit_lines" class="input" value="5" min="0">
                </div>
                <div class="field">
                    <label class="field-label" for="real_estate_loans">Real Estate Loans</label>
                    <input id="real_estate_loans" type="number" name="real_estate_loans" class="input" value="0" min="0">
                </div>
            </div>

            <button type="submit" class="primary-btn">Run Predictive Analysis</button>
        </form>
    </section>

    <section>
        {results_html}
    </section>
</div>
"""


# -----------------------------------------------------------------------------
# FRAUD PAGE
# -----------------------------------------------------------------------------
FRAUD_BODY = """
<section class="section-head">
    <div class="eyebrow">Transaction monitoring</div>
    <h1 class="page-title">Fraud Detection</h1>
    <p class="page-subtitle">
        Test transaction records against the fraud-classification model and
        inspect the model's risk confidence and predicted transaction status.
    </p>
</section>

<div class="fraud-hero">
    <div>
        <h1>Real-time Transaction Shield</h1>
        <p>
            PCA-transformed transaction features are evaluated by the trained
            fraud classifier to identify suspicious banking activity.
        </p>
    </div>
    <div class="hero-stats">
        <div class="hero-stat">
            <div class="hero-stat-value">99.1%</div>
            <div class="hero-stat-label">Accuracy</div>
        </div>
        <div class="hero-stat">
            <div class="hero-stat-value">0.1s</div>
            <div class="hero-stat-label">Processing</div>
        </div>
    </div>
</div>

<div class="fraud-layout">
    <div>
        <section class="control-panel">
            <div class="control-title">Simulation Controls</div>
            {controls_html}
        </section>

        <div class="metadata">
            <strong># Model metadata</strong><br>
            Type: XGBoostClassifier<br>
            Features: V1-V28 (PCA), Amount, Time<br>
            Scale: StandardScaler<br>
            Transaction classification enabled
        </div>
    </div>

    <section>
        {fraud_html}
    </section>
</div>
"""


# -----------------------------------------------------------------------------
# DASHBOARD PAGE
# -----------------------------------------------------------------------------
DASHBOARD_BODY = """
<section class="section-head">
    <div class="eyebrow">Decision activity</div>
    <h1 class="page-title">Assessment Dashboard</h1>
    <p class="page-subtitle">
        Review historical customer assessments and the decisions generated by
        the FinanSafe lending-risk workflow.
    </p>
</section>

<div class="dashboard-grid">
    <div class="dashboard-stat">
        <div class="metric-label">Total Assessments</div>
        <div class="metric-value">{total}</div>
    </div>
    <div class="dashboard-stat">
        <div class="metric-label">Approved</div>
        <div class="metric-value" style="color:#167c5a">{approved}</div>
    </div>
    <div class="dashboard-stat">
        <div class="metric-label">Rejected</div>
        <div class="metric-value" style="color:#b42318">{rejected}</div>
    </div>
    <div class="dashboard-stat">
        <div class="metric-label">Under Review</div>
        <div class="metric-value" style="color:#946200">{review}</div>
    </div>
</div>

<section class="table-panel">
    {table_html}
</section>
"""


# -----------------------------------------------------------------------------
# RESULT RENDERERS
# -----------------------------------------------------------------------------
def render_results(results):
    if not results:
        return """
        <div class="empty-state">
            <div>
                <div class="empty-mark">
                    <svg xmlns="http://www.w3.org/2000/svg" width="23" height="23" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                        <path d="M14 2v6h6"/>
                        <path d="M8 13h8M8 17h8"/>
                    </svg>
                </div>
                <h3 class="empty-title">Ready for assessment</h3>
                <p class="empty-copy">
                    Complete the applicant profile to generate the credit-risk report.
                </p>
            </div>
        </div>
        """

    name = html.escape(str(results['name']))
    segment_name = html.escape(str(results['segment_name']))
    rec_verdict = html.escape(str(results['rec_verdict']))
    rec_reason = html.escape(str(results['rec_reason']))
    behavioral_label = html.escape(str(results['behavioral_label']))
    default_label = html.escape(str(results['default_label']))

    return f"""
    <div class="result-stack">
        <section class="result-panel">
            <div class="result-top">
                <div>
                    <h2 class="result-name">{name}</h2>
                    <div class="result-meta">
                        Profile analysis · {datetime.now().strftime('%b %d, %Y')}
                    </div>
                </div>
                <div class="segment">{segment_name}</div>
            </div>

            <div class="profile-grid">
                <div class="profile-item">
                    <div class="small-label">Monthly Income</div>
                    <div class="profile-value">${html.escape(str(results['income']))}</div>
                </div>
                <div class="profile-item">
                    <div class="small-label">Loan Amount</div>
                    <div class="profile-value">${html.escape(str(results['loan_amount']))}</div>
                </div>
                <div class="profile-item">
                    <div class="small-label">Applicant Age</div>
                    <div class="profile-value">{html.escape(str(results['age']))}</div>
                </div>
            </div>

            <div class="risk-grid">
                <div class="risk-box">
                    <div class="risk-head">
                        <span class="risk-name">Behavioral Risk</span>
                        <span class="risk-label" style="color:{results['behavioral_color']}">{behavioral_label}</span>
                    </div>
                    <div class="risk-track">
                        <div class="risk-fill" style="width:{results['behavioral_score']}%;background:{results['behavioral_color']}"></div>
                    </div>
                    <p class="risk-note">
                        Based on payment history, debt exposure and requested loan size.
                    </p>
                </div>

                <div class="risk-box">
                    <div class="risk-head">
                        <span class="risk-name">Default Probability</span>
                        <span class="risk-label" style="color:{results['default_color']}">{default_label}</span>
                    </div>
                    <div class="risk-track">
                        <div class="risk-fill" style="width:{results['default_score']}%;background:{results['default_color']}"></div>
                    </div>
                    <p class="risk-note">
                        ML-generated probability: {results['default_score']}%
                    </p>
                </div>
            </div>
        </section>

        <section class="decision">
            <div class="decision-label">Final Determination</div>
            <div class="decision-value" style="color:{results['rec_color']}">{rec_verdict}</div>
            <p class="decision-copy">{rec_reason}</p>
        </section>
    </div>
    """


def render_fraud(fraud_result):
    if not fraud_result:
        return """
        <div class="empty-state" style="min-height:420px">
            <div>
                <div class="empty-mark">
                    <svg xmlns="http://www.w3.org/2000/svg" width="23" height="23" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="11" cy="11" r="7.5"/>
                        <path d="m20 20-3.7-3.7"/>
                    </svg>
                </div>
                <h3 class="empty-title">Monitoring Stream</h3>
                <p class="empty-copy">
                    Select a transaction scenario to run automated verification.
                </p>
            </div>
        </div>
        """

    verdict = html.escape(str(fraud_result['verdict']))
    location = html.escape(str(fraud_result['location']))
    actual = html.escape(str(fraud_result['actual']))

    return f"""
    <div class="fraud-result">
        <div class="transaction-head">
            <div>
                <div class="txn-id">TXN ID: {html.escape(str(fraud_result['txn_id']))}</div>
                <div class="txn-amount">${html.escape(str(fraud_result['amount']))}</div>
            </div>
            <div>
                <div class="txn-location">{location}</div>
                <div class="txn-time">{html.escape(str(fraud_result['time']))}</div>
            </div>
        </div>

        <div class="fraud-stats">
            <div class="fraud-stat">
                <div class="small-label">Risk Confidence</div>
                <div class="fraud-stat-value" style="color:{fraud_result['risk_color']}">{fraud_result['risk_score']}%</div>
            </div>
            <div class="fraud-stat">
                <div class="small-label">Known Label</div>
                <div class="fraud-stat-value" style="color:#0b1f33">{actual}</div>
            </div>
        </div>

        <div class="fraud-bar">
            <div class="fraud-bar-fill" style="width:{fraud_result['risk_score']}%;background:{fraud_result['risk_color']}"></div>
        </div>

        <div class="verdict {fraud_result['verdict_class']}">
            {verdict}
        </div>
    </div>
    """


# -----------------------------------------------------------------------------
# ROUTES
# -----------------------------------------------------------------------------
@app.route('/')
def home():
    html_page = BASE.format(
        title='Customer Assessment',
        nav_assess_cls='active',
        nav_fraud_cls='',
        nav_dash_cls='',
        body=ASSESS_BODY.format(results_html=render_results(None))
    )
    return html_page


@app.route('/fraud')
def fraud_page():
    if df_fraud is not None and fraud_model is not None:
        controls = """
        <form action="/fraud_predict" method="post">
            <input type="hidden" name="type" value="fraud">
            <button type="submit" class="sim-btn sim-fraud">Simulate Fraud</button>
        </form>

        <form action="/fraud_predict" method="post">
            <input type="hidden" name="type" value="legit">
            <button type="submit" class="sim-btn sim-legit">Simulate Legitimate</button>
        </form>

        <form action="/fraud_predict" method="post">
            <input type="hidden" name="type" value="random">
            <button type="submit" class="sim-btn sim-random">Random Stream</button>
        </form>
        """
    else:
        controls = """
        <div class="notice">
            <strong>Transaction dataset unavailable.</strong><br>
            Upload <code>creditcard.csv</code> to the project root to enable
            transaction simulation in the local environment.
        </div>
        """

    html_page = BASE.format(
        title='Fraud Detection',
        nav_assess_cls='',
        nav_fraud_cls='active',
        nav_dash_cls='',
        body=FRAUD_BODY.format(
            controls_html=controls,
            fraud_html=render_fraud(None)
        )
    )
    return html_page


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.form

        behavioral_score, behavioral_label, behavioral_color = calculate_behavioral_risk(
            float(data['debt_ratio']),
            float(data['utilization']),
            int(data['late_30_59']),
            int(data['late_60_89']),
            int(data['late_90']),
            int(data['dependents']),
            float(data['income']),
            float(data['loan_amount'])
        )

        if default_model is None:
            raise RuntimeError(
                'Credit default model is unavailable. Check the model file '
                'or the CREDIT_MODEL_URL environment variable.'
            )

        default_features = pd.DataFrame([[
            float(data['utilization']),
            int(data['age']),
            int(data['late_30_59']),
            float(data['debt_ratio']),
            float(data['income']),
            int(data['credit_lines']),
            int(data['late_90']),
            int(data['real_estate_loans']),
            int(data['late_60_89']),
            int(data['dependents'])
        ]], columns=[
            'RevolvingUtilizationOfUnsecuredLines',
            'age',
            'NumberOfTime30-59DaysPastDueNotWorse',
            'DebtRatio',
            'MonthlyIncome',
            'NumberOfOpenCreditLinesAndLoans',
            'NumberOfTimes90DaysLate',
            'NumberRealEstateLoansOrLines',
            'NumberOfTime60-89DaysPastDueNotWorse',
            'NumberOfDependents'
        ])

        default_prob = default_model.predict_proba(default_features)[0][1]
        default_score = round(default_prob * 100, 1)

        if default_score >= 60:
            default_label = 'HIGH RISK'
            default_color = '#b42318'
        elif default_score >= 30:
            default_label = 'MEDIUM RISK'
            default_color = '#946200'
        else:
            default_label = 'LOW RISK'
            default_color = '#167c5a'

        annual_income_k = float(data['annual_income']) / 1000

        if segment_model is None:
            raise RuntimeError(
                'Customer segmentation model is unavailable.'
            )

        segment = int(
            segment_model.predict(
                np.array([[annual_income_k, float(data['spending_score'])]])
            )[0]
        )
        segment_name = segment_names.get(segment, 'Unknown')

        rec_verdict, rec_reason, rec_color = get_loan_recommendation(
            default_prob,
            behavioral_score,
            segment,
            float(data['income']),
            float(data['loan_amount'])
        )

        with open(DATA_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                data['name'],
                data['age'],
                data['income'],
                data['annual_income'],
                data['spending_score'],
                data['debt_ratio'],
                data['utilization'],
                data['dependents'],
                data['credit_lines'],
                data['real_estate_loans'],
                data['late_30_59'],
                data['late_60_89'],
                data['late_90'],
                data['loan_amount'],
                behavioral_score,
                behavioral_label,
                default_score,
                default_label,
                segment_name,
                rec_verdict
            ])

        results = {
            'name': data['name'],
            'age': data['age'],
            'income': data['income'],
            'loan_amount': data['loan_amount'],
            'segment_name': segment_name,
            'behavioral_score': behavioral_score,
            'behavioral_label': behavioral_label,
            'behavioral_color': behavioral_color,
            'default_score': default_score,
            'default_label': default_label,
            'default_color': default_color,
            'rec_verdict': rec_verdict,
            'rec_reason': rec_reason,
            'rec_color': rec_color
        }

        html_page = BASE.format(
            title='Customer Assessment',
            nav_assess_cls='active',
            nav_fraud_cls='',
            nav_dash_cls='',
            body=ASSESS_BODY.format(results_html=render_results(results))
        )
        return html_page

    except Exception as e:
        message = html.escape(str(e))
        error_body = f"""
        <section class="section-head">
            <div class="eyebrow">Assessment error</div>
            <h1 class="page-title">Analysis could not be completed</h1>
            <p class="page-subtitle">The application reached the analysis route, but the prediction could not be completed.</p>
        </section>
        <section class="panel" style="padding:24px;max-width:760px">
            <div class="notice" style="background:#fff1f0;color:#8f2118">
                <strong>Technical message:</strong><br>{message}
            </div>
            <p style="margin:18px 0 0;color:#687582;font-size:12px;line-height:1.7">
                If this occurs on Render, verify that the required model files are
                available and that CREDIT_MODEL_URL points to the published credit model.
            </p>
        </section>
        """
        return BASE.format(
            title='Assessment Error',
            nav_assess_cls='active',
            nav_fraud_cls='',
            nav_dash_cls='',
            body=error_body
        ), 500


@app.route('/fraud_predict', methods=['POST'])
def fraud_predict():
    if df_fraud is None or fraud_model is None:
        return redirect(url_for('fraud_page'))

    if fraud_samples is None or legit_samples is None:
        return redirect(url_for('fraud_page'))

    transaction_type = request.form.get('type', 'random')

    if transaction_type == 'fraud':
        idx = random.randint(0, len(fraud_samples) - 1)
        sample = fraud_samples.iloc[idx]
        amount = fraud_amounts.iloc[idx]
        time_v = fraud_times.iloc[idx]
        actual = 1

    elif transaction_type == 'legit':
        idx = random.randint(0, len(legit_samples) - 1)
        sample = legit_samples.iloc[idx]
        amount = legit_amounts.iloc[idx]
        time_v = legit_times.iloc[idx]
        actual = 0

    else:
        if random.random() < 0.5:
            idx = random.randint(0, len(fraud_samples) - 1)
            sample = fraud_samples.iloc[idx]
            amount = fraud_amounts.iloc[idx]
            time_v = fraud_times.iloc[idx]
            actual = 1
        else:
            idx = random.randint(0, len(legit_samples) - 1)
            sample = legit_samples.iloc[idx]
            amount = legit_amounts.iloc[idx]
            time_v = legit_times.iloc[idx]
            actual = 0

    feature_cols = [col for col in df_fraud.columns if col != 'Class']

    df_input = pd.DataFrame(
        sample[feature_cols].values.reshape(1, -1),
        columns=feature_cols
    )

    try:
        probability = fraud_model.predict_proba(df_input)[0][1]
        prediction = fraud_model.predict(df_input)[0]
    except Exception as e:
        error_body = f"""
        <section class="section-head">
            <div class="eyebrow">Fraud model error</div>
            <h1 class="page-title">Transaction analysis failed</h1>
            <p class="page-subtitle">The transaction was selected, but the fraud model could not process its feature set.</p>
        </section>
        <section class="panel" style="padding:24px">
            <div class="notice" style="background:#fff1f0;color:#8f2118">
                <strong>Technical message:</strong><br>{html.escape(str(e))}
            </div>
        </section>
        """
        return BASE.format(
            title='Fraud Error',
            nav_assess_cls='',
            nav_fraud_cls='active',
            nav_dash_cls='',
            body=error_body
        ), 500

    risk_score = round(probability * 100, 1)

    if risk_score >= 60:
        risk_color = '#b42318'
    elif risk_score >= 30:
        risk_color = '#946200'
    else:
        risk_color = '#167c5a'

    fraud_result = {
        'txn_id': f'TXN-{random.randint(100000, 999999)}',
        'amount': f'{amount:.2f}',
        'time': seconds_to_time(time_v),
        'location': random.choice(LOCATIONS),
        'risk_score': risk_score,
        'risk_color': risk_color,
        'verdict': (
            'Fraud detected — Transaction blocked'
            if prediction == 1
            else 'Legitimate — Transaction approved'
        ),
        'verdict_class': (
            'verdict-fraud' if prediction == 1 else 'verdict-legit'
        ),
        'actual': 'FRAUD' if actual == 1 else 'Legitimate'
    }

    controls = """
    <form action="/fraud_predict" method="post">
        <input type="hidden" name="type" value="fraud">
        <button type="submit" class="sim-btn sim-fraud">Simulate Fraud</button>
    </form>
    <form action="/fraud_predict" method="post">
        <input type="hidden" name="type" value="legit">
        <button type="submit" class="sim-btn sim-legit">Simulate Legitimate</button>
    </form>
    <form action="/fraud_predict" method="post">
        <input type="hidden" name="type" value="random">
        <button type="submit" class="sim-btn sim-random">Random Stream</button>
    </form>
    """

    html_page = BASE.format(
        title='Fraud Detection',
        nav_assess_cls='',
        nav_fraud_cls='active',
        nav_dash_cls='',
        body=FRAUD_BODY.format(
            controls_html=controls,
            fraud_html=render_fraud(fraud_result)
        )
    )
    return html_page


@app.route('/dashboard')
def dashboard():
    records = []
    total = 0
    approved = 0
    rejected = 0
    review = 0

    if os.path.exists(DATA_FILE):
        try:
            df_data = pd.read_csv(DATA_FILE)

            if not df_data.empty:
                records = df_data.to_dict('records')
                total = len(records)
                approved = sum(
                    1 for r in records
                    if r.get('recommendation') == 'APPROVED'
                )
                rejected = sum(
                    1 for r in records
                    if r.get('recommendation') == 'REJECTED'
                )
                review = sum(
                    1 for r in records
                    if r.get('recommendation') == 'REVIEW'
                )
        except Exception as e:
            print(f'Warning: Dashboard data error: {e}')

    rows = ''

    # Newest assessments are displayed first while preserving all records.
    for r in reversed(records):
        rec = str(r.get('recommendation', ''))
        safe_rec = html.escape(rec)
        status_class = (
            'status-APPROVED' if rec == 'APPROVED'
            else 'status-REJECTED' if rec == 'REJECTED'
            else 'status-REVIEW' if rec == 'REVIEW'
            else ''
        )

        rows += f"""
        <tr>
            <td>{html.escape(str(r.get('timestamp', '')))}</td>
            <td style="font-weight:700;color:#0b1f33">{html.escape(str(r.get('name', '')))}</td>
            <td>{html.escape(str(r.get('age', '')))}</td>
            <td>${html.escape(str(r.get('income', '')))}</td>
            <td style="font-weight:700;color:#167c5a">${html.escape(str(r.get('loan_amount', '')))}</td>
            <td>{html.escape(str(r.get('default_label', '')))}</td>
            <td><span class="status {status_class}">{safe_rec}</span></td>
        </tr>
        """

    if not records:
        rows = """
        <tr>
            <td colspan="7" style="padding:50px;text-align:center;color:#9aa5ad">
                No assessment activity has been recorded yet.
            </td>
        </tr>
        """

    table_html = f"""
    <table>
        <thead>
            <tr>
                <th>Timestamp</th>
                <th>Applicant</th>
                <th>Age</th>
                <th>Monthly Income</th>
                <th>Loan</th>
                <th>ML Risk</th>
                <th>Outcome</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
    """

    html_page = BASE.format(
        title='Dashboard',
        nav_assess_cls='',
        nav_fraud_cls='',
        nav_dash_cls='active',
        body=DASHBOARD_BODY.format(
            total=total,
            approved=approved,
            rejected=rejected,
            review=review,
            table_html=table_html
        )
    )
    return html_page


# -----------------------------------------------------------------------------
# LOCAL DEVELOPMENT
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
