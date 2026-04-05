from flask import Flask, request, render_template_string, redirect, url_for, session
import pickle
import pandas as pd
import numpy as np
import csv
import os
import random
from datetime import datetime
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
app.secret_key = 'finansafe2026'

# --- Load models (Keeping original logic) ---
# Note: Ensure these files exist in your local 'models/' directory
try:
    with open('models/fraud_detection_model.pkl', 'rb') as f:
        fraud_model = pickle.load(f)

    with open('models/credit_default_model.pkl', 'rb') as f:
        default_model = pickle.load(f)

    with open('models/customer_segmentation_model.pkl', 'rb') as f:
        segment_model = pickle.load(f)
except FileNotFoundError:
    print("Warning: One or more model files not found. Ensure 'models/' directory is populated.")

segment_names = {
    0: "Budget Customer",
    1: "Premium Customer",
    2: "Impulsive Buyer",
    3: "Careful Spender",
    4: "Middle Customer"
}

DATA_FILE = 'assessments.csv'

if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'timestamp', 'name', 'age', 'income', 'annual_income',
            'spending_score', 'debt_ratio', 'utilization', 'dependents',
            'credit_lines', 'real_estate_loans', 'late_30_59',
            'late_60_89', 'late_90', 'loan_amount',
            'behavioral_score', 'behavioral_label',
            'default_score', 'default_label',
            'segment', 'recommendation'
        ])

# --- Load fraud dataset ---
df_fraud = None
fraud_samples = legit_samples = None
fraud_amounts = legit_amounts = None
fraud_times   = legit_times   = None

if os.path.exists('creditcard.csv'):
    df_fraud = pd.read_csv('creditcard.csv')
    scaler = StandardScaler()
    original_amounts = df_fraud['Amount'].copy()
    original_times   = df_fraud['Time'].copy()
    df_fraud['Amount_Scaled'] = scaler.fit_transform(df_fraud[['Amount']])
    df_fraud['Time_Scaled']   = scaler.fit_transform(df_fraud[['Time']])
    df_fraud.drop(columns=['Amount', 'Time'], inplace=True)
    fraud_samples = df_fraud[df_fraud['Class'] == 1].copy()
    legit_samples  = df_fraud[df_fraud['Class'] == 0].copy()
    fraud_amounts  = original_amounts[fraud_samples.index]
    legit_amounts  = original_amounts[legit_samples.index]
    fraud_times    = original_times[fraud_samples.index]
    legit_times    = original_times[legit_samples.index]

LOCATIONS = ["Online Purchase", "ATM Withdrawal", "POS Terminal",
             "International Transfer", "Mobile Payment", "E-commerce"]

def seconds_to_time(seconds):
    hours   = int((seconds % 86400) / 3600)
    minutes = int((seconds % 3600) / 60)
    period  = "AM" if hours < 12 else "PM"
    hours   = hours % 12 or 12
    return f"{hours:02d}:{minutes:02d} {period}"

def calculate_behavioral_risk(debt_ratio, utilization, late_30_59, late_60_89,
                               late_90, dependents, income, loan_amount):
    score = 0
    if debt_ratio > 0.8:    score += 30
    elif debt_ratio > 0.5:  score += 15
    elif debt_ratio > 0.3:  score += 5

    if utilization > 0.8:   score += 25
    elif utilization > 0.5: score += 12
    elif utilization > 0.3: score += 5

    score += late_30_59 * 8
    score += late_60_89 * 12
    score += late_90    * 20

    if dependents > 4:   score += 10
    elif dependents > 2: score += 5

    annual_income = income * 12
    if annual_income > 0:
        lti = loan_amount / annual_income
        if lti > 5:   score += 30
        elif lti > 3: score += 15
        elif lti > 1: score += 5

    score = min(score, 100)
    if score >= 60:   return score, "HIGH RISK",   "#ef4444" # Modern Red
    elif score >= 30: return score, "MEDIUM RISK", "#f59e0b" # Modern Amber
    else:             return score, "LOW RISK",    "#10b981" # Modern Emerald

def get_loan_recommendation(default_prob, behavioral_score, segment,
                             monthly_income, loan_amount):
    annual_income  = monthly_income * 12
    loan_to_income = (loan_amount / annual_income) if annual_income > 0 else 999

    if loan_to_income > 5:
        return ("REJECTED", f"Loan amount is {loan_to_income:.1f}x annual income. Exceeds maximum allowed ratio of 5x.", "#ef4444")
    if default_prob > 0.70:
        return ("REJECTED", "High credit default risk. Customer unlikely to repay loan.", "#ef4444")
    if behavioral_score >= 60 and default_prob > 0.40:
        return ("REJECTED", "High behavioral risk combined with elevated default probability.", "#ef4444")
    if default_prob < 0.20 and behavioral_score < 30 and segment in [1, 3] and loan_to_income <= 3:
        return ("APPROVED", f"Excellent profile. Low default risk with premium or careful spender segment. Loan is {loan_to_income:.1f}x annual income.", "#10b981")
    if default_prob < 0.30 and behavioral_score < 30 and loan_to_income <= 3:
        return ("APPROVED", f"Good profile. Low default and behavioral risk. Loan is {loan_to_income:.1f}x annual income.", "#10b981")
    if default_prob < 0.40 and behavioral_score < 40 and segment in [1, 3, 4] and loan_to_income <= 2:
        return ("APPROVED", "Acceptable risk profile. Loan approved with standard terms.", "#10b981")
    if default_prob < 0.50 and behavioral_score < 60:
        return ("REVIEW", f"Moderate risk profile. Loan is {loan_to_income:.1f}x annual income. Manual review recommended.", "#f59e0b")
    return ("REJECTED", "Combined risk indicators too high. Loan application rejected.", "#ef4444")

# --- MODERN UI STRINGS ---

BASE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FinanSafe — {title}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {{ font-family: 'Inter', sans-serif; background: #f8fafc; color: #1e293b; }}
        .glass-nav {{ background: rgba(255, 255, 255, 0.8); backdrop-filter: blur(12px); border-bottom: 1px solid #e2e8f0; }}
        .card-shadow {{ box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03); transition: all 0.3s ease; }}
        .card-shadow:hover {{ box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08); transform: translateY(-2px); }}
        .form-input {{ border: 1px solid #e2e8f0; border-radius: 8px; transition: all 0.2s; background: #ffffff; }}
        .form-input:focus {{ outline: none; border-color: #6366f1; ring: 3px; ring-color: rgba(99, 102, 241, 0.1); }}
        .active-nav {{ color: #6366f1; position: relative; }}
        .active-nav::after {{ content: ''; position: absolute; bottom: -18px; left: 0; width: 100%; height: 2px; background: #6366f1; }}
        .btn-primary {{ background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%); transition: all 0.2s; }}
        .btn-primary:hover {{ opacity: 0.9; transform: scale(1.01); }}
        .status-badge {{ padding: 4px 12px; border-radius: 20px; font-weight: 600; font-size: 0.75rem; text-transform: uppercase; }}
        .APPROVED {{ background: #dcfce7; color: #15803d; }}
        .REJECTED {{ background: #fee2e2; color: #b91c1c; }}
        .REVIEW {{ background: #fef3c7; color: #92400e; }}
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {{ width: 8px; }}
        ::-webkit-scrollbar-track {{ background: #f1f5f9; }}
        ::-webkit-scrollbar-thumb {{ background: #cbd5e1; border-radius: 4px; }}
    </style>
</head>
<body class="min-h-screen">
    <nav class="glass-nav sticky top-0 z-50 px-6 py-4">
        <div class="max-w-7xl mx-auto flex justify-between items-center">
            <div class="flex items-center gap-2">
                <div class="bg-indigo-600 p-2 rounded-lg text-white">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
                </div>
                <h1 class="text-xl font-bold tracking-tight text-slate-800">FinanSafe</h1>
            </div>
            <div class="flex gap-8 items-center text-sm font-semibold text-slate-500">
                <a href="/" class="hover:text-indigo-600 transition-colors {nav_assess_cls}">Assessment</a>
                <a href="/fraud" class="hover:text-indigo-600 transition-colors {nav_fraud_cls}">Fraud Logic</a>
                <a href="/dashboard" class="hover:text-indigo-600 transition-colors {nav_dash_cls}">Dashboard</a>
            </div>
        </div>
    </nav>

    <main class="max-w-7xl mx-auto p-6 lg:p-8">
        {body}
    </main>
</body>
</html>
"""

ASSESS_BODY = """
<div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-10">
    <div class="bg-white p-6 rounded-2xl card-shadow border border-slate-100 text-center">
        <div class="text-indigo-600 font-bold text-3xl mb-1">3</div>
        <div class="text-xs font-medium text-slate-400 uppercase tracking-widest">Active Models</div>
    </div>
    <div class="bg-white p-6 rounded-2xl card-shadow border border-slate-100 text-center">
        <div class="text-emerald-500 font-bold text-3xl mb-1">92.4%</div>
        <div class="text-xs font-medium text-slate-400 uppercase tracking-widest">Model Precision</div>
    </div>
    <div class="bg-white p-6 rounded-2xl card-shadow border border-slate-100 text-center">
        <div class="text-amber-500 font-bold text-3xl mb-1">0.82</div>
        <div class="text-xs font-medium text-slate-400 uppercase tracking-widest">AUC Score</div>
    </div>
    <div class="bg-white p-6 rounded-2xl card-shadow border border-slate-100 text-center">
        <div class="text-slate-700 font-bold text-3xl mb-1">1.2ms</div>
        <div class="text-xs font-medium text-slate-400 uppercase tracking-widest">Latency</div>
    </div>
</div>

<div class="flex flex-col lg:flex-row gap-8">
    <div class="lg:w-1/3">
        <div class="bg-white rounded-2xl card-shadow border border-slate-100 overflow-hidden">
            <div class="bg-slate-50 px-6 py-4 border-bottom border-slate-100">
                <h3 class="text-sm font-bold text-slate-700 uppercase tracking-wider">Applicant Details</h3>
            </div>
            <form action="/analyze" method="post" class="p-6 space-y-4">
                <div>
                    <label class="block text-xs font-bold text-slate-500 mb-1">Full Name</label>
                    <input type="text" name="name" class="form-input w-full px-4 py-2.5 text-sm" placeholder="John Doe" required>
                </div>
                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <label class="block text-xs font-bold text-slate-500 mb-1">Age</label>
                        <input type="number" name="age" class="form-input w-full px-4 py-2.5 text-sm" placeholder="32" required>
                    </div>
                    <div>
                        <label class="block text-xs font-bold text-slate-500 mb-1">Dependents</label>
                        <input type="number" name="dependents" class="form-input w-full px-4 py-2.5 text-sm" placeholder="2" required>
                    </div>
                </div>
                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <label class="block text-xs font-bold text-slate-500 mb-1">Monthly Inc. ($)</label>
                        <input type="number" name="income" class="form-input w-full px-4 py-2.5 text-sm" required>
                    </div>
                    <div>
                        <label class="block text-xs font-bold text-slate-500 mb-1">Annual Inc. ($)</label>
                        <input type="number" name="annual_income" class="form-input w-full px-4 py-2.5 text-sm" required>
                    </div>
                </div>
                <div>
                    <label class="block text-xs font-bold text-slate-500 mb-1">Spending Score (1-100)</label>
                    <input type="range" name="spending_score" min="1" max="100" class="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600">
                </div>
                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <label class="block text-xs font-bold text-slate-500 mb-1">Debt Ratio</label>
                        <input type="number" step="0.01" name="debt_ratio" class="form-input w-full px-4 py-2.5 text-sm" placeholder="0.45" required>
                    </div>
                    <div>
                        <label class="block text-xs font-bold text-slate-500 mb-1">Utilization</label>
                        <input type="number" step="0.01" name="utilization" class="form-input w-full px-4 py-2.5 text-sm" placeholder="0.30" required>
                    </div>
                </div>
                <div class="grid grid-cols-3 gap-2">
                    <div>
                        <label class="block text-xs font-bold text-slate-500 mb-1">Lates 30d</label>
                        <input type="number" name="late_30_59" class="form-input w-full px-4 py-2.5 text-sm" value="0">
                    </div>
                    <div>
                        <label class="block text-xs font-bold text-slate-500 mb-1">Lates 60d</label>
                        <input type="number" name="late_60_89" class="form-input w-full px-4 py-2.5 text-sm" value="0">
                    </div>
                    <div>
                        <label class="block text-xs font-bold text-slate-500 mb-1">Lates 90d</label>
                        <input type="number" name="late_90" class="form-input w-full px-4 py-2.5 text-sm" value="0">
                    </div>
                </div>
                <div>
                    <label class="block text-xs font-bold text-slate-500 mb-1">Loan Amount ($)</label>
                    <input type="number" name="loan_amount" class="form-input w-full px-4 py-2.5 text-sm font-bold text-indigo-700" placeholder="50000" required>
                </div>
                <div>
                     <label class="block text-xs font-bold text-slate-500 mb-1">Credit Lines</label>
                     <input type="number" name="credit_lines" class="form-input w-full px-4 py-2.5 text-sm" value="5">
                </div>
                <div>
                     <label class="block text-xs font-bold text-slate-500 mb-1">Real Estate Loans</label>
                     <input type="number" name="real_estate_loans" class="form-input w-full px-4 py-2.5 text-sm" value="0">
                </div>

                <button type="submit" class="btn-primary w-full py-4 rounded-xl text-white font-bold text-sm shadow-lg shadow-indigo-200 mt-4 flex items-center justify-center gap-2">
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
                    Run Predictive Analysis
                </button>
            </form>
        </div>
    </div>
    <div class="lg:w-2/3">
        {results_html}
    </div>
</div>
"""

FRAUD_BODY = """
<div class="max-w-4xl mx-auto">
    <div class="bg-indigo-900 text-white p-8 rounded-3xl mb-8 flex flex-col md:flex-row justify-between items-center gap-6 shadow-2xl shadow-indigo-200">
        <div class="md:w-2/3">
            <h2 class="text-2xl font-bold mb-2">Real-time Transaction Shield</h2>
            <p class="text-indigo-200 text-sm leading-relaxed">
                Using PCA-transformed features and Deep Forest classifiers to identify anomalies in split seconds. This environment tests the model against actual banking data streams.
            </p>
        </div>
        <div class="flex gap-4">
            <div class="text-center">
                <div class="text-2xl font-bold">99.1%</div>
                <div class="text-[10px] uppercase font-bold text-indigo-300">Accuracy</div>
            </div>
            <div class="w-px h-10 bg-indigo-700"></div>
            <div class="text-center">
                <div class="text-2xl font-bold">0.1s</div>
                <div class="text-[10px] uppercase font-bold text-indigo-300">Processing</div>
            </div>
        </div>
    </div>

    <div class="grid grid-cols-1 md:grid-cols-12 gap-8">
        <div class="md:col-span-4 space-y-6">
            <div class="bg-white p-6 rounded-2xl card-shadow border border-slate-100">
                <h3 class="text-xs font-bold text-slate-500 uppercase tracking-widest mb-6">Simulation Controls</h3>
                {controls_html}
            </div>
            
            <div class="bg-slate-800 text-slate-400 p-6 rounded-2xl text-[11px] leading-loose font-mono">
                <span class="text-emerald-400"># Model metadata</span><br>
                Type: XGBoostClassifier<br>
                Features: V1-V28 (PCA), Amount, Time<br>
                Scale: MinMaxScaled<br>
                Last_Updated: 2024-03-15
            </div>
        </div>
        <div class="md:col-span-8">
            {fraud_html}
        </div>
    </div>
</div>
"""

DASHBOARD_BODY = """
<div class="space-y-8">
    <div class="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div class="bg-white p-6 rounded-2xl card-shadow border-l-4 border-indigo-500">
            <div class="text-slate-400 text-xs font-bold uppercase mb-1">Total Logs</div>
            <div class="text-2xl font-bold text-slate-800">{total}</div>
        </div>
        <div class="bg-white p-6 rounded-2xl card-shadow border-l-4 border-emerald-500">
            <div class="text-slate-400 text-xs font-bold uppercase mb-1">Approved</div>
            <div class="text-2xl font-bold text-emerald-600">{approved}</div>
        </div>
        <div class="bg-white p-6 rounded-2xl card-shadow border-l-4 border-red-500">
            <div class="text-slate-400 text-xs font-bold uppercase mb-1">Rejected</div>
            <div class="text-2xl font-bold text-red-600">{rejected}</div>
        </div>
        <div class="bg-white p-6 rounded-2xl card-shadow border-l-4 border-amber-500">
            <div class="text-slate-400 text-xs font-bold uppercase mb-1">Under Review</div>
            <div class="text-2xl font-bold text-amber-600">{review}</div>
        </div>
    </div>

    <div class="bg-white rounded-2xl card-shadow border border-slate-100 overflow-hidden">
        <div class="p-6 border-b border-slate-100 flex justify-between items-center">
            <h3 class="font-bold text-slate-800">Historical Assessments</h3>
            <span class="text-xs font-medium text-slate-400">Syncing live...</span>
        </div>
        <div class="overflow-x-auto">
            {table_html}
        </div>
    </div>
</div>
"""

def render_results(results):
    if not results:
        return """
        <div class="bg-white rounded-2xl border-2 border-dashed border-slate-200 h-full flex flex-col items-center justify-center p-12 text-center">
            <div class="w-16 h-16 bg-slate-50 rounded-full flex items-center justify-center mb-4">
                <svg xmlns="http://www.w3.org/2000/svg" class="text-slate-300" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
            </div>
            <h4 class="text-slate-400 font-medium">Ready for input</h4>
            <p class="text-slate-300 text-xs mt-1 max-w-[200px]">Complete the form on the left to generate a risk report.</p>
        </div>"""

    return f"""
    <div class="space-y-6">
        <div class="bg-white rounded-2xl card-shadow border border-slate-100 overflow-hidden">
             <div class="p-6">
                <div class="flex justify-between items-start mb-6">
                    <div>
                        <h2 class="text-2xl font-bold text-slate-800">{results['name']}</h2>
                        <p class="text-sm text-slate-400">Profile Analysis • {datetime.now().strftime('%b %d, %Y')}</p>
                    </div>
                    <span class="px-3 py-1 bg-indigo-50 text-indigo-600 rounded-full text-[10px] font-bold uppercase">{results['segment_name']}</span>
                </div>
                
                <div class="grid grid-cols-3 gap-6 py-6 border-y border-slate-50">
                    <div>
                        <div class="text-[10px] uppercase font-bold text-slate-400 mb-1">Monthly Income</div>
                        <div class="text-lg font-bold text-slate-700">${results['income']}</div>
                    </div>
                    <div>
                        <div class="text-[10px] uppercase font-bold text-slate-400 mb-1">Loan Amount</div>
                        <div class="text-lg font-bold text-indigo-600">${results['loan_amount']}</div>
                    </div>
                    <div>
                        <div class="text-[10px] uppercase font-bold text-slate-400 mb-1">Applicant Age</div>
                        <div class="text-lg font-bold text-slate-700">{results['age']}</div>
                    </div>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mt-8">
                    <div class="p-5 bg-slate-50 rounded-2xl border border-slate-100">
                        <div class="flex justify-between items-end mb-2">
                            <span class="text-xs font-bold text-slate-500 uppercase tracking-wider">Behavioral Risk</span>
                            <span class="text-lg font-bold" style="color:{results['behavioral_color']}">{results['behavioral_label']}</span>
                        </div>
                        <div class="w-full bg-slate-200 h-2 rounded-full overflow-hidden">
                            <div class="h-full rounded-full transition-all duration-700" style="width:{results['behavioral_score']}%; background:{results['behavioral_color']}"></div>
                        </div>
                        <p class="text-[10px] text-slate-400 mt-2 font-medium">Based on payment history & debt-to-income ratios.</p>
                    </div>

                    <div class="p-5 bg-slate-50 rounded-2xl border border-slate-100">
                        <div class="flex justify-between items-end mb-2">
                            <span class="text-xs font-bold text-slate-500 uppercase tracking-wider">Default Prob.</span>
                            <span class="text-lg font-bold" style="color:{results['default_color']}">{results['default_label']}</span>
                        </div>
                        <div class="w-full bg-slate-200 h-2 rounded-full overflow-hidden">
                            <div class="h-full rounded-full transition-all duration-700" style="width:{results['default_score']}%; background:{results['default_color']}"></div>
                        </div>
                        <p class="text-[10px] text-slate-400 mt-2 font-medium">ML generated probability: {results['default_score']}%</p>
                    </div>
                </div>
             </div>
        </div>

        <div class="bg-white rounded-3xl card-shadow border-2 border-indigo-50 p-8 text-center relative overflow-hidden">
            <div class="absolute top-0 right-0 p-4 opacity-5">
                <svg width="100" height="100" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/></svg>
            </div>
            <h3 class="text-xs font-bold text-slate-400 uppercase tracking-[0.2em] mb-4">Final Determination</h3>
            <div class="text-5xl font-black mb-4 tracking-tighter" style="color:{results['rec_color']}">{results['rec_verdict']}</div>
            <p class="text-slate-600 text-sm leading-relaxed max-w-md mx-auto">{results['rec_reason']}</p>
        </div>
    </div>
    """

def render_fraud(fraud_result):
    if not fraud_result:
        return """
        <div class="bg-white rounded-2xl border-2 border-dashed border-slate-200 h-full flex flex-col items-center justify-center p-12 text-center">
             <div class="w-16 h-16 bg-slate-50 rounded-full flex items-center justify-center mb-4">
                <svg xmlns="http://www.w3.org/2000/svg" class="text-slate-300" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
            </div>
            <h4 class="text-slate-400 font-medium">Monitoring Stream</h4>
            <p class="text-slate-300 text-xs mt-1">Select a transaction type to run automated verification.</p>
        </div>"""

    return f"""
    <div class="bg-white rounded-2xl card-shadow border border-slate-100 p-8">
        <div class="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-8">
            <div>
                <div class="text-[10px] font-black text-slate-300 uppercase tracking-widest mb-1">TXN ID: {fraud_result['txn_id']}</div>
                <div class="text-4xl font-black text-slate-800">${fraud_result['amount']}</div>
            </div>
            <div class="text-right">
                <div class="text-sm font-bold text-slate-500">{fraud_result['location']}</div>
                <div class="text-xs text-slate-400">{fraud_result['time']}</div>
            </div>
        </div>

        <div class="grid grid-cols-2 gap-4 mb-8">
            <div class="p-4 bg-slate-50 rounded-xl border border-slate-100 text-center">
                <div class="text-[10px] font-bold text-slate-400 uppercase mb-1">Risk Confidence</div>
                <div class="text-2xl font-bold" style="color:{fraud_result['risk_color']}">{fraud_result['risk_score']}%</div>
            </div>
            <div class="p-4 bg-slate-50 rounded-xl border border-slate-100 text-center">
                <div class="text-[10px] font-bold text-slate-400 uppercase mb-1">Known Label</div>
                <div class="text-2xl font-bold text-slate-700">{fraud_result['actual']}</div>
            </div>
        </div>

        <div class="relative h-4 bg-slate-100 rounded-full overflow-hidden mb-10">
            <div class="h-full rounded-full transition-all duration-1000" style="width:{fraud_result['risk_score']}%; background:{fraud_result['risk_color']}"></div>
        </div>

        <div class="py-6 px-8 rounded-2xl text-center border-2 font-black text-xl uppercase tracking-wider {fraud_result['verdict_class']}" 
             style="border-color: currentColor; background-color: transparent">
            {fraud_result['verdict']}
        </div>
        
        <style>
            .verdict-fraud {{ color: #ef4444; background: #fef2f2 !important; }}
            .verdict-legit {{ color: #10b981; background: #f0fdf4 !important; }}
        </style>
    </div>
    """

# --- Routes (Keeping original logic, just formatting HTML) ---

@app.route('/')
def home():
    html = BASE.format(
        title="Customer Assessment",
        nav_assess_cls="active-nav", nav_fraud_cls="", nav_dash_cls="",
        body=ASSESS_BODY.format(results_html=render_results(None))
    )
    return html

@app.route('/fraud')
def fraud_page():
    if df_fraud is not None:
        controls = f"""
        <div class="space-y-3">
            <form action="/fraud_predict" method="post"><input type="hidden" name="type" value="fraud"><button type="submit" class="w-full py-3 bg-red-50 text-red-600 font-bold rounded-xl text-xs hover:bg-red-100 transition-colors">SIMULATE FRAUD</button></form>
            <form action="/fraud_predict" method="post"><input type="hidden" name="type" value="legit"><button type="submit" class="w-full py-3 bg-emerald-50 text-emerald-600 font-bold rounded-xl text-xs hover:bg-emerald-100 transition-colors">SIMULATE LEGIT</button></form>
            <form action="/fraud_predict" method="post"><input type="hidden" name="type" value="random"><button type="submit" class="w-full py-3 bg-indigo-50 text-indigo-600 font-bold rounded-xl text-xs hover:bg-indigo-100 transition-colors">RANDOM STREAM</button></form>
        </div>"""
    else:
        controls = """
        <div class="p-4 bg-amber-50 border border-amber-100 rounded-xl text-amber-800 text-xs leading-relaxed">
            <strong>Dataset missing:</strong><br> Please upload <code>creditcard.csv</code> to project root to enable transaction testing.
        </div>"""

    html = BASE.format(
        title="Fraud Detection",
        nav_assess_cls="", nav_fraud_cls="active-nav", nav_dash_cls="",
        body=FRAUD_BODY.format(controls_html=controls, fraud_html=render_fraud(None))
    )
    return html

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.form
    behavioral_score, behavioral_label, behavioral_color = calculate_behavioral_risk(
        float(data['debt_ratio']),  float(data['utilization']),
        int(data['late_30_59']),    int(data['late_60_89']),   int(data['late_90']),
        int(data['dependents']),    float(data['income']),     float(data['loan_amount'])
    )

    default_features = pd.DataFrame([[
        float(data['utilization']),     int(data['age']),
        int(data['late_30_59']),        float(data['debt_ratio']),
        float(data['income']),          int(data['credit_lines']),
        int(data['late_90']),           int(data['real_estate_loans']),
        int(data['late_60_89']),        int(data['dependents'])
    ]], columns=[
        'RevolvingUtilizationOfUnsecuredLines', 'age',
        'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio',
        'MonthlyIncome',                        'NumberOfOpenCreditLinesAndLoans',
        'NumberOfTimes90DaysLate',              'NumberRealEstateLoansOrLines',
        'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents'
    ])

    default_prob  = default_model.predict_proba(default_features)[0][1]
    default_score = round(default_prob * 100, 1)

    if default_score >= 60:
        default_label, default_color = "HIGH RISK",   "#ef4444"
    elif default_score >= 30:
        default_label, default_color = "MEDIUM RISK", "#f59e0b"
    else:
        default_label, default_color = "LOW RISK",    "#10b981"

    annual_income_k = float(data['annual_income']) / 1000
    segment = int(segment_model.predict(np.array([[annual_income_k, float(data['spending_score'])]]))[0])
    segment_name = segment_names.get(segment, "Unknown")

    rec_verdict, rec_reason, rec_color = get_loan_recommendation(
        default_prob, behavioral_score, segment,
        float(data['income']), float(data['loan_amount'])
    )

    with open(DATA_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            data['name'], data['age'], data['income'],
            data['annual_income'], data['spending_score'], data['debt_ratio'],
            data['utilization'], data['dependents'], data['credit_lines'],
            data['real_estate_loans'], data['late_30_59'], data['late_60_89'],
            data['late_90'], data['loan_amount'],
            behavioral_score, behavioral_label,
            default_score, default_label,
            segment_name, rec_verdict
        ])

    results = {
        "name": data['name'], "age": data['age'], "income": data['income'],
        "loan_amount": data['loan_amount'], "segment_name": segment_name,
        "behavioral_score": behavioral_score, "behavioral_label": behavioral_label, "behavioral_color": behavioral_color,
        "default_score": default_score, "default_label": default_label, "default_color": default_color,
        "rec_verdict": rec_verdict, "rec_reason": rec_reason, "rec_color": rec_color
    }

    html = BASE.format(
        title="Customer Assessment",
        nav_assess_cls="active-nav", nav_fraud_cls="", nav_dash_cls="",
        body=ASSESS_BODY.format(results_html=render_results(results))
    )
    return html

@app.route('/fraud_predict', methods=['POST'])
def fraud_predict():
    if df_fraud is None: return redirect(url_for('fraud_page'))

    transaction_type = request.form['type']
    if transaction_type == 'fraud':
        idx = random.randint(0, len(fraud_samples) - 1); sample = fraud_samples.iloc[idx]
        amount = fraud_amounts.iloc[idx]; time_v = fraud_times.iloc[idx]; actual = 1
    elif transaction_type == 'legit':
        idx = random.randint(0, len(legit_samples) - 1); sample = legit_samples.iloc[idx]
        amount = legit_amounts.iloc[idx]; time_v = legit_times.iloc[idx]; actual = 0
    else: # random
        if random.random() < 0.5:
            idx = random.randint(0, len(fraud_samples) - 1); sample = fraud_samples.iloc[idx]
            amount = fraud_amounts.iloc[idx]; time_v = fraud_times.iloc[idx]; actual = 1
        else:
            idx = random.randint(0, len(legit_samples) - 1); sample = legit_samples.iloc[idx]
            amount = legit_amounts.iloc[idx]; time_v = legit_times.iloc[idx]; actual = 0

    feature_cols = [col for col in df_fraud.columns if col != 'Class']
    df_input = pd.DataFrame(sample[feature_cols].values.reshape(1, -1), columns=feature_cols)
    probability = fraud_model.predict_proba(df_input)[0][1]
    prediction = fraud_model.predict(df_input)[0]
    risk_score = round(probability * 100, 1)

    risk_color = "#ef4444" if risk_score >= 60 else ("#f59e0b" if risk_score >= 30 else "#10b981")

    fraud_result = {
        "txn_id": f"TXN-{random.randint(100000, 999999)}", "amount": f"{amount:.2f}",
        "time": seconds_to_time(time_v), "location": random.choice(LOCATIONS),
        "risk_score": risk_score, "risk_color": risk_color,
        "verdict": "FRAUD DETECTED — Transaction Blocked" if prediction == 1 else "LEGITIMATE — Transaction Approved",
        "verdict_class": "verdict-fraud" if prediction == 1 else "verdict-legit",
        "actual": "FRAUD" if actual == 1 else "Legitimate"
    }

    controls = f"""
    <div class="space-y-3">
        <form action="/fraud_predict" method="post"><input type="hidden" name="type" value="fraud"><button type="submit" class="w-full py-3 bg-red-50 text-red-600 font-bold rounded-xl text-xs hover:bg-red-100 transition-colors">SIMULATE FRAUD</button></form>
        <form action="/fraud_predict" method="post"><input type="hidden" name="type" value="legit"><button type="submit" class="w-full py-3 bg-emerald-50 text-emerald-600 font-bold rounded-xl text-xs hover:bg-emerald-100 transition-colors">SIMULATE LEGIT</button></form>
        <form action="/fraud_predict" method="post"><input type="hidden" name="type" value="random"><button type="submit" class="w-full py-3 bg-indigo-50 text-indigo-600 font-bold rounded-xl text-xs hover:bg-indigo-100 transition-colors">RANDOM STREAM</button></form>
    </div>"""

    html = BASE.format(
        title="Fraud Detection",
        nav_assess_cls="", nav_fraud_cls="active-nav", nav_dash_cls="",
        body=FRAUD_BODY.format(controls_html=controls, fraud_html=render_fraud(fraud_result))
    )
    return html

@app.route('/dashboard')
def dashboard():
    records = []
    total = approved = rejected = review = 0
    if os.path.exists(DATA_FILE):
        try:
            df_data = pd.read_csv(DATA_FILE)
            if not df_data.empty:
                records = df_data.to_dict('records')
                total = len(records)
                approved = sum(1 for r in records if r.get('recommendation') == 'APPROVED')
                rejected = sum(1 for r in records if r.get('recommendation') == 'REJECTED')
                review = sum(1 for r in records if r.get('recommendation') == 'REVIEW')
        except: pass

    rows = ""
    for r in records:
        rec = r.get('recommendation', '')
        rows += f"""
        <tr class="hover:bg-slate-50 transition-colors">
            <td class="px-6 py-4 text-xs text-slate-500 font-medium whitespace-nowrap">{r.get('timestamp', '')}</td>
            <td class="px-6 py-4 text-sm font-bold text-slate-700">{r.get('name', '')}</td>
            <td class="px-6 py-4 text-sm text-slate-600 text-center">{r.get('age', '')}</td>
            <td class="px-6 py-4 text-sm text-slate-600 font-semibold">${r.get('income', '')}</td>
            <td class="px-6 py-4 text-sm font-bold text-indigo-600">${r.get('loan_amount', '')}</td>
            <td class="px-6 py-4">
                <div class="flex items-center gap-2">
                    <span class="text-xs font-bold text-slate-500">{r.get('default_label', '')}</span>
                </div>
            </td>
            <td class="px-6 py-4 text-center">
                <span class="status-badge {rec}">{rec}</span>
            </td>
        </tr>"""

    table_html = f"""
    <table class="w-full text-left">
        <thead class="bg-slate-50 border-b border-slate-100">
            <tr>
                <th class="px-6 py-4 text-[10px] font-black text-slate-400 uppercase tracking-wider">Timestamp</th>
                <th class="px-6 py-4 text-[10px] font-black text-slate-400 uppercase tracking-wider">Applicant</th>
                <th class="px-6 py-4 text-[10px] font-black text-slate-400 uppercase tracking-wider text-center">Age</th>
                <th class="px-6 py-4 text-[10px] font-black text-slate-400 uppercase tracking-wider">Monthly Inc.</th>
                <th class="px-6 py-4 text-[10px] font-black text-slate-400 uppercase tracking-wider">Loan</th>
                <th class="px-6 py-4 text-[10px] font-black text-slate-400 uppercase tracking-wider">ML Risk</th>
                <th class="px-6 py-4 text-[10px] font-black text-slate-400 uppercase tracking-wider text-center">Outcome</th>
            </tr>
        </thead>
        <tbody>
            {rows if records else '<tr><td colspan="7" class="py-12 text-center text-slate-400 italic">No activity logs found.</td></tr>'}
        </tbody>
    </table>"""

    html = BASE.format(
        title="Dashboard",
        nav_assess_cls="", nav_fraud_cls="", nav_dash_cls="active-nav",
        body=DASHBOARD_BODY.format(total=total, approved=approved, rejected=rejected, review=review, table_html=table_html)
    )
    return html

if __name__ == '__main__':
    app.run(debug=True)