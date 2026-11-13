import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import brentq
import json
import io
import sqlite3
from datetime import datetime

# ═══════════════════════════════════════════════════════════════
#  DATABASE SETUP (SQLite)
# ═══════════════════════════════════════════════════════════════
DB_FILE = "lms_data.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS borrowers (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 name TEXT, abn TEXT, group_name TEXT, 
                 risk_grade TEXT, industry TEXT, status TEXT)''')
                 
    c.execute('''CREATE TABLE IF NOT EXISTS deals (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 borrower_id INTEGER, deal_name TEXT,
                 facility_amount REAL, deal_status TEXT,
                 settlement_date TEXT, maturity_date TEXT,
                 facility_type TEXT, security_type TEXT,
                 property_value REAL, params_json TEXT,
                 FOREIGN KEY(borrower_id) REFERENCES borrowers(id))''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS actuals (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 deal_id INTEGER, month INTEGER,
                 actual_drawdown REAL, actual_interest REAL,
                 actual_principal REAL, actual_fees REAL,
                 FOREIGN KEY(deal_id) REFERENCES deals(id))''')
    conn.commit()
    conn.close()

init_db()

def run_query(query, params=()):
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

def execute_query(query, params=()):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(query, params)
    conn.commit()
    conn.close()

# ═══════════════════════════════════════════════════════════════
#  FINANCIAL MATH ENGINE
# ═══════════════════════════════════════════════════════════════
def to_monthly(r): return (1 + r) ** (1/12) - 1
def to_annual(r): return (1 + r) ** 12 - 1
def cpr_to_smm(cpr): return 1 - (1 - cpr) ** (1/12) if cpr > 0 else 0.0

def calc_irr(cfs):
    if len(cfs) < 2 or all(c == 0 for c in cfs[1:]): return np.nan
    try:
        def npv(r): return sum(cf / (1 + r)**t for t, cf in enumerate(cfs))
        return to_annual(brentq(npv, -0.5, 5.0, xtol=1e-10))
    except: return np.nan

def compute_fees(facility, outstanding, month, fee_config,
                 is_default=False, is_restructure=False, restructure_amount=0, is_exit=False, construction_end=0):
    fees = {}
    if month == 0:
        fees["Establishment"] = facility * fee_config["establishment_pct"]
    
    undrawn_balance = max(facility - outstanding, 0)
    if 0 < month <= construction_end:
        fees["Line Fee (Undrawn)"] = undrawn_balance * fee_config["line_fee_pct"] / 12
        
    if is_default: fees["Risk Review"] = fee_config["risk_review_amount"]
    if is_restructure: fees["Restructure"] = restructure_amount * fee_config["restructure_pct"]
    
    if month > 0:
        if fee_config["admin_type"] == "fixed":
            fees["Admin"] = fee_config["admin_amount"]
        else:
            fees["Admin"] = outstanding * fee_config["admin_pct"] / 12
            
    if is_exit: fees["Discharge"] = fee_config["discharge_amount"]
    return fees

def build_schedule(facility, target_irr, n_months,
                   interest_timing="arrears", advance_n=1, capitalise_months=0,
                   bbsy_curve=None, margin=0.0, reset_freq=12, drawdown_events=None, cpr=0.0,
                   fee_config=None, default_month=None, restructure_month=None,
                   restructure_amount=0, construction_end=0, property_value=0, property_growth=0,
                   actuals_dict=None, penalty_rate_annual=0.0, max_lvr_covenant=100.0):
    
    r_m = to_monthly(target_irr)
    smm = cpr_to_smm(cpr)
    if drawdown_events is None: drawdown_events = [(0, facility)]
    if fee_config is None: fee_config = {"establishment_pct":0, "line_fee_pct":0, "admin_type":"fixed", "admin_amount":0, "discharge_amount":0, "risk_review_amount":0, "restructure_pct":0}

    total_commitment = sum(a for _, a in drawdown_events)
    dd_idx = 0
    balance = 0.0
    cum_principal = 0.0
    arrears_balance = 0.0

    def get_rate(m):
        if bbsy_curve and len(bbsy_curve) > 0:
            p = min(m // reset_freq, len(bbsy_curve) - 1)
            return to_monthly(bbsy_curve[p] + margin)
        return r_m

    rows = []

    for m in range(n_months + 1):
        is_actual = (actuals_dict is not None and m in actuals_dict)
        act = actuals_dict.get(m, {}) if is_actual else {}

        # Drawdowns
        if is_actual and "drawdown" in act:
            dd_amt = act["drawdown"]
        else:
            dd_amt = 0.0
            while dd_idx < len(drawdown_events) and drawdown_events[dd_idx][0] == m:
                dd_amt += drawdown_events[dd_idx][1]
                dd_idx += 1
        balance += dd_amt

        is_def = default_month is not None and m == default_month
        is_res = restructure_month is not None and m == restructure_month
        is_exit = m == n_months or (balance <= 0 and m > 0)

        # Fees
        month_fees = compute_fees(facility, balance, m, fee_config, is_def, is_res, restructure_amount, is_exit, construction_end)
        
        if arrears_balance > 0:
            penalty_int = arrears_balance * (penalty_rate_annual / 12)
            month_fees["Penalty Interest"] = penalty_int
            
        if is_actual and "fees" in act:
            total_fees = act["fees"]
        else:
            total_fees = sum(month_fees.values())

        prop_val = property_value * (1 + property_growth/12) ** m if property_value > 0 else 0
        lvr = balance / prop_val * 100 if prop_val > 0 else 0
        current_limit = max(facility - cum_principal, 0)
        covenant_breach = lvr > max_lvr_covenant

        if m == 0:
            net_disb = dd_amt - total_fees
            rows.append({
                "Month": m, "Mode": "Actual" if is_actual else "Forecast", "Drawdown": dd_amt, "Opening": balance, "Rate": get_rate(m),
                "Interest": 0.0, "Interest Paid": 0.0, "Interest Capitalised": 0.0, "Principal": 0.0, "Prepayment": 0.0,
                "Fees": total_fees, "Fee Detail": month_fees, "Total Payment": -net_disb, "Closing": balance, "Arrears": 0.0,
                "Current Limit": current_limit, "LVR": lvr, "Covenant Breach": covenant_breach, "Borrower CF": net_disb, "Anniversary": False
            })
            continue

        rate_m = get_rate(m - 1)
        interest = balance * rate_m
        is_cap = m <= capitalise_months
        cap_amt = interest if is_cap else 0.0
        expected_cash_int = 0.0 if is_cap else interest

        if is_cap: balance += cap_amt

        remaining = n_months - m + 1
        expected_princ = (total_commitment - cum_principal) / max(remaining, 1) if (remaining > 0 and m > capitalise_months) else 0.0
        expected_prepay = max(balance - expected_princ, 0) * smm if (smm > 0 and m > capitalise_months) else 0.0

        if is_actual:
            cash_int = act.get("interest", 0.0)
            total_princ = act.get("principal", 0.0)
            prepay = 0.0
            
            expected_total = expected_cash_int + expected_princ + expected_prepay + sum(month_fees.values())
            
