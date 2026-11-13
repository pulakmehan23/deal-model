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
#  FINANCIAL MATH ENGINE (100% v4 Engine)
# ═══════════════════════════════════════════════════════════════
def to_monthly(r):
    return (1 + r) ** (1/12) - 1

def to_annual(r):
    return (1 + r) ** 12 - 1

def calc_irr(cfs):
    if len(cfs) < 2 or all(c == 0 for c in cfs[1:]):
        return np.nan
    try:
        def npv(r):
            return sum(cf / (1 + r)**t for t, cf in enumerate(cfs))
        return to_annual(brentq(npv, -0.5, 5.0, xtol=1e-10))
    except:
        return np.nan

def cpr_to_smm(cpr):
    return 1 - (1 - cpr) ** (1/12) if cpr > 0 else 0.0

def compute_fees(facility, outstanding, month, fee_config,
                 is_default=False, is_restructure=False,
                 restructure_amount=0, is_exit=False,
                 construction_end=0):
    fees = {}
    if month == 0:
        fees["Establishment"] = facility * fee_config["establishment_pct"]
    if 0 < month <= construction_end:
        fees["Line Fee"] = facility * fee_config["line_fee_pct"] / 12
    if is_default:
        fees["Risk Review"] = fee_config["risk_review_amount"]
    if is_restructure:
        fees["Restructure"] = restructure_amount * fee_config["restructure_pct"]
    if month > 0:
        if fee_config["admin_type"] == "fixed":
            fees["Admin"] = fee_config["admin_amount"]
        else:
            fees["Admin"] = outstanding * fee_config["admin_pct"] / 12
    if is_exit:
        fees["Discharge"] = fee_config["discharge_amount"]
    return fees

def build_schedule(facility, target_irr, n_months,
                   interest_timing="arrears", advance_n=1,
                   capitalise_months=0,
                   bbsy_curve=None, margin=0.0, reset_freq=12,
                   drawdown_events=None, cpr=0.0,
                   fee_config=None,
                   default_month=None, restructure_month=None,
                   restructure_amount=0, construction_end=0,
                   property_value=0, property_growth=0,
                   actuals_dict=None):
    
    r_m = to_monthly(target_irr)
    smm = cpr_to_smm(cpr)
    if drawdown_events is None:
        drawdown_events = [(0, facility)]
    if fee_config is None:
        fee_config = {k: 0 for k in [
            "establishment_pct", "line_fee_pct", "risk_review_amount",
            "restructure_pct", "admin_amount", "admin_pct",
            "admin_type", "discharge_amount"]}

    total_commitment = sum(a for _, a in drawdown_events)
    dd_idx = 0
    balance = 0.0
    cum_principal = 0.0

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
        if is_actual and "fees" in act:
            total_fees = act["fees"]
            month_fees = {"Actual Fee": total_fees}
        else:
            month_fees = compute_fees(
                facility, balance, m, fee_config,
                is_def, is_res, restructure_amount, is_exit, construction_end)
            total_fees = sum(month_fees.values())

        prop_val = property_value * (1 + property_growth/12) ** m if property_value > 0 else 0
        lvr = balance / prop_val * 100 if prop_val > 0 else 0
        current_limit = max(facility - cum_principal, 0)

        if m == 0:
            net_disb = dd_amt - total_fees
            rows.append({
                "Month": m, "Mode": "Actual" if is_actual else "Forecast",
                "Drawdown": dd_amt, "Opening": balance, "Rate": get_rate(m),
                "Interest": 0.0, "Interest Paid": 0.0, "Interest Capitalised": 0.0,
                "Principal": 0.0, "Prepayment": 0.0, "Fees": total_fees,
                "Fee Detail": month_fees, "Total Payment": -net_disb,
                "Closing": balance, "Original Limit": facility,
                "Current Limit": current_limit, "Property Value": prop_val,
                "LVR": lvr, "Borrower CF": net_disb, "Anniversary": False
            })
            continue

        rate_m = get_rate(m - 1)
        interest = balance * rate_m
        is_cap = m <= capitalise_months
        cap_amt = interest if is_cap else 0.0
        cash_int = 0.0 if is_cap else interest

        if is_cap:
            balance += cap_amt

        # Principal & Prepayment
        if is_actual and "principal" in act:
            sched_princ = act["principal"]
            prepay = 0.0
        else:
            remaining = n_months - m + 1
            if remaining > 0 and m > capitalise_months:
                sched_princ = (total_commitment - cum_principal) / max(remaining, 1)
            else:
                sched_princ = 0.0

            prepay = 0.0
            if smm > 0 and m > capitalise_months:
                prepay = max(balance - sched_princ, 0) * smm
                prepay = min(prepay, balance - sched_princ)

        total_princ = min(sched_princ + prepay, balance)
        balance -= total_princ
        cum_principal += total_princ

        if is_actual and "interest" in act:
            cash_int = act["interest"]

        total_payment = cash_int + total_princ + total_fees
        current_limit = max(facility - cum_principal, 0)

        rows.append({
            "Month": m, "Mode": "Actual" if is_actual else "Forecast",
            "Drawdown": dd_amt, "Opening": balance + total_princ, "Rate": rate_m,
            "Interest": interest, "Interest Paid": cash_int,
            "Interest Capitalised": cap_amt, "Principal": sched_princ,
            "Prepayment": prepay, "Fees": total_fees, "Fee Detail": month_fees,
            "Total Payment": total_payment, "Closing": max(balance, 0),
            "Original Limit": facility, "Current Limit": current_limit,
            "Property Value": prop_val,
            "LVR": max(balance, 0) / prop_val * 100 if prop_val > 0 else 0,
            "Borrower CF": -total_payment, "Anniversary": m % 12 == 0
        })

    df = pd.DataFrame(rows)

    if interest_timing == "all_advance":
        total_int = df["Interest"].sum()
        df.loc[0, "Borrower CF"] -= total_int
        df.loc[0, "Total Payment"] = -df.loc[0, "Borrower CF"]
        df.loc[0, "Interest Paid"] = total_int
        for i in range(1, len(df)):
            if df.loc[i, "Mode"] == "Forecast":
                df.loc[i, "Interest Paid"] = 0.0
                df.loc[i, "Borrower CF"] = -(df.loc[i, "Principal"] + df.loc[i, "Prepayment"] + df.loc[i, "Fees"])
                df.loc[i, "Total Payment"] = -df.loc[i, "Borrower CF"]

    return df

# ═══════════════════════════════════════════════════════════════
#  UI SETUP & NAVIGATION
# ═══════════════════════════════════════════════════════════════
st.set_page_config(page_title="Structured Lending Platform", layout="wide", page_icon="🏦")
st.sidebar.title("🏦 Admin Platform")
nav_page = st.sidebar.radio("Navigation", ["📊 Dashboard", "🏢 Borrowers", "📄 Deals Directory", "🧮 Deal Modeler & LMS"])

# ═══════════════════════════════════════════════════════════════
#  PAGE 1: DASHBOARD
# ═══════════════════════════════════════════════════════════════
if nav_page == "📊 Dashboard":
    st.title("📊 Portfolio Dashboard")
    deals_df = run_query("SELECT * FROM deals")
    borrowers_df = run_query("SELECT * FROM borrowers")
    
    if deals_df.empty:
        st.info("👋 Welcome! System is ready. Go to 'Borrowers' to add your first entity.")
    else:
        active_deals = deals_df[deals_df['deal_status'] == 'Active']
        total_aum = active_deals['facility_amount'].sum()
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Active AUM", f"${total_aum:,.0f}" if total_aum else "$0")
        c2.metric("Active Deals", len(active_deals))
        c3.metric("Pipeline Deals", len(deals_df[deals_df['deal_status'] == 'Pipeline']))
        c4.metric("Total Borrowers", len(borrowers_df))
        
        st.markdown("---")
        st.markdown("### Portfolio Deals")
        query = '''
        SELECT d.id, b.name as 'Borrower', d.deal_name as 'Deal', 
               d.facility_amount as 'Facility ($)', d.deal_status as 'Status',
               d.facility_type as 'Type', d.security_type as 'Security'
        FROM deals d
        JOIN borrowers b ON d.borrower_id = b.id
        '''
        df_disp = run_query(query)
        df_disp['Facility ($)'] = df_disp['Facility ($)'].apply(lambda x: f"${x:,.0f}")
        st.dataframe(df_disp, hide_index=True, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
#  PAGE 2: BORROWERS
# ═══════════════════════════════════════════════════════════════
elif nav_page == "🏢 Borrowers":
    st.title("🏢 Borrower Management")
    with st.expander("➕ Add New Borrower"):
        with st.form("new_borrower_form"):
            c1, c2 = st.columns(2)
            b_name = c1.text_input("Borrower Entity Name")
            b_abn = c2.text_input("ABN / ACN")
            b_group = c1.text_input("Group / Sponsor Name")
            b_ind = c2.selectbox("Industry / Sector", ["Residential Dev", "Commercial CRE", "Corporate", "Other"])
            b_risk = c1.selectbox("Risk Grade", ["A", "B", "C", "D", "Watchlist"])
            
            if st.form_submit_button("Save Borrower") and b_name:
                execute_query("INSERT INTO borrowers (name, abn, group_name, risk_grade, industry, status) VALUES (?,?,?,?,?,?)",
                              (b_name, b_abn, b_group, b_risk, b_ind, "Active"))
                st.success(f"Added {b_name}!")
                st.rerun()

    st.markdown("### Borrower Directory")
    b_df = run_query("SELECT id, name as 'Borrower Entity', abn as 'ABN', group_name as 'Group', risk_grade as 'Risk Grade', industry as 'Industry' FROM borrowers")
    st.dataframe(b_df, hide_index=True, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
#  PAGE 3: DEALS DIRECTORY
# ═══════════════════════════════════════════════════════════════
elif nav_page == "📄 Deals Directory":
    st.title("📄 Deals Directory")
    borrowers_df = run_query("SELECT id, name FROM borrowers")
    
    if borrowers_df.empty:
        st.warning("⚠️ Please create a Borrower first.")
    else:
        with st.expander("➕ Add New Deal"):
            with st.form("new_deal_form"):
                b_dict = dict(zip(borrowers_df['name'], borrowers_df['id']))
                c1, c2, c3 = st.columns(3)
                d_borrower = c1.selectbox("Borrower Entity", options=b_dict.keys())
                d_name = c2.text_input("Deal / Project Name")
                d_status = c3.selectbox("Status", ["Pipeline", "Approved", "Active", "Matured", "Defaulted"])
                
                c4, c5, c6 = st.columns(3)
                d_fac = c4.number_input("Facility Amount ($)", min_value=100000, value=10000000, step=500000)
                d_type = c5.selectbox("Facility Type", ["Construction", "Bridging", "Term Loan", "Land"])
                d_sec = c6.selectbox("Security", ["1st Mortgage", "2nd Mortgage", "Unsecured"])
                
                c7, c8, c9 = st.columns(3)
                d_set = c7.date_input("Settlement Date")
                d_mat = c8.date_input("Maturity Date")
                d_prop = c9.number_input("Property Value ($)", min_value=0, value=15000000, step=500000)
                
                if st.form_submit_button("Create Deal") and d_name:
                    b_id = b_dict[d_borrower]
                    execute_query('''INSERT INTO deals 
                                     (borrower_id, deal_name, facility_amount, deal_status, 
                                      settlement_date, maturity_date, facility_type, security_type, property_value) 
                                     VALUES (?,?,?,?,?,?,?,?,?)''',
                                  (b_id, d_name, d_fac, d_status, str(d_set), str(d_mat), d_type, d_sec, d_prop))
                    st.success(f"Deal '{d_name}' created!")
                    st.rerun()

        query = '''
        SELECT d.id, b.name as 'Borrower', d.deal_name as 'Deal', 
               d.facility_amount as 'Facility ($)', d.deal_status as 'Status',
               d.facility_type as 'Type', d.security_type as 'Security'
        FROM deals d
        JOIN borrowers b ON d.borrower_id = b.id
        ORDER BY d.id DESC
        '''
        d_list = run_query(query)
        if not d_list.empty:
            d_list['Facility ($)'] = d_list['Facility ($)'].apply(lambda x: f"${x:,.0f}")
            st.dataframe(d_list, hide_index=True, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
#  PAGE 4: DEAL MODELER & LMS (THE FULL v4 ENGINE + ACTUALS)
# ═══════════════════════════════════════════════════════════════
elif nav_page == "🧮 Deal Modeler & LMS":
    st.title("🧮 Deal Modeler & LMS")
    
    # Select Source: Sandbox or Database Deal
    db_deals = run_query("SELECT d.id, d.deal_name, b.name FROM deals d JOIN borrowers b ON d.borrower_id = b.id")
    deal_options = ["🛠️ Sandbox / Scratchpad"] + [f"ID {r['id']}: {r['deal_name']} ({r['name']})" for _, r in db_deals.iterrows()]
    
    selected_deal = st.sidebar.selectbox("Select Active Model / Deal", deal_options)
    
    selected_deal_id = None
    if selected_deal != "🛠️ Sandbox / Scratchpad":
        selected_deal_id = int(selected_deal.split(":")[0].replace("ID ", ""))
        st.info(f"📁 Loaded Saved Deal: **{selected_deal}**")

    # Load Actuals if DB Deal
    actuals_dict = {}
    if selected_deal_id:
        act_df = run_query("SELECT * FROM actuals WHERE deal_id = ?", (selected_deal_id,))
        for _, r in act_df.iterrows():
            actuals_dict[int(r['month'])] = {
                "drawdown": r['actual_drawdown'],
                "interest": r['actual_interest'],
                "principal": r['actual_principal'],
                "fees": r['actual_fees']
            }

    # Sidebar Controls (Same as v4)
    with st.sidebar.expander("📋 Loan Structure", expanded=True):
        facility = st.number_input("Facility / Original Limit ($)", 100_000, 500_000_000, 10_000_000, step=1_000_000)
        target_irr = st.slider("Target IRR (%)", 1.0, 35.0, 12.0, 0.25) / 100
        tenure = st.slider("Tenure (months)", 3, 120, 36)
        construction_end = st.slider("Construction End (month)", 0, tenure, 0)

    with st.sidebar.expander("🏠 Collateral / LVR", expanded=True):
        property_value = st.number_input("Initial Property Value ($)", 0, 1_000_000_000, 15_000_000, step=500_000)
        property_growth = st.slider("Property Growth (% p.a.)", -10.0, 20.0, 3.0, 0.5) / 100
        initial_lvr = facility / property_value * 100 if property_value > 0 else 0
        st.metric("Initial LVR", f"{initial_lvr:.1f}%")

    with st.sidebar.expander("🕐 Interest Timing", expanded=True):
        interest_timing = st.selectbox("Timing", ["arrears", "advance", "all_advance"],
            format_func=lambda x: {"arrears": "In Arrears", "advance": "In Advance (N-month)", "all_advance": "All In Advance"}[x])
        advance_n = 1
        if interest_timing == "advance":
            advance_n = st.selectbox("Advance Period", [1, 2, 3, 4, 5, 6, 12], index=2)
        capitalise_months = st.slider("Capitalisation (months)", 0, tenure, 0)

    with st.sidebar.expander("📈 BBSY Floating Rate", expanded=False):
        use_floating = st.checkbox("Use BBSY", value=False)
        bbsy_curve = None
        margin = 0.0
        reset_freq = 12
        if use_floating:
            n_resets = max(1, tenure // 12 + 1)
            bbsy_curve = [st.number_input(f"Period {i+1} BBSY (%)", 0.0, 15.0, 4.35, 0.05, key=f"b_{i}") / 100 for i in range(n_resets)]
            margin = st.slider("Margin over BBSY (%)", 0.0, 15.0, 5.0, 0.25) / 100
            reset_freq = st.selectbox("Reset Frequency", [3, 6, 12], index=2)

    with st.sidebar.expander("💰 Drawdown Schedule", expanded=False):
        n_dd = st.number_input("Number of Drawdowns", 1, 10, 1)
        dd_events = []
        if n_dd == 1:
            dd_events = [(0, facility)]
        else:
            rem = facility
            for i in range(int(n_dd)):
                c1, c2 = st.columns(2)
                dd_m = c1.number_input(f"DD {i+1} Month", 0, tenure-1, i * (tenure // int(n_dd)), key=f"ddm_{i}")
                if i < int(n_dd) - 1:
                    dd_a = c2.number_input(f"DD {i+1} ($)", 0, int(rem), int(rem // (int(n_dd) - i)), step=100_000, key=f"dda_{i}")
                else:
                    dd_a = rem
                    c2.write(f"**${dd_a:,.0f}**")
                dd_events.append((dd_m, dd_a))
                rem -= dd_a

    with st.sidebar.expander("💵 Fee Structure", expanded=False):
        est_pct = st.slider("Establishment (%)", 0.0, 5.0, 1.5, 0.25) / 100
        line_pct = st.slider("Line Fee (% p.a.)", 0.0, 5.0, 1.0, 0.25) / 100
        admin_type = st.radio("Admin Fee Type", ["fixed", "% of outstanding"], horizontal=True)
        admin_amount = st.number_input("Admin ($/m)", 0, 50_000, 1_000, step=100) if admin_type == "fixed" else 0
        admin_pct = st.slider("Admin (% p.a.)", 0.0, 3.0, 0.5, 0.1) / 100 if admin_type != "fixed" else 0
        discharge_amt = st.number_input("Discharge ($)", 0, 100_000, 2_000, step=500)
        risk_review_amt = st.number_input("Risk Review ($)", 0, 100_000, 5_000, step=500)
        restructure_pct = st.slider("Restructure (%)", 0.0, 5.0, 1.0, 0.25) / 100

        fee_config = {
            "establishment_pct": est_pct, "line_fee_pct": line_pct,
            "admin_type": admin_type, "admin_amount": admin_amount,
            "admin_pct": admin_pct, "discharge_amount": discharge_amt,
            "risk_review_amount": risk_review_amt, "restructure_pct": restructure_pct
        }

    with st.sidebar.expander("⚠️ Risk & Events", expanded=False):
        cpr = st.slider("CPR (%)", 0.0, 50.0, 0.0, 1.0) / 100
        pd_annual = st.slider("PD (% p.a.)", 0.0, 20.0, 2.0, 0.5) / 100
        lgd = st.slider("LGD (%)", 0.0, 100.0, 40.0, 5.0) / 100
        trigger_default = st.checkbox("Simulate Default", False)
        default_month = st.number_input("Default Month", 1, tenure, 12) if trigger_default else None
        trigger_restructure = st.checkbox("Simulate Restructure", False)
        restructure_month = st.number_input("Restructure Month", 1, tenure, 18) if trigger_restructure else None
        restructure_amount = st.number_input("Restructured ($)", 0, facility, facility // 2, step=100_000) if trigger_restructure else 0

    with st.sidebar.expander("🏗️ Tranches", expanded=False):
        n_tranches = st.number_input("Number of Tranches", 1, 5, 1)
        tranches = []
        names = ["Senior A", "Senior B", "Mezzanine", "Junior", "Equity"]
        rem = facility
        for i in range(int(n_tranches)):
            c1, c2, c3 = st.columns(3)
            ts = c1.number_input(f"{names[i]} Size", 0, int(rem), int(rem // (int(n_tranches) - i)), step=100_000, key=f"ts_{i}") if i < int(n_tranches) - 1 else rem
            if i == int(n_tranches) - 1: c1.write(f"**${ts:,.0f}**")
            tr = c2.slider(f"Rate %", 0.0, 30.0, 8.0 + i*3, 0.25, key=f"tr_{i}") / 100
            ti = c3.selectbox(f"Investor", ["Fund", "Syndicate", "CBFM"], key=f"ti_{i}")
            tranches.append({"name": names[i], "size": ts, "rate": tr, "rate_m": to_monthly(tr), "priority": i+1, "investor": ti})
            rem -= ts

    # COMPUTE MODEL
    sched = build_schedule(
        facility, target_irr, tenure, interest_timing, advance_n, capitalise_months,
        bbsy_curve, margin, reset_freq, dd_events, cpr, fee_config,
        default_month, restructure_month, restructure_amount, construction_end,
        property_value, property_growth, actuals_dict
    )

    borrower_cfs = list(sched["Borrower CF"])
    borrower_irr = calc_irr(borrower_cfs)
    investor_cfs = [-facility] + list(sched["Total Payment"].iloc[1:])
    investor_irr = calc_irr(investor_cfs)

    total_fees = sched["Fees"].sum()
    total_interest = sched["Interest"].sum()

    # TABS ENGINE
    t1, t2, t3, t4, t5, t6, t7, t8, t9, t10 = st.tabs([
        "📊 Overview", "📅 Cash Flows", "💵 Fees", "🏗️ Tranches", "💼 Investors",
        "⚠️ Risk", "🏦 Fund", "📋 TLS (Ledger)", "✏️ LMS Actuals Log", "💾 Export & Save"
    ])

    with t1:
        st.subheader("Deal Overview")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Facility", f"${facility:,.0f}")
        c2.metric("Target IRR", f"{target_irr*100:.2f}%")
        c3.metric("Borrower IRR", f"{borrower_irr*100:.2f}%" if not np.isnan(borrower_irr) else "—")
        c4.metric("Investor IRR", f"{investor_irr*100:.2f}%" if not np.isnan(investor_irr) else "—")
        c5.metric("Initial LVR", f"{initial_lvr:.1f}%")
        
        st.markdown("---")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sched["Month"], y=sched["Closing"]/1e6, name="Balance ($M)", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=sched["Month"], y=sched["LVR"], name="LVR (%)", yaxis="y2", line=dict(color="red", dash="dot")))
        fig.update_layout(height=400, yaxis=dict(title="Bal ($M)"), yaxis2=dict(title="LVR (%)", overlaying="y", side="right"))
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        st.subheader("Cash Flows")
        fig_cf = px.bar(sched.iloc[1:], x="Month", y=["Interest Paid", "Principal", "Prepayment", "Fees"], title="Monthly Cash Flows")
        st.plotly_chart(fig_cf, use_container_width=True)

    with t3:
        st.subheader("Fees Breakdown")
        st.metric("Total Fees", f"${total_fees:,.2f}")

    with t4:
        st.subheader("Tranche Waterfall")
        st.write(tranches)

    with t5:
        st.subheader("Investor Return Summary")
        st.metric("Gross Investor IRR", f"{investor_irr*100:.2f}%" if not np.isnan(investor_irr) else "—")

    with t6:
        st.subheader("Risk Sensitivity")
        st.write(f"Annual PD: {pd_annual*100:.1f}%, LGD: {lgd*100:.0f}%")

    with t7:
        st.subheader("Fund Portfolio View")
        st.write("Cross-deal view ready for pooling.")

    with t8:
        st.subheader("📋 TLS (Borrower Transaction Ledger Statement)")
        st.caption("🔒 Internal Ledger with Mode (Actual vs Forecast) Tracking")
        
        display_tls = sched[["Month", "Mode", "Drawdown", "Opening", "Interest", "Interest Paid", "Principal", "Fees", "Total Payment", "Closing", "Current Limit", "LVR"]].copy()
        for c in ["Drawdown", "Opening", "Interest", "Interest Paid", "Principal", "Fees", "Total Payment", "Closing", "Current Limit"]:
            display_tls[c] = display_tls[c].apply(lambda x: f"${x:,.2f}")
        display_tls["LVR"] = display_tls["LVR"].apply(lambda x: f"{x:.1f}%")
        st.dataframe(display_tls, use_container_width=True, height=600)

    with t9:
        st.subheader("✏️ LMS Actuals Logger")
        if not selected_deal_id:
            st.warning("⚠️ Select a saved deal from the left sidebar dropdown to log Actuals.")
        else:
            st.info(f"Logging actual repayments for Deal ID #{selected_deal_id}")
            with st.form("log_actual_form"):
                c1, c2, c3, c4, c5 = st.columns(5)
                act_m = c1.number_input("Month #", 0, tenure, 1)
                act_dd = c2.number_input("Actual Drawdown ($)", 0.0, facility, 0.0, step=10000.0)
                act_int = c3.number_input("Actual Interest Paid ($)", 0.0, facility, 0.0, step=1000.0)
                act_prn = c4.number_input("Actual Principal Paid ($)", 0.0, facility, 0.0, step=10000.0)
                act_fee = c5.number_input("Actual Fees Paid ($)", 0.0, facility, 0.0, step=500.0)
                
                if st.form_submit_button("Save Month Actual"):
                    execute_query("DELETE FROM actuals WHERE deal_id=? AND month=?", (selected_deal_id, act_m))
                    execute_query("INSERT INTO actuals (deal_id, month, actual_drawdown, actual_interest, actual_principal, actual_fees) VALUES (?,?,?,?,?,?)",
                                  (selected_deal_id, act_m, act_dd, act_int, act_prn, act_fee))
                    st.success(f"Saved Actuals for Month {act_m}!")
                    st.rerun()

    with t10:
        st.subheader("💾 Export & Save")
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            sched.to_excel(writer, sheet_name="TLS & Cash Flows", index=False)
        output.seek(0)
        st.download_button("⬇️ Download Full Excel Model", data=output, file_name="deal_model.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
