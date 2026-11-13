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
    
    # 🏗️ CONSTRUCTION FEATURE: Line fee on UNDRAWN balance
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
    arrears_balance = 0.0  # 📋 LMS FEATURE: Track unpaid amounts

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
        
        # 📋 LMS FEATURE: Penalty Interest on Arrears
        if arrears_balance > 0:
            penalty_int = arrears_balance * (penalty_rate_annual / 12)
            month_fees["Penalty Interest"] = penalty_int
            
        if is_actual and "fees" in act:
            total_fees = act["fees"] # Override with actual paid
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
            
            # Update Arrears
            expected_total = expected_cash_int + expected_princ + expected_prepay + sum(month_fees.values())
            actual_total = cash_int + total_princ + total_fees
            shortfall = max(expected_total - actual_total, 0)
            arrears_balance += shortfall
            
        else:
            cash_int = expected_cash_int
            total_princ = min(expected_princ + expected_prepay, balance)
            prepay = expected_prepay
            arrears_balance = max(arrears_balance - total_payment if 'total_payment' in locals() else 0, 0) # Simplistic forecast arrears burndown

        balance -= total_princ
        cum_principal += total_princ
        total_payment = cash_int + total_princ + total_fees
        current_limit = max(facility - cum_principal, 0)

        rows.append({
            "Month": m, "Mode": "Actual" if is_actual else "Forecast", "Drawdown": dd_amt, "Opening": balance + total_princ, "Rate": rate_m,
            "Interest": interest, "Interest Paid": cash_int, "Interest Capitalised": cap_amt, "Principal": expected_princ if not is_actual else total_princ,
            "Prepayment": prepay, "Fees": total_fees, "Fee Detail": month_fees, "Total Payment": total_payment, "Closing": max(balance, 0),
            "Arrears": arrears_balance, "Current Limit": current_limit, "LVR": max(balance, 0) / prop_val * 100 if prop_val > 0 else 0,
            "Covenant Breach": (max(balance, 0) / prop_val * 100) > max_lvr_covenant if prop_val > 0 else False,
            "Borrower CF": -total_payment, "Anniversary": m % 12 == 0
        })

    return pd.DataFrame(rows)

# ═══════════════════════════════════════════════════════════════
#  UI SETUP & NAVIGATION
# ═══════════════════════════════════════════════════════════════
st.set_page_config(page_title="Structured Lending Platform", layout="wide", page_icon="🏦")

# Custom CSS for better looking cards/containers
st.markdown("""
<style>
    div[data-testid="stExpander"] div[role="button"] p {font-size: 1.1rem; font-weight: 600;}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🏦 Admin Platform")
nav_page = st.sidebar.radio("Navigation", ["📊 Dashboard", "🏢 Borrowers", "📄 Deals Directory", "🧮 Deal Modeler & LMS"])

# ═══════════════════════════════════════════════════════════════
#  PAGE: DEAL MODELER & LMS (CENTER SCREEN UI)
# ═══════════════════════════════════════════════════════════════
if nav_page == "🧮 Deal Modeler & LMS":
    
    # Header & Deal Selection
    c_title, c_select = st.columns([1, 2])
    c_title.title("🧮 Deal Modeler")
    db_deals = run_query("SELECT d.id, d.deal_name, b.name FROM deals d JOIN borrowers b ON d.borrower_id = b.id")
    deal_options = ["🛠️ Sandbox / Scratchpad"] + [f"ID {r['id']}: {r['deal_name']} ({r['name']})" for _, r in db_deals.iterrows()]
    selected_deal = c_select.selectbox("📂 Load Database Deal or Use Sandbox", deal_options, label_visibility="collapsed")
    
    selected_deal_id = int(selected_deal.split(":")[0].replace("ID ", "")) if selected_deal != "🛠️ Sandbox / Scratchpad" else None
    actuals_dict = {}
    if selected_deal_id:
        act_df = run_query("SELECT * FROM actuals WHERE deal_id = ?", (selected_deal_id,))
        for _, r in act_df.iterrows():
            actuals_dict[int(r['month'])] = {"drawdown": r['actual_drawdown'], "interest": r['actual_interest'], "principal": r['actual_principal'], "fees": r['actual_fees']}

    # ═══════════════════════════════════════════════════════════════
    #  CENTERED INPUT PARAMETERS
    # ═══════════════════════════════════════════════════════════════
    with st.expander("⚙️ DEAL PARAMETERS & ASSUMPTIONS", expanded=True):
        tab_loan, tab_draw, tab_fees, tab_risk = st.tabs(["📝 Core Loan Terms", "🏗️ Construction & Drawdowns", "💵 Fees & Interest", "⚠️ Risk & Covenants"])
        
        with tab_loan:
            col1, col2, col3, col4 = st.columns(4)
            facility = col1.number_input("Facility Limit ($)", 100_000, 500_000_000, 10_000_000, step=1_000_000)
            target_irr = col2.number_input("Target IRR (%)", 1.0, 35.0, 12.0, 0.5) / 100
            tenure = col3.number_input("Tenure (months)", 3, 120, 24)
            capitalise_months = col4.number_input("PIK Period (months)", 0, tenure, 0)
            
            col1, col2, col3, col4 = st.columns(4)
            property_value = col1.number_input("Initial Prop Value ($)", 0, 100_000_000, 15_000_000, step=500_000)
            property_growth = col2.number_input("Prop Growth (% p.a.)", -10.0, 20.0, 3.0, 0.5) / 100
            max_lvr_covenant = col3.number_input("Max LVR Covenant (%)", 10.0, 100.0, 75.0, 1.0)
            initial_lvr = facility / property_value * 100 if property_value > 0 else 0
            col4.metric("Starting LVR", f"{initial_lvr:.1f}%")

        with tab_draw:
            col1, col2 = st.columns([1, 2])
            construction_end = col1.number_input("Construction End (Month)", 0, tenure, 0, help="Line fee applies until this month")
            n_dd = col1.number_input("Number of Drawdowns", 1, 10, 1)
            
            dd_events = []
            col2.markdown("**Drawdown Schedule**")
            dd_cols = col2.columns(3)
            rem = facility
            for i in range(int(n_dd)):
                dd_m = dd_cols[i%3].number_input(f"DD {i+1} Month", 0, tenure-1, i*(tenure//int(n_dd)), key=f"ddm_{i}")
                dd_a = dd_cols[i%3].number_input(f"DD {i+1} ($)", 0, int(rem), int(rem//(int(n_dd)-i)), step=100_000, key=f"dda_{i}") if i < int(n_dd)-1 else rem
                if i == int(n_dd)-1: dd_cols[i%3].info(f"Final: ${dd_a:,.0f}")
                dd_events.append((dd_m, dd_a))
                rem -= dd_a

        with tab_fees:
            col1, col2, col3, col4 = st.columns(4)
            est_pct = col1.number_input("Establishment (%)", 0.0, 5.0, 1.5, 0.25) / 100
            line_pct = col2.number_input("Line Fee (% p.a.)", 0.0, 5.0, 1.0, 0.25, help="Charged on Undrawn Balance") / 100
            admin_amount = col3.number_input("Admin Fee ($/m)", 0, 10_000, 1_000, step=100)
            discharge_amt = col4.number_input("Discharge Fee ($)", 0, 50_000, 2_000, step=500)
            
            col1, col2, col3, col4 = st.columns(4)
            interest_timing = col1.selectbox("Interest Timing", ["arrears", "advance", "all_advance"], format_func=lambda x: x.replace("_", " ").title())
            penalty_rate_annual = col2.number_input("Penalty Rate (% p.a.)", 0.0, 30.0, 18.0, 1.0) / 100
            risk_review_amt = col3.number_input("Risk Review Fee ($)", 0, 50_000, 5_000, step=500)
            restructure_pct = col4.number_input("Restructure Fee (%)", 0.0, 5.0, 1.0, 0.25) / 100

            fee_config = {"establishment_pct": est_pct, "line_fee_pct": line_pct, "admin_type": "fixed", "admin_amount": admin_amount, "discharge_amount": discharge_amt, "risk_review_amount": risk_review_amt, "restructure_pct": restructure_pct}

        with tab_risk:
            col1, col2, col3 = st.columns(3)
            cpr = col1.number_input("CPR (Prepayment %)", 0.0, 50.0, 0.0, 1.0) / 100
            pd_annual = col2.number_input("PD (% p.a.)", 0.0, 20.0, 2.0, 0.5) / 100
            lgd = col3.number_input("LGD (%)", 0.0, 100.0, 40.0, 5.0) / 100
            
            col1, col2 = st.columns(2)
            default_month = col1.number_input("Simulate Default Month", 0, tenure, 0)
            default_month = default_month if default_month > 0 else None
            restructure_month = col2.number_input("Simulate Restructure Month", 0, tenure, 0)
            restructure_month = restructure_month if restructure_month > 0 else None

    # COMPUTE MODEL
    sched = build_schedule(
        facility, target_irr, tenure, interest_timing, 1, capitalise_months,
        None, 0, 12, dd_events, cpr, fee_config,
        default_month, restructure_month, facility//2, construction_end,
        property_value, property_growth, actuals_dict, penalty_rate_annual, max_lvr_covenant
    )

    # ═══════════════════════════════════════════════════════════════
    #  OUTPUT TABS
    # ═══════════════════════════════════════════════════════════════
    st.markdown("### 📊 Model Outputs & Ledger")
    t1, t2, t3, t4, t5 = st.tabs(["📋 TLS (Ledger)", "📈 Economics", "💵 Fees", "✏️ Log LMS Actuals", "💾 Export"])

    with t1:
        st.caption("🔒 Internal Ledger. Shows Undrawn Line Fees, Arrears, and Covenant Breaches.")
        display_tls = sched[["Month", "Mode", "Opening", "Drawdown", "Interest Paid", "Principal", "Fees", "Arrears", "Closing", "LVR", "Covenant Breach"]].copy()
        
        # Color formatting for breaches and arrears
        def highlight_risk(row):
            styles = [''] * len(row)
            if row['Covenant Breach']: styles[display_tls.columns.get_loc('LVR')] = 'background-color: #ffcccc; color: #990000'
            if row['Arrears'] > 0: styles[display_tls.columns.get_loc('Arrears')] = 'background-color: #ffe6cc; color: #cc0000'
            if row['Mode'] == 'Actual': styles[display_tls.columns.get_loc('Mode')] = 'background-color: #e6f9e6; color: #006600'
            return styles

        for c in ["Opening", "Drawdown", "Interest Paid", "Principal", "Fees", "Arrears", "Closing"]:
            display_tls[c] = display_tls[c].apply(lambda x: f"${x:,.2f}")
        display_tls["LVR"] = display_tls["LVR"].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(display_tls.style.apply(highlight_risk, axis=1), use_container_width=True, height=500)

    with t2:
        c1, c2, c3 = st.columns(3)
        borrower_irr = calc_irr(list(sched["Borrower CF"]))
        c1.metric("Effective Borrower IRR", f"{borrower_irr*100:.2f}%" if not np.isnan(borrower_irr) else "—")
        c2.metric("Total Interest Earned", f"${sched['Interest'].sum():,.0f}")
        c3.metric("Total Fees Earned", f"${sched['Fees'].sum():,.0f}")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sched["Month"], y=sched["Closing"]/1e6, name="Balance ($M)", fill="tozeroy"))
        fig.add_hline(y=facility/1e6, line_dash="dash", annotation_text="Facility Limit")
        fig.update_layout(height=350, title="Facility Utilisation Over Time")
        st.plotly_chart(fig, use_container_width=True)

    with t3:
        st.dataframe(sched[["Month", "Fee Detail"]], use_container_width=True)

    with t4:
        if not selected_deal_id: st.warning("Save/Select a deal to log actuals.")
        else:
            with st.form("log_actual"):
                c1, c2, c3, c4 = st.columns(4)
                act_m = c1.number_input("Month", 1, tenure, 1)
                act_dd = c2.number_input("Drawdown ($)", 0.0, step=10000.0)
                act_int = c3.number_input("Interest Paid ($)", 0.0, step=1000.0)
                act_prn = c4.number_input("Principal Paid ($)", 0.0, step=10000.0)
                if st.form_submit_button("Save Actual Payment"):
                    execute_query("DELETE FROM actuals WHERE deal_id=? AND month=?", (selected_deal_id, act_m))
                    execute_query("INSERT INTO actuals (deal_id, month, actual_drawdown, actual_interest, actual_principal, actual_fees) VALUES (?,?,?,?,?,?)",
                                  (selected_deal_id, act_m, act_dd, act_int, act_prn, 0.0))
                    st.success("Saved!")
                    st.rerun()

    with t5:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            sched.to_excel(writer, sheet_name="Ledger", index=False)
        st.download_button("⬇️ Download Excel Model", data=output.getvalue(), file_name="deal_model.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ═══════════════════════════════════════════════════════════════
#  OTHER PAGES (Dashboard, Borrowers, Deals)
# ═══════════════════════════════════════════════════════════════
elif nav_page == "📊 Dashboard":
    st.title("📊 Portfolio Dashboard")
    st.info("High-level portfolio metrics and status tracking.")

elif nav_page == "🏢 Borrowers":
    st.title("🏢 Borrower Management")
    st.info("Borrower CRM directory.")

elif nav_page == "📄 Deals Directory":
    st.title("📄 Deals Directory")
    st.info("List of all deals. To model a deal, go to the Deal Modeler tab.")
