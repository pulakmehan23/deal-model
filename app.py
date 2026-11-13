import streamlit as st
import sqlite3
import json
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import brentq

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG & STYLING
# ═══════════════════════════════════════════════════════════════
st.set_page_config(page_title="Private Credit Platform", layout="wide", page_icon="🏦")

st.markdown("""
<style>
    .main .block-container {padding-top:1.5rem; max-width:95%;}
    div[data-testid="stMetric"] {background:#f8f9fa; border:1px solid #e9ecef; border-radius:8px; padding:12px; text-align:center;}
    div[data-testid="stMetric"] label {font-size:0.75rem; color:#6c757d;}
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {font-size:1.3rem; font-weight:700;}
    .stTabs [data-baseweb="tab-list"] {gap:8px;}
    .stTabs [data-baseweb="tab"] {padding:8px 16px; font-weight:600;}
    h1,h2,h3 {color:#1a1a2e;}
    .breach-alert {background:#fff3cd; border:1px solid #ffc107; border-radius:8px; padding:12px; margin:8px 0; font-weight:600; color:#856404;}
    .covenant-ok {background:#d4edda; border:1px solid #28a745; border-radius:8px; padding:12px; margin:8px 0; font-weight:600; color:#155724;}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════
DB = "lms_data.db"

def get_conn():
    return sqlite3.connect(DB, check_same_thread=False)

def init_db():
    c = get_conn()
    c.execute("""CREATE TABLE IF NOT EXISTS borrowers (
        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, abn TEXT, group_name TEXT,
        risk_grade TEXT DEFAULT 'Investment Grade', industry TEXT DEFAULT 'Real Estate',
        status TEXT DEFAULT 'Active')""")
    c.execute("""CREATE TABLE IF NOT EXISTS deals (
        id INTEGER PRIMARY KEY AUTOINCREMENT, borrower_id INTEGER, deal_name TEXT,
        facility_amount REAL, deal_status TEXT DEFAULT 'Pipeline',
        settlement_date TEXT, maturity_date TEXT, facility_type TEXT DEFAULT 'Construction',
        security_type TEXT DEFAULT 'First Mortgage', property_value REAL DEFAULT 0,
        params_json TEXT DEFAULT '{}',
        FOREIGN KEY (borrower_id) REFERENCES borrowers(id))""")
    c.execute("""CREATE TABLE IF NOT EXISTS actuals (
        id INTEGER PRIMARY KEY AUTOINCREMENT, deal_id INTEGER, month INTEGER,
        actual_drawdown REAL DEFAULT 0, actual_interest REAL DEFAULT 0,
        actual_principal REAL DEFAULT 0, actual_fees REAL DEFAULT 0,
        FOREIGN KEY (deal_id) REFERENCES deals(id))""")
    c.commit()
    c.close()

init_db()

# ═══════════════════════════════════════════════════════════════
# FINANCIAL ENGINE
# ═══════════════════════════════════════════════════════════════
def to_monthly(r):
    return (1.0 + r) ** (1.0 / 12.0) - 1.0

def to_annual(m):
    return (1.0 + m) ** 12.0 - 1.0

def cpr_to_smm(c):
    return 1.0 - (1.0 - c) ** (1.0 / 12.0)

def calc_irr(cfs):
    if not cfs or len(cfs) < 2:
        return 0.0
    vals = [float(x) for x in cfs]
    if all(v >= 0 for v in vals) or all(v <= 0 for v in vals):
        return 0.0
    def npv(r):
        return sum(v / (1 + r) ** i for i, v in enumerate(vals))
    try:
        m = brentq(npv, -0.99, 10.0, maxiter=1000)
        return to_annual(m)
    except:
        return 0.0

def compute_fees(p):
    fa = float(p.get("facility_amount", 0))
    term = int(p.get("term_months", 12))
    constr = int(p.get("construction_months", 6))
    estab_r = float(p.get("estab_fee_pct", 1.5)) / 100.0
    line_r = float(p.get("line_fee_pct", 0.5)) / 100.0
    admin_m = float(p.get("admin_fee_monthly", 500))
    discharge = float(p.get("discharge_fee", 1500))
    risk_fee = float(p.get("risk_review_fee", 2500))
    restruct_fee = float(p.get("restructure_fee", 0))
    legal_fee = float(p.get("legal_fee", 3500))
    other_upfront = float(p.get("other_upfront_fees", 0))
    estab_total = fa * estab_r
    line_total = fa * line_r * (constr / 12.0)
    admin_total = admin_m * term
    return {
        "establishment": estab_total,
        "line_fee_total": line_total,
        "admin_total": admin_total,
        "discharge": discharge,
        "risk_review": risk_fee,
        "restructure": restruct_fee,
        "legal_fee": legal_fee,
        "other_upfront": other_upfront,
        "upfront_total": estab_total + legal_fee + risk_fee + other_upfront,
        "line_fee_rate": line_r,
        "admin_monthly": admin_m,
    }

def build_schedule(p, actuals_map):
    fa = float(p.get("facility_amount", 0))
    term = int(p.get("term_months", 12))
    constr = int(p.get("construction_months", 6))
    annual_r = float(p.get("borrower_irr", 12.0)) / 100.0
    mr = to_monthly(annual_r)
    cpr = float(p.get("cpr", 0)) / 100.0
    smm = cpr_to_smm(cpr) if cpr > 0 else 0.0
    pv = float(p.get("property_value", 0))
    pv_growth = float(p.get("pv_growth_annual", 3.0)) / 100.0
    pv_mg = (1 + pv_growth) ** (1.0 / 12.0) - 1.0
    cov_lvr = float(p.get("covenant_lvr", 75.0)) / 100.0
    penalty_r = float(p.get("penalty_rate", 4.0)) / 100.0
    penalty_mr = to_monthly(penalty_r)
    dd_schedule = json.loads(p.get("drawdown_schedule", "{}"))
    fees = compute_fees(p)
    line_r = fees["line_fee_rate"]
    admin_m = fees["admin_monthly"]
    upfront = fees["upfront_total"]

    rows = []
    bal = 0.0
    cum_drawn = 0.0
    cum_interest = 0.0
    arrears = 0.0
    borrower_cfs = []

    for m in range(1, term + 1):
        dd_gross = float(dd_schedule.get(str(m), 0))
        dd_net = dd_gross
        fees_this = 0.0

        if m == 1 and dd_gross > 0:
            fees_this = min(upfront, dd_gross)
            dd_net = dd_gross - fees_this

        cum_drawn += dd_gross
        bal += dd_gross

        interest_due = bal * mr
        penalty_int = arrears * penalty_mr if arrears > 0 else 0.0

        in_constr = m <= constr
        undrawn = max(fa - cum_drawn, 0)
        line_fee = undrawn * (line_r / 12.0) if in_constr else 0.0

        prepay = bal * smm if not in_constr and bal > 0 else 0.0
        sched_prin = 0.0 if in_constr else (bal / max(term - m + 1, 1))
        total_prin = sched_prin + prepay

        total_due = interest_due + penalty_int + total_prin + line_fee + admin_m

        act = actuals_map.get(m, {})
        act_int = float(act.get("actual_interest", 0))
        act_prin = float(act.get("actual_principal", 0))
        act_fees = float(act.get("actual_fees", 0))
        act_total = act_int + act_prin + act_fees

        has_actual = bool(act)
        if has_actual:
            shortfall = max(total_due - act_total, 0)
            arrears += shortfall

        end_bal = bal - total_prin
        if end_bal < 0:
            end_bal = 0

        cur_pv = pv * ((1 + pv_mg) ** m) if pv > 0 else 0
        lvr = (end_bal / cur_pv * 100) if cur_pv > 0 else 0
        breach = lvr > (cov_lvr * 100)

        cf = 0.0
        if dd_net > 0:
            cf -= dd_net
        cf += interest_due + penalty_int + total_prin + line_fee + admin_m
        borrower_cfs.append(cf)

        cum_interest += interest_due

        rows.append({
            "Month": m,
            "Drawdown_Gross": dd_gross,
            "Upfront_Fees_Deducted": fees_this,
            "Drawdown_Net": dd_net,
            "Opening_Balance": bal - dd_gross,
            "Balance_After_Draw": bal,
            "Interest_Due": interest_due,
            "Penalty_Interest": penalty_int,
            "Line_Fee": line_fee,
            "Admin_Fee": admin_m,
            "Scheduled_Principal": sched_prin,
            "Prepayment": prepay,
            "Total_Principal": total_prin,
            "Total_Due": total_due,
            "Actual_Interest": act_int if has_actual else None,
            "Actual_Principal": act_prin if has_actual else None,
            "Actual_Fees": act_fees if has_actual else None,
            "Arrears": arrears,
            "Ending_Balance": end_bal,
            "Property_Value": cur_pv,
            "LVR_pct": lvr,
            "Covenant_Breach": breach,
            "Cum_Interest": cum_interest,
            "Cum_Drawn": cum_drawn,
            "Is_Construction": in_constr,
        })
        bal = end_bal

    borrower_irr = calc_irr(borrower_cfs)
    return pd.DataFrame(rows), fees, borrower_irr, borrower_cfs

# ═══════════════════════════════════════════════════════════════
# INVESTOR ALLOCATION & LEDGER
# ═══════════════════════════════════════════════════════════════
FUND_NAMES = ["BPCFMC", "BPCFLC", "CBCF"]
FUND_RATIOS = {"BPCFMC": 0.50, "BPCFLC": 0.15, "CBCF": 0.35}

def build_investor_ledger(schedule_df, p):
    fund_target_irr = float(p.get("fund_target_irr", 8.0)) / 100.0
    cbfm_target_irr = float(p.get("cbfm_target_irr", 10.0)) / 100.0
    synd_target_irr = float(p.get("syndicate_target_irr", 12.0)) / 100.0

    fund_mr = to_monthly(fund_target_irr)
    cbfm_mr = to_monthly(cbfm_target_irr)
    synd_mr = to_monthly(synd_target_irr)

    cbfm_alloc_pct = float(p.get("cbfm_alloc_pct", 0)) / 100.0
    synd_alloc_pct = float(p.get("syndicate_alloc_pct", 0)) / 100.0

    investor_types = FUND_NAMES + (["CBFM"] if cbfm_alloc_pct > 0 else []) + (["Syndicate"] if synd_alloc_pct > 0 else [])

    inv_bal = {n: 0.0 for n in investor_types}
    inv_funded = {n: 0.0 for n in investor_types}
    inv_interest_total = {n: 0.0 for n in investor_types}
    inv_principal_ret = {n: 0.0 for n in investor_types}
    inv_cfs = {n: [] for n in investor_types}

    ledger_rows = []
    first_dd_done = False

    for _, row in schedule_df.iterrows():
        m = int(row["Month"])
        dd_net = float(row["Drawdown_Net"])
        total_prin = float(row["Total_Principal"])

        alloc = {n: 0.0 for n in investor_types}
        if dd_net > 0:
            if not first_dd_done:
                for fn in FUND_NAMES:
                    alloc[fn] = dd_net * FUND_RATIOS[fn]
                first_dd_done = True
            else:
                cbfm_dd = dd_net * cbfm_alloc_pct
                synd_dd = dd_net * synd_alloc_pct
                fund_dd = dd_net - cbfm_dd - synd_dd
                if "CBFM" in alloc: alloc["CBFM"] = cbfm_dd
                if "Syndicate" in alloc: alloc["Syndicate"] = synd_dd
                for fn in FUND_NAMES:
                    alloc[fn] = fund_dd * FUND_RATIOS[fn]

        for n in investor_types:
            inv_bal[n] += alloc[n]
            inv_funded[n] += alloc[n]

        total_inv_bal = sum(inv_bal.values())
        interest_this = {}
        for n in investor_types:
            if n in FUND_NAMES: interest_this[n] = inv_bal[n] * fund_mr
            elif n == "CBFM": interest_this[n] = inv_bal[n] * cbfm_mr
            elif n == "Syndicate": interest_this[n] = inv_bal[n] * synd_mr
            else: interest_this[n] = 0.0
            inv_interest_total[n] += interest_this[n]

        prin_return = {}
        if total_inv_bal > 0 and total_prin > 0:
            for n in investor_types:
                share = inv_bal[n] / total_inv_bal if total_inv_bal > 0 else 0
                prin_return[n] = total_prin * share
                inv_bal[n] -= prin_return[n]
                if inv_bal[n] < 0: inv_bal[n] = 0
                inv_principal_ret[n] += prin_return[n]
        else:
            for n in investor_types:
                prin_return[n] = 0.0

        for n in investor_types:
            cf = -alloc[n] + interest_this[n] + prin_return[n]
            inv_cfs[n].append(cf)

        for n in investor_types:
            ledger_rows.append({
                "Month": m,
                "Investor": n,
                "Capital_Called": alloc[n],
                "Interest_Earned": interest_this[n],
                "Principal_Returned": prin_return[n],
                "Ending_Balance": inv_bal[n],
                "Cumulative_Funded": inv_funded[n],
                "Cumulative_Interest": inv_interest_total[n],
            })

    ledger_df = pd.DataFrame(ledger_rows)

    inv_xirr = {n: calc_irr(inv_cfs[n]) for n in investor_types}

    target_map = {fn: fund_target_irr for fn in FUND_NAMES}
    target_map["CBFM"] = cbfm_target_irr
    target_map["Syndicate"] = synd_target_irr

    summary_rows = []
    for n in investor_types:
        summary_rows.append({
            "Investor": n,
            "Total_Funded": inv_funded[n],
            "Total_Interest_Earned": inv_interest_total[n],
            "Total_Principal_Returned": inv_principal_ret[n],
            "Outstanding_Balance": inv_bal[n],
            "Target_XIRR": target_map.get(n, 0) * 100,
            "Actual_XIRR": inv_xirr.get(n, 0) * 100,
        })
    summary_df = pd.DataFrame(summary_rows)

    return ledger_df, summary_df, inv_xirr, inv_cfs

# ═══════════════════════════════════════════════════════════════
# P&L ENGINE
# ═══════════════════════════════════════════════════════════════
def build_pnl(schedule_df, fees, investor_summary_df, p):
    total_borrower_interest = schedule_df["Interest_Due"].sum()
    total_penalty_interest = schedule_df["Penalty_Interest"].sum()
    total_line_fees = schedule_df["Line_Fee"].sum()
    total_admin_fees = schedule_df["Admin_Fee"].sum()

    estab = fees["establishment"]
    legal = fees["legal_fee"]
    risk = fees["risk_review"]
    discharge = fees["discharge"]
    other_upfront = fees["other_upfront"]
    restructure = fees["restructure"]

    total_fee_income = estab + legal + risk + discharge + other_upfront + restructure + total_line_fees + total_admin_fees
    total_income = total_borrower_interest + total_penalty_interest + total_fee_income

    total_investor_expense = investor_summary_df["Total_Interest_Earned"].sum() if len(investor_summary_df) > 0 else 0
    net_revenue = total_income - total_investor_expense

    total_funded = investor_summary_df["Total_Funded"].sum() if len(investor_summary_df) > 0 else 0
    wtd_inv_rate = 0
    if total_funded > 0:
        for _, row in investor_summary_df.iterrows():
            wtd_inv_rate += (row["Total_Funded"] / total_funded) * row["Target_XIRR"]

    borrower_irr_pct = float(p.get("borrower_irr", 12.0))
    spread = borrower_irr_pct - wtd_inv_rate

    return {
        "Borrower Interest Income": total_borrower_interest,
        "Penalty Interest Income": total_penalty_interest,
        "Establishment Fee": estab,
        "Legal Fee": legal,
        "Risk Review Fee": risk,
        "Line Fees (Construction)": total_line_fees,
        "Admin Fees": total_admin_fees,
        "Discharge Fee": discharge,
        "Other Upfront Fees": other_upfront,
        "Restructure Fee": restructure,
        "─── Total Income ───": total_income,
        "Investor Interest Expense": total_investor_expense,
        "─── Net Revenue ───": net_revenue,
        "": "",
        "Borrower IRR (%)": borrower_irr_pct,
        "Wtd Avg Investor Cost (%)": wtd_inv_rate,
        "Platform Spread (bps)": spread * 100,
    }

# ═══════════════════════════════════════════════════════════════
# SPV CASHFLOWS
# ═══════════════════════════════════════════════════════════════
def build_cashflows(schedule_df, investor_ledger_df, fees, p):
    term = int(p.get("term_months", 12))
    rows = []
    cum_spv = 0.0

    for m in range(1, term + 1):
        s = schedule_df[schedule_df["Month"] == m].iloc[0]
        inv_m = investor_ledger_df[investor_ledger_df["Month"] == m]

        investor_capital_called = inv_m["Capital_Called"].sum()
        borrower_interest_received = float(s["Interest_Due"]) + float(s["Penalty_Interest"])
        borrower_fees_received = float(s["Line_Fee"]) + float(s["Admin_Fee"])

        if m == 1:
            borrower_fees_received += fees["upfront_total"]

        total_inflow = investor_capital_called + borrower_interest_received + borrower_fees_received

        net_disbursed = float(s["Drawdown_Net"])
        investor_interest_paid = inv_m["Interest_Earned"].sum()
        investor_principal_returned = inv_m["Principal_Returned"].sum()

        total_outflow = net_disbursed + investor_interest_paid + investor_principal_returned
        net_cf = total_inflow - total_outflow
        cum_spv += net_cf

        rows.append({
            "Month": m,
            "Investor_Capital_Called": investor_capital_called,
            "Borrower_Interest_Received": borrower_interest_received,
            "Borrower_Fees_Received": borrower_fees_received,
            "Total_Inflow": total_inflow,
            "Net_Disbursed_to_Borrower": net_disbursed,
            "Investor_Interest_Paid": investor_interest_paid,
            "Investor_Principal_Returned": investor_principal_returned,
            "Total_Outflow": total_outflow,
            "Net_SPV_Cashflow": net_cf,
            "Cumulative_SPV_Cash": cum_spv,
        })
    return pd.DataFrame(rows)

# ═══════════════════════════════════════════════════════════════
# DB HELPERS
# ═══════════════════════════════════════════════════════════════
def load_borrowers():
    c = get_conn()
    df = pd.read_sql("SELECT * FROM borrowers ORDER BY name", c)
    c.close()
    return df

def save_borrower(name, abn, group, grade, industry):
    c = get_conn()
    c.execute("INSERT INTO borrowers (name, abn, group_name, risk_grade, industry) VALUES (?,?,?,?,?)",
              (name, abn, group, grade, industry))
    c.commit()
    c.close()

def load_deals():
    c = get_conn()
    df = pd.read_sql("""SELECT d.*, b.name as borrower_name FROM deals d
                        LEFT JOIN borrowers b ON d.borrower_id = b.id ORDER BY d.id DESC""", c)
    c.close()
    return df

def save_deal(borrower_id, deal_name, facility, status, settle, mature, ftype, stype, pv, params):
    c = get_conn()
    c.execute("""INSERT INTO deals (borrower_id, deal_name, facility_amount, deal_status,
              settlement_date, maturity_date, facility_type, security_type, property_value, params_json)
              VALUES (?,?,?,?,?,?,?,?,?,?)""",
              (borrower_id, deal_name, facility, status, settle, mature, ftype, stype, pv, json.dumps(params)))
    c.commit()
    c.close()

def update_deal_params(deal_id, params):
    c = get_conn()
    c.execute("UPDATE deals SET params_json=? WHERE id=?", (json.dumps(params), deal_id))
    c.commit()
    c.close()

def load_actuals(deal_id):
    c = get_conn()
    df = pd.read_sql("SELECT * FROM actuals WHERE deal_id=? ORDER BY month", c, params=(deal_id,))
    c.close()
    return df

def save_actual(deal_id, month, dd, interest, principal, fees_amt):
    c = get_conn()
    existing = c.execute("SELECT id FROM actuals WHERE deal_id=? AND month=?", (deal_id, month)).fetchone()
    if existing:
        c.execute("UPDATE actuals SET actual_drawdown=?, actual_interest=?, actual_principal=?, actual_fees=? WHERE id=?",
                  (dd, interest, principal, fees_amt, existing[0]))
    else:
        c.execute("INSERT INTO actuals (deal_id, month, actual_drawdown, actual_interest, actual_principal, actual_fees) VALUES (?,?,?,?,?,?)",
                  (deal_id, month, dd, interest, principal, fees_amt))
    c.commit()
    c.close()

def fmt(v):
    if v is None: return ""
    return f"${v:,.2f}"

def pct(v):
    return f"{v:.2f}%"

# ═══════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════
def page_dashboard():
    st.title("📊 Platform Dashboard")
    borr = load_borrowers()
    deals = load_deals()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Borrowers", len(borr))
    col2.metric("Total Deals", len(deals))
    total_facility = deals["facility_amount"].sum() if len(deals) > 0 else 0
    col3.metric("Total Facilities", fmt(total_facility))
    active = len(deals[deals["deal_status"] == "Active"]) if len(deals) > 0 else 0
    col4.metric("Active Deals", active)

    if len(deals) > 0:
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Deals by Status")
            status_counts = deals["deal_status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            fig = px.pie(status_counts, values="Count", names="Status", hole=0.4)
            fig.update_layout(height=350, margin=dict(t=30, b=30))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("Facility by Type")
            if "facility_type" in deals.columns:
                type_sum = deals.groupby("facility_type")["facility_amount"].sum().reset_index()
                type_sum.columns = ["Type", "Amount"]
                fig2 = px.bar(type_sum, x="Type", y="Amount", color="Type")
                fig2.update_layout(height=350, margin=dict(t=30, b=30), showlegend=False)
                fig2.update_yaxes(tickprefix="$", tickformat=",.")
                st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Recent Deals")
        display_cols = ["id", "borrower_name", "deal_name", "facility_amount", "deal_status", "facility_type"]
        avail = [c for c in display_cols if c in deals.columns]
        st.dataframe(deals[avail].head(10), use_container_width=True, hide_index=True)
    else:
        st.info("No deals yet. Create borrowers and deals to get started.")

# ═══════════════════════════════════════════════════════════════
# PAGE: BORROWERS
# ═══════════════════════════════════════════════════════════════
def page_borrowers():
    st.title("🏢 Borrowers Directory")
    with st.expander("➕ Add New Borrower", expanded=False):
        with st.form("add_borrower"):
            c1, c2 = st.columns(2)
            name = c1.text_input("Borrower Name*")
            abn = c2.text_input("ABN/ACN")
            c3, c4 = st.columns(2)
            group = c3.text_input("Group / Sponsor")
            grade = c4.selectbox("Risk Grade", ["Investment Grade", "Sub-Investment Grade", "Unrated"])
            industry = st.selectbox("Industry", ["Real Estate", "Infrastructure", "Healthcare", "Agriculture", "Manufacturing", "Technology", "Other"])
            if st.form_submit_button("💾 Save Borrower"):
                if name.strip():
                    save_borrower(name.strip(), abn.strip(), group.strip(), grade, industry)
                    st.success(f"Borrower '{name}' saved!")
                    st.rerun()
                else:
                    st.error("Name is required.")

    borr = load_borrowers()
    if len(borr) > 0:
        st.subheader(f"All Borrowers ({len(borr)})")
        st.dataframe(borr, use_container_width=True, hide_index=True)
    else:
        st.info("No borrowers added yet.")

# ═══════════════════════════════════════════════════════════════
# PAGE: DEALS DIRECTORY
# ═══════════════════════════════════════════════════════════════
def page_deals():
    st.title("📄 Deals Directory")
    borr = load_borrowers()
    if len(borr) == 0:
        st.warning("Add borrowers first.")
        return

    with st.expander("➕ Create New Deal", expanded=False):
        with st.form("add_deal"):
            c1, c2 = st.columns(2)
            borrower_sel = c1.selectbox("Borrower", borr["name"].tolist())
            deal_name = c2.text_input("Deal Name*")
            c3, c4 = st.columns(2)
            facility = c3.number_input("Facility Amount ($)", min_value=0.0, value=5000000.0, step=100000.0)
            status = c4.selectbox("Status", ["Pipeline", "Active", "Settled", "Closed", "Default"])
            c5, c6 = st.columns(2)
            settle = c5.date_input("Settlement Date", value=date.today())
            mature = c6.date_input("Maturity Date", value=date.today() + timedelta(days=365))
            c7, c8 = st.columns(2)
            ftype = c7.selectbox("Facility Type", ["Construction", "Term Loan", "Bridging", "Mezzanine", "Revolving"])
            stype = c8.selectbox("Security Type", ["First Mortgage", "Second Mortgage", "Unsecured", "Personal Guarantee"])
            pv = st.number_input("Property / Security Value ($)", min_value=0.0, value=8000000.0, step=100000.0)

            if st.form_submit_button("💾 Save Deal"):
                if deal_name.strip():
                    bid = int(borr[borr["name"] == borrower_sel]["id"].values[0])
                    save_deal(bid, deal_name.strip(), facility, status, str(settle), str(mature), ftype, stype, pv, {})
                    st.success(f"Deal '{deal_name}' created!")
                    st.rerun()
                else:
                    st.error("Deal name is required.")

    deals = load_deals()
    if len(deals) > 0:
        st.subheader(f"All Deals ({len(deals)})")
        st.dataframe(deals[["id", "borrower_name", "deal_name", "facility_amount", "deal_status",
                            "facility_type", "settlement_date", "maturity_date"]], use_container_width=True, hide_index=True)
    else:
        st.info("No deals created yet.")

# ═══════════════════════════════════════════════════════════════
# PAGE: DEAL MODELER & LMS
# ═══════════════════════════════════════════════════════════════
def page_modeler():
    st.title("🧮 Deal Modeler & Loan Management System")
    deals = load_deals()
    if len(deals) == 0:
        st.warning("Create a deal first in the Deals Directory.")
        return

    deal_options = {f"{r['deal_name']} (ID:{r['id']})": r['id'] for _, r in deals.iterrows()}
    selected = st.selectbox("Select Deal", list(deal_options.keys()))
    deal_id = deal_options[selected]
    deal_row = deals[deals["id"] == deal_id].iloc[0]
    existing_params = json.loads(deal_row["params_json"]) if deal_row["params_json"] else {}
    fa = float(deal_row["facility_amount"])
    pv = float(deal_row.get("property_value", 0) or 0)

    # INPUT PANEL
    with st.expander("⚙️ Deal Parameters & Assumptions", expanded=True):
        st.markdown("##### Core Terms")
        c1, c2, c3, c4 = st.columns(4)
        term = c1.number_input("Term (Months)", 1, 120, int(existing_params.get("term_months", 12)))
        constr = c2.number_input("Construction Period (Mo)", 0, term, int(existing_params.get("construction_months", min(6, term))))
        borrower_irr = c3.number_input("Borrower IRR (% pa)", 0.0, 50.0, float(existing_params.get("borrower_irr", 12.0)), 0.25)
        cpr = c4.number_input("CPR (%)", 0.0, 50.0, float(existing_params.get("cpr", 0.0)), 0.5)

        st.markdown("##### Investor Returns & Allocation")
        ic1, ic2, ic3 = st.columns(3)
        fund_target_irr = ic1.number_input("Fund Target XIRR (% pa)", 0.0, 50.0, float(existing_params.get("fund_target_irr", 8.0)), 0.25)
        cbfm_target_irr = ic2.number_input("CBFM Target XIRR (% pa)", 0.0, 50.0, float(existing_params.get("cbfm_target_irr", 10.0)), 0.25)
        synd_target_irr = ic3.number_input("Syndicate Target XIRR (% pa)", 0.0, 50.0, float(existing_params.get("syndicate_target_irr", 12.0)), 0.25)
        ia1, ia2 = st.columns(2)
        cbfm_alloc = ia1.number_input("CBFM Allocation for DD2+ (%)", 0.0, 100.0, float(existing_params.get("cbfm_alloc_pct", 0.0)), 5.0)
        synd_alloc = ia2.number_input("Syndicate Allocation for DD2+ (%)", 0.0, 100.0, float(existing_params.get("syndicate_alloc_pct", 0.0)), 5.0)
        if cbfm_alloc + synd_alloc > 100:
            st.error("CBFM + Syndicate cannot exceed 100%")

        st.markdown("##### Property & Covenants")
        p1, p2, p3, p4 = st.columns(4)
        pv_growth = p1.number_input("Property Growth (% pa)", -10.0, 30.0, float(existing_params.get("pv_growth_annual", 3.0)), 0.5)
        cov_lvr = p2.number_input("Covenant LVR (%)", 0.0, 100.0, float(existing_params.get("covenant_lvr", 75.0)), 1.0)
        penalty_rate = p3.number_input("Penalty Rate (% pa)", 0.0, 30.0, float(existing_params.get("penalty_rate", 4.0)), 0.5)

        st.markdown("##### Fees")
        f1, f2, f3, f4 = st.columns(4)
        estab_pct = f1.number_input("Establishment Fee (%)", 0.0, 10.0, float(existing_params.get("estab_fee_pct", 1.5)), 0.25)
        line_pct = f2.number_input("Undrawn Line Fee (% pa)", 0.0, 5.0, float(existing_params.get("line_fee_pct", 0.5)), 0.1)
        admin_m = f3.number_input("Admin Fee ($/mo)", 0.0, 5000.0, float(existing_params.get("admin_fee_monthly", 500.0)), 50.0)
        discharge_f = f4.number_input("Discharge Fee ($)", 0.0, 50000.0, float(existing_params.get("discharge_fee", 1500.0)), 100.0)
        f5, f6, f7 = st.columns(3)
        risk_f = f5.number_input("Risk Review Fee ($)", 0.0, 50000.0, float(existing_params.get("risk_review_fee", 2500.0)), 100.0)
        legal_f = f6.number_input("Legal Fee ($)", 0.0, 50000.0, float(existing_params.get("legal_fee", 3500.0)), 100.0)
        other_upfront = f7.number_input("Other Upfront Fees ($)", 0.0, 100000.0, float(existing_params.get("other_upfront_fees", 0.0)), 100.0)
        restructure_f = st.number_input("Restructure Fee ($)", 0.0, 100000.0, float(existing_params.get("restructure_fee", 0.0)), 100.0)

        st.markdown("##### Drawdown Schedule")
        dd_existing = json.loads(existing_params.get("drawdown_schedule", "{}"))
        dd_cols = st.columns(min(term, 6))
        dd_schedule = {}
        for i in range(term):
            with dd_cols[i % len(dd_cols)]:
                val = st.number_input(f"DD Mo {i+1} ($)", 0.0, fa, float(dd_existing.get(str(i+1), 0.0)), 10000.0, key=f"dd_{i+1}")
                if val > 0: dd_schedule[str(i+1)] = val

    params = {
        "facility_amount": fa, "term_months": term, "construction_months": constr,
        "borrower_irr": borrower_irr, "cpr": cpr,
        "fund_target_irr": fund_target_irr, "cbfm_target_irr": cbfm_target_irr,
        "syndicate_target_irr": synd_target_irr,
        "cbfm_alloc_pct": cbfm_alloc, "syndicate_alloc_pct": synd_alloc,
        "property_value": pv, "pv_growth_annual": pv_growth, "covenant_lvr": cov_lvr,
        "penalty_rate": penalty_rate,
        "estab_fee_pct": estab_pct, "line_fee_pct": line_pct,
        "admin_fee_monthly": admin_m, "discharge_fee": discharge_f,
        "risk_review_fee": risk_f, "legal_fee": legal_f,
        "other_upfront_fees": other_upfront, "restructure_fee": restructure_f,
        "drawdown_schedule": json.dumps(dd_schedule),
    }

    if st.button("💾 Save Parameters & Run Model", type="primary", use_container_width=True):
        update_deal_params(deal_id, params)
        st.success("Parameters saved!")

    actuals_df = load_actuals(deal_id)
    actuals_map = {}
    for _, ar in actuals_df.iterrows():
        actuals_map[int(ar["month"])] = {
            "actual_drawdown": ar["actual_drawdown"],
            "actual_interest": ar["actual_interest"],
            "actual_principal": ar["actual_principal"],
            "actual_fees": ar["actual_fees"],
        }

    schedule_df, fees, borrower_irr_calc, borrower_cfs = build_schedule(params, actuals_map)
    investor_ledger_df, investor_summary_df, inv_xirr, inv_cfs = build_investor_ledger(schedule_df, params)
    pnl_dict = build_pnl(schedule_df, fees, investor_summary_df, params)
    cashflow_df = build_cashflows(schedule_df, investor_ledger_df, fees, params)

    st.markdown("---")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Facility", fmt(fa))
    k2.metric("Total Drawn", fmt(schedule_df["Drawdown_Gross"].sum()))
    k3.metric("Total Interest", fmt(schedule_df["Interest_Due"].sum()))
    k4.metric("Borrower IRR", pct(borrower_irr_calc * 100))
    wtd_inv = float(pnl_dict.get("Wtd Avg Investor Cost (%)", 0))
    k5.metric("Wtd Inv Cost", pct(wtd_inv))
    k6.metric("Spread (bps)", f"{float(pnl_dict.get('Platform Spread (bps)', 0)):.0f}")

    tabs = st.tabs(["📋 TLS Ledger", "🏦 Investor Base", "📈 P&L", "💸 Cashflows",
                     "📊 Economics & Settlement", "💵 Fees", "✏️ Log Actuals", "💾 Export"])

    # --- TAB: TLS Ledger ---
    with tabs[0]:
        st.subheader("📋 Borrower True Loan Servicing Ledger")
        display_cols = ["Month", "Drawdown_Gross", "Upfront_Fees_Deducted", "Drawdown_Net",
                       "Opening_Balance", "Balance_After_Draw", "Interest_Due", "Penalty_Interest",
                       "Line_Fee", "Admin_Fee", "Scheduled_Principal", "Prepayment", "Total_Principal",
                       "Total_Due", "Arrears", "Ending_Balance", "Property_Value", "LVR_pct", "Covenant_Breach"]
        tls = schedule_df[display_cols].copy()

        breaches = tls[tls["Covenant_Breach"] == True]
        if len(breaches) > 0:
            st.markdown(f'<div class="breach-alert">⚠️ COVENANT BREACH detected in months: {", ".join(str(int(m)) for m in breaches["Month"].tolist())}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="covenant-ok">✅ All covenants within limits</div>', unsafe_allow_html=True)

        def highlight_arrears(row):
            styles = [""] * len(row)
            try:
                arr_idx = tls.columns.get_loc("Arrears")
                if float(row.iloc[arr_idx]) > 0:
                    styles[arr_idx] = "background-color: #ffe0e0; font-weight: bold"
            except: pass
            try:
                breach_idx = tls.columns.get_loc("Covenant_Breach")
                if row.iloc[breach_idx]:
                    lvr_idx = tls.columns.get_loc("LVR_pct")
                    styles[lvr_idx] = "background-color: #ff6b6b; color: white; font-weight: bold"
                    styles[breach_idx] = "background-color: #ff6b6b; color: white; font-weight: bold"
            except: pass
            return styles

        money_cols = [c for c in tls.columns if c not in ["Month", "LVR_pct", "Covenant_Breach", "Is_Construction"]]
        fmt_dict = {c: "${:,.2f}" for c in money_cols}
        fmt_dict["LVR_pct"] = "{:.1f}%"
        fmt_dict["Month"] = "{:.0f}"
        styled_tls = tls.style.apply(highlight_arrears, axis=1).format(fmt_dict)
        st.dataframe(styled_tls, use_container_width=True, height=500)

        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=schedule_df["Month"], y=schedule_df["Ending_Balance"], mode="lines+markers", name="Loan Balance", line=dict(color="#1f77b4")))
            fig.add_trace(go.Scatter(x=schedule_df["Month"], y=schedule_df["Property_Value"], mode="lines", name="Property Value", line=dict(color="#2ca02c", dash="dash")))
            fig.update_layout(title="Balance vs Property Value", xaxis_title="Month", yaxis_title="$", yaxis_tickprefix="$", height=350)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = go.Figure()
            colors = ["#2ca02c" if not b else "#ff4444" for b in schedule_df["Covenant_Breach"]]
            fig2.add_trace(go.Bar(x=schedule_df["Month"], y=schedule_df["LVR_pct"], marker_color=colors, name="LVR"))
            fig2.add_hline(y=float(cov_lvr), line_dash="dash", line_color="red", annotation_text=f"Covenant: {cov_lvr}%")
            fig2.update_layout(title="Dynamic LVR Tracking", xaxis_title="Month", yaxis_title="LVR %", height=350)
            st.plotly_chart(fig2, use_container_width=True)

    # --- TAB: Investor Base ---
    with tabs[1]:
        st.subheader("🏦 Investor Base & Returns")
        st.info("""
        **DD1 (First Drawdown)**: 100% allocated to Funds → BPCFMC (50%), BPCFLC (15%), CBCF (35%)
        **DD2+ (Subsequent)**: Allocated across Funds, CBFM, and Syndicate per your specified percentages.
        Investor capital called = Net cash disbursed to borrower (after upfront fee deduction).
        """)

        st.markdown("##### Investor Summary")
        if len(investor_summary_df) > 0:
            inv_styled = investor_summary_df.style.format({
                "Total_Funded": "${:,.2f}",
                "Total_Interest_Earned": "${:,.2f}",
                "Total_Principal_Returned": "${:,.2f}",
                "Outstanding_Balance": "${:,.2f}",
                "Target_XIRR": "{:.2f}%",
                "Actual_XIRR": "{:.2f}%",
            })
            st.dataframe(inv_styled, use_container_width=True, hide_index=True)

            fig = go.Figure()
            fig.add_trace(go.Bar(name="Target XIRR", x=investor_summary_df["Investor"], y=investor_summary_df["Target_XIRR"], marker_color="#4CAF50"))
            fig.add_trace(go.Bar(name="Actual XIRR", x=investor_summary_df["Investor"], y=investor_summary_df["Actual_XIRR"], marker_color="#2196F3"))
            fig.update_layout(barmode="group", title="Target vs Actual XIRR by Investor", yaxis_title="XIRR (%)", height=350)
            st.plotly_chart(fig, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                fig2 = px.pie(investor_summary_df, values="Total_Funded", names="Investor", title="Capital Allocation", hole=0.4)
                fig2.update_layout(height=350)
                st.plotly_chart(fig2, use_container_width=True)
            with c2:
                fig3 = px.pie(investor_summary_df, values="Outstanding_Balance", names="Investor", title="Outstanding Exposure", hole=0.4)
                fig3.update_layout(height=350)
                st.plotly_chart(fig3, use_container_width=True)

        st.markdown("##### Detailed Investor Ledger")
        inv_filter = st.multiselect("Filter Investors", investor_ledger_df["Investor"].unique().tolist(),
                                    default=investor_ledger_df["Investor"].unique().tolist())
        filtered_ledger = investor_ledger_df[investor_ledger_df["Investor"].isin(inv_filter)]
        money_cols_inv = ["Capital_Called", "Interest_Earned", "Principal_Returned", "Ending_Balance",
                          "Cumulative_Funded", "Cumulative_Interest"]
        fmt_inv = {c: "${:,.2f}" for c in money_cols_inv}
        fmt_inv["Month"] = "{:.0f}"
        st.dataframe(filtered_ledger.style.format(fmt_inv), use_container_width=True, height=400, hide_index=True)

        fig4 = go.Figure()
        for inv in inv_filter:
            inv_data = filtered_ledger[filtered_ledger["Investor"] == inv]
            fig4.add_trace(go.Scatter(x=inv_data["Month"], y=inv_data["Ending_Balance"], mode="lines+markers", name=inv))
        fig4.update_layout(title="Investor Outstanding Balances Over Time", xaxis_title="Month", yaxis_title="$", yaxis_tickprefix="$", height=350)
        st.plotly_chart(fig4, use_container_width=True)

    # --- TAB: P&L ---
    with tabs[2]:
        st.subheader("📈 Platform P&L")

        pnl_items = []
        for k, v in pnl_dict.items():
            if isinstance(v, str): pnl_items.append({"Item": k, "Amount": v})
            elif "%" in k or "bps" in k: pnl_items.append({"Item": k, "Amount": f"{v:.2f}"})
            else: pnl_items.append({"Item": k, "Amount": fmt(v)})

        pnl_df = pd.DataFrame(pnl_items)

        def highlight_pnl(row):
            styles = ["", ""]
            item = str(row["Item"])
            if "Total Income" in item: styles = ["background-color: #d4edda; font-weight:bold"] * 2
            elif "Net Revenue" in item: styles = ["background-color: #cce5ff; font-weight:bold"] * 2
            elif "Expense" in item: styles = ["background-color: #f8d7da"] * 2
            return styles

        st.dataframe(pnl_df.style.apply(highlight_pnl, axis=1), use_container_width=True, hide_index=True, height=550)

        st.markdown("##### Income vs Expense Breakdown")
        income_items = {
            "Borrower Interest": float(schedule_df["Interest_Due"].sum()),
            "Penalty Interest": float(schedule_df["Penalty_Interest"].sum()),
            "Establishment Fee": fees["establishment"],
            "Legal Fee": fees["legal_fee"],
            "Line Fees": float(schedule_df["Line_Fee"].sum()),
            "Admin Fees": float(schedule_df["Admin_Fee"].sum()),
            "Other Fees": fees["risk_review"] + fees["discharge"] + fees["other_upfront"] + fees["restructure"],
        }
        total_income = sum(income_items.values())
        total_inv_exp = investor_summary_df["Total_Interest_Earned"].sum() if len(investor_summary_df) > 0 else 0

        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(income_items.keys()), y=list(income_items.values()), marker_color="#4CAF50", name="Income"))
        fig.add_trace(go.Bar(x=["Investor Expense"], y=[total_inv_exp], marker_color="#f44336", name="Expense"))
        fig.add_trace(go.Bar(x=["Net Revenue"], y=[total_income - total_inv_exp], marker_color="#2196F3", name="Net"))
        fig.update_layout(title="P&L Breakdown", yaxis_tickprefix="$", yaxis_tickformat=",.", height=400)
        st.plotly_chart(fig, use_container_width=True)

    # --- TAB: Cashflows ---
    with tabs[3]:
        st.subheader("💸 SPV Cashflow Statement")
        cf_money = [c for c in cashflow_df.columns if c != "Month"]
        cf_fmt = {c: "${:,.2f}" for c in cf_money}
        cf_fmt["Month"] = "{:.0f}"

        def highlight_cf(row):
            styles = [""] * len(row)
            try:
                net_idx = cashflow_df.columns.get_loc("Net_SPV_Cashflow")
                if float(row.iloc[net_idx]) < 0: styles[net_idx] = "background-color: #ffe0e0; font-weight:bold"
                else: styles[net_idx] = "background-color: #d4edda"
            except: pass
            return styles

        st.dataframe(cashflow_df.style.apply(highlight_cf, axis=1).format(cf_fmt),
                     use_container_width=True, height=450, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=cashflow_df["Month"], y=cashflow_df["Total_Inflow"], name="Inflows", marker_color="#4CAF50"))
            fig.add_trace(go.Bar(x=cashflow_df["Month"], y=-cashflow_df["Total_Outflow"], name="Outflows", marker_color="#f44336"))
            fig.update_layout(barmode="relative", title="Monthly SPV Inflows & Outflows",
                            xaxis_title="Month", yaxis_title="$", yaxis_tickprefix="$", height=350)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=cashflow_df["Month"], y=cashflow_df["Cumulative_SPV_Cash"],
                                      mode="lines+markers", fill="tozeroy", line=dict(color="#2196F3"), name="Cum Cash"))
            fig2.update_layout(title="Cumulative SPV Cash Position", xaxis_title="Month",
                             yaxis_title="$", yaxis_tickprefix="$", height=350)
            st.plotly_chart(fig2, use_container_width=True)

    # --- TAB: Economics & Settlement ---
    with tabs[4]:
        st.subheader("📊 Economics & Settlement Summary")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("##### Deal Economics")
            econ = {
                "Facility Amount": fmt(fa),
                "Property Value": fmt(pv),
                "Day 1 LVR": pct(fa / pv * 100) if pv > 0 else "N/A",
                "Term": f"{term} months",
                "Construction Period": f"{constr} months",
                "Borrower IRR": pct(borrower_irr),
                "Monthly Rate": pct(to_monthly(borrower_irr / 100) * 100),
            }
            for k, v in econ.items(): st.markdown(f"**{k}:** {v}")

        with c2:
            st.markdown("##### Settlement Waterfall")
            dd1 = float(dd_schedule.get("1", 0))
            upfront_total = fees["upfront_total"]
            net_to_borrower = dd1 - upfront_total
            settle_items = {
                "Gross DD1": fmt(dd1),
                "Less: Establishment Fee": fmt(-fees["establishment"]),
                "Less: Legal Fee": fmt(-fees["legal_fee"]),
                "Less: Risk Review": fmt(-fees["risk_review"]),
                "Less: Other Upfront": fmt(-fees["other_upfront"]),
                "═══════════════": "══════════",
                "Net to Borrower": fmt(net_to_borrower),
            }
            for k, v in settle_items.items(): st.markdown(f"**{k}:** {v}")

        with c3:
            st.markdown("##### Investor Economics")
            borrower_irr_pct = float(borrower_irr)
            total_funded = investor_summary_df["Total_Funded"].sum() if len(investor_summary_df) > 0 else 0
            wtd_inv_rate = 0
            if total_funded > 0:
                for _, r in investor_summary_df.iterrows():
                    wtd_inv_rate += (r["Total_Funded"] / total_funded) * r["Target_XIRR"]
            spread_bps = (borrower_irr_pct - wtd_inv_rate) * 100

            inv_econ = {
                "Borrower Rate": pct(borrower_irr_pct),
                "Wtd Investor Cost": pct(wtd_inv_rate),
                "Platform Spread": f"{spread_bps:.0f} bps",
                "Total Investor Capital": fmt(total_funded),
                "Total Investor Interest": fmt(investor_summary_df["Total_Interest_Earned"].sum() if len(investor_summary_df) > 0 else 0),
            }
            for k, v in inv_econ.items(): st.markdown(f"**{k}:** {v}")

        st.markdown("---")
        st.markdown("##### Fund NAV Allocation (DD1)")
        nav_c1, nav_c2, nav_c3 = st.columns(3)
        dd1_net = float(schedule_df[schedule_df["Month"] == 1]["Drawdown_Net"].sum()) if len(schedule_df) > 0 else 0
        nav_c1.metric("BPCFMC (50%)", fmt(dd1_net * 0.50))
        nav_c2.metric("BPCFLC (15%)", fmt(dd1_net * 0.15))
        nav_c3.metric("CBCF (35%)", fmt(dd1_net * 0.35))

    # --- TAB: Fees ---
    with tabs[5]:
        st.subheader("💵 Fee Summary")
        fee_items = {
            "Establishment Fee": fees["establishment"],
            "Legal Fee": fees["legal_fee"],
            "Risk Review Fee": fees["risk_review"],
            "Other Upfront Fees": fees["other_upfront"],
            "Upfront Total (Deducted at DD1)": fees["upfront_total"],
            "": None,
            "Line Fee (Construction Period Total)": fees["line_fee_total"],
            "Admin Fees (Total)": fees["admin_total"],
            "Discharge Fee": fees["discharge"],
            "Restructure Fee": fees["restructure"],
        }
        fee_rows = []
        for k, v in fee_items.items():
            if v is None: fee_rows.append({"Fee Type": "───────────────", "Amount": "────────"})
            else: fee_rows.append({"Fee Type": k, "Amount": fmt(v)})
        st.table(pd.DataFrame(fee_rows))

        total_all_fees = fees["upfront_total"] + fees["line_fee_total"] + fees["admin_total"] + fees["discharge"] + fees["restructure"]
        st.metric("Total All Fees", fmt(total_all_fees))

    # --- TAB: Log Actuals ---
    with tabs[6]:
        st.subheader("✏️ Log Monthly Actuals")
        st.info("Enter actual payments received. The model will calculate arrears where actuals fall short of scheduled amounts.")
        with st.form("log_actuals"):
            ac1, ac2 = st.columns(2)
            act_month = ac1.number_input("Month #", 1, term, 1)
            sched_row = schedule_df[schedule_df["Month"] == act_month]
            if len(sched_row) > 0:
                sr = sched_row.iloc[0]
                ac2.markdown(f"**Scheduled Due:** {fmt(sr['Total_Due'])}")
            ac3, ac4, ac5, ac6 = st.columns(4)
            a_dd = ac3.number_input("Actual Drawdown ($)", 0.0, fa, 0.0, 1000.0)
            a_int = ac4.number_input("Actual Interest Paid ($)", 0.0, 1e8, 0.0, 100.0)
            a_prin = ac5.number_input("Actual Principal Paid ($)", 0.0, 1e8, 0.0, 100.0)
            a_fees = ac6.number_input("Actual Fees Paid ($)", 0.0, 1e8, 0.0, 100.0)

            if st.form_submit_button("💾 Save Actual", type="primary"):
                save_actual(deal_id, act_month, a_dd, a_int, a_prin, a_fees)
                st.success(f"Actuals saved for Month {act_month}")
                st.rerun()

        if len(actuals_df) > 0:
            st.markdown("##### Logged Actuals")
            st.dataframe(actuals_df[["month", "actual_drawdown", "actual_interest", "actual_principal", "actual_fees"]],
                        use_container_width=True, hide_index=True)

    # --- TAB: Export ---
    with tabs[7]:
        st.subheader("💾 Export to Excel")
        if st.button("📥 Generate Excel Export", type="primary", use_container_width=True):
            buf = BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                tls_export = schedule_df[["Month", "Drawdown_Gross", "Upfront_Fees_Deducted", "Drawdown_Net",
                                          "Opening_Balance", "Balance_After_Draw", "Interest_Due", "Penalty_Interest",
                                          "Line_Fee", "Admin_Fee", "Scheduled_Principal", "Prepayment",
                                          "Total_Principal", "Total_Due", "Arrears", "Ending_Balance",
                                          "Property_Value", "LVR_pct", "Covenant_Breach"]].copy()
                tls_export.to_excel(writer, sheet_name="TLS_Ledger", index=False)
                investor_ledger_df.to_excel(writer, sheet_name="Investor_Ledger", index=False)
                investor_summary_df.to_excel(writer, sheet_name="Investor_Summary", index=False)
                pd.DataFrame(pnl_items).to_excel(writer, sheet_name="PnL", index=False)
                cashflow_df.to_excel(writer, sheet_name="Cashflows", index=False)
                pd.DataFrame(fee_rows).to_excel(writer, sheet_name="Fees", index=False)
            buf.seek(0)
            st.download_button("⬇️ Download Excel", data=buf,
                              file_name=f"Deal_{deal_id}_Model_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                              mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.success("Export ready!")


# ═══════════════════════════════════════════════════════════════
# NAVIGATION
# ═══════════════════════════════════════════════════════════════
PAGES = {
    "📊 Dashboard": page_dashboard,
    "🏢 Borrowers": page_borrowers,
    "📄 Deals Directory": page_deals,
    "🧮 Deal Modeler & LMS": page_modeler,
}

with st.sidebar:
    st.image("https://img.icons8.com/3d-fluency/94/bank-building.png", width=60)
    st.title("Private Credit Platform")
    st.markdown("---")
    selection = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
    st.markdown("---")
    st.caption("v3.0 — Investor Base + P&L + Cashflows")

try:
    PAGES[selection]()
except Exception as e:
    st.error(f"Error: {e}")
    st.exception(e)
