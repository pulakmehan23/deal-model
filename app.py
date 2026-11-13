import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import brentq
import io
import sqlite3
from datetime import datetime

# ═══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Structured Lending Platform",
    layout="wide",
    page_icon="🏦"
)

st.markdown("""
<style>
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.05rem; font-weight: 600;
    }
    .block-container {padding-top: 1.5rem;}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  DATABASE
# ═══════════════════════════════════════════════════════════════
DB_FILE = "lms_data.db"

def get_conn():
    return sqlite3.connect(DB_FILE, check_same_thread=False)

def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS borrowers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT, abn TEXT, group_name TEXT,
        risk_grade TEXT, industry TEXT, status TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS deals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        borrower_id INTEGER, deal_name TEXT,
        facility_amount REAL, deal_status TEXT,
        settlement_date TEXT, maturity_date TEXT,
        facility_type TEXT, security_type TEXT,
        property_value REAL, params_json TEXT,
        FOREIGN KEY(borrower_id) REFERENCES borrowers(id)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS actuals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        deal_id INTEGER, month INTEGER,
        actual_drawdown REAL, actual_interest REAL,
        actual_principal REAL, actual_fees REAL,
        FOREIGN KEY(deal_id) REFERENCES deals(id)
    )''')
    conn.commit()
    conn.close()

def run_query(query, params=()):
    conn = get_conn()
    try:
        df = pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()
    return df

def execute_query(query, params=()):
    conn = get_conn()
    try:
        c = conn.cursor()
        c.execute(query, params)
        conn.commit()
    finally:
        conn.close()

try:
    init_db()
except Exception as e:
    st.error(f"Database init error: {e}")

# ═══════════════════════════════════════════════════════════════
#  FINANCIAL ENGINE
# ═══════════════════════════════════════════════════════════════
def to_monthly(r): return (1 + r) ** (1 / 12) - 1
def to_annual(r): return (1 + r) ** 12 - 1
def cpr_to_smm(cpr): return 1 - (1 - cpr) ** (1 / 12) if cpr > 0 else 0.0

def calc_irr(cfs):
    if len(cfs) < 2 or all(c == 0 for c in cfs[1:]):
        return np.nan
    try:
        def npv(r):
            return sum(cf / (1 + r) ** t for t, cf in enumerate(cfs))
        return to_annual(brentq(npv, -0.5, 5.0, xtol=1e-10))
    except Exception:
        return np.nan

def compute_fees(facility, outstanding, month, fee_config,
                 is_default=False, is_restructure=False,
                 restructure_amount=0, is_exit=False, construction_end=0):
    fees = {}
    if month == 0:
        fees["Establishment"] = facility * fee_config.get("establishment_pct", 0)

    undrawn = max(facility - outstanding, 0)
    if 0 < month <= construction_end:
        fees["Line Fee (Undrawn)"] = undrawn * fee_config.get("line_fee_pct", 0) / 12

    if is_default:
        fees["Risk Review"] = fee_config.get("risk_review_amount", 0)
    if is_restructure:
        fees["Restructure"] = restructure_amount * fee_config.get("restructure_pct", 0)

    if month > 0:
        if fee_config.get("admin_type", "fixed") == "fixed":
            fees["Admin"] = fee_config.get("admin_amount", 0)
        else:
            fees["Admin"] = outstanding * fee_config.get("admin_pct", 0) / 12

    if is_exit:
        fees["Discharge"] = fee_config.get("discharge_amount", 0)
    return fees

def build_schedule(
    facility, target_irr, n_months,
    interest_timing="arrears", advance_n=1, capitalise_months=0,
    bbsy_curve=None, margin=0.0, reset_freq=12,
    drawdown_events=None, cpr=0.0, fee_config=None,
    default_month=None, restructure_month=None, restructure_amount=0,
    construction_end=0, property_value=0, property_growth=0,
    actuals_dict=None, penalty_rate_annual=0.0, max_lvr_covenant=100.0
):
    r_m = to_monthly(target_irr)
    smm = cpr_to_smm(cpr)
    if drawdown_events is None:
        drawdown_events = [(0, facility)]
    if fee_config is None:
        fee_config = {}

    total_commitment = sum(a for _, a in drawdown_events)
    dd_idx = 0
    balance = 0.0
    cum_principal = 0.0
    arrears_balance = 0.0
    rows = []

    def get_rate(m):
        if bbsy_curve and len(bbsy_curve) > 0:
            p = min(m // reset_freq, len(bbsy_curve) - 1)
            return to_monthly(bbsy_curve[p] + margin)
        return r_m

    for m in range(n_months + 1):
        is_actual = actuals_dict is not None and m in actuals_dict
        act = actuals_dict.get(m, {}) if is_actual else {}

        # Drawdowns (Gross)
        if is_actual and "drawdown" in act:
            dd_amt = float(act["drawdown"] or 0)
        else:
            dd_amt = 0.0
            while dd_idx < len(drawdown_events) and drawdown_events[dd_idx][0] == m:
                dd_amt += drawdown_events[dd_idx][1]
                dd_idx += 1
        balance += dd_amt

        is_def = default_month is not None and m == default_month
        is_res = restructure_month is not None and m == restructure_month
        is_exit = (m == n_months) or (balance <= 0 and m > 0)

        month_fees = compute_fees(
            facility, balance, m, fee_config,
            is_def, is_res, restructure_amount, is_exit, construction_end
        )

        if arrears_balance > 0 and penalty_rate_annual > 0:
            month_fees["Penalty Interest"] = arrears_balance * (penalty_rate_annual / 12)

        if is_actual and "fees" in act:
            total_fees = float(act["fees"] or 0)
        else:
            total_fees = sum(month_fees.values())

        prop_val = property_value * ((1 + property_growth / 12) ** m) if property_value > 0 else 0
        current_limit = max(facility - cum_principal, 0)

        # ── SETTLEMENT / MONTH 0 (NET DISBURSEMENT LOGIC) ──
        if m == 0:
            # Net Cash Disbursed = Gross Drawdown minus Upfront Fees
            net_cash_disbursed = dd_amt - total_fees
            lvr0 = (balance / prop_val * 100) if prop_val > 0 else 0
            rows.append({
                "Month": m, "Mode": "Actual" if is_actual else "Forecast",
                "Gross Drawdown": dd_amt, 
                "Fees Deducted": total_fees,
                "Net Cash Disbursed": net_cash_disbursed,
                "Opening": balance, "Rate": get_rate(m),
                "Interest": 0.0, "Interest Paid": 0.0, "Interest Capitalised": 0.0,
                "Principal": 0.0, "Prepayment": 0.0, "Fees": total_fees,
                "Fee Detail": str(month_fees), "Total Payment": -net_cash_disbursed,
                "Closing": balance, "Arrears": 0.0, "Current Limit": current_limit,
                "LVR": lvr0, "Covenant Breach": lvr0 > max_lvr_covenant,
                "Borrower CF": net_cash_disbursed, "Anniversary": False
            })
            continue

        rate_m = get_rate(m - 1)
        interest = balance * rate_m
        is_cap = m <= capitalise_months
        cap_amt = interest if is_cap else 0.0
        expected_cash_int = 0.0 if is_cap else interest
        if is_cap:
            balance += cap_amt

        remaining = n_months - m + 1
        if remaining > 0 and m > capitalise_months:
            expected_princ = (total_commitment - cum_principal) / max(remaining, 1)
        else:
            expected_princ = 0.0

        expected_prepay = 0.0
        if smm > 0 and m > capitalise_months:
            expected_prepay = max(balance - expected_princ, 0) * smm

        if is_actual:
            cash_int = float(act.get("interest", 0) or 0)
            total_princ = float(act.get("principal", 0) or 0)
            prepay = 0.0
            expected_total = expected_cash_int + expected_princ + expected_prepay + sum(month_fees.values())
            actual_total = cash_int + total_princ + total_fees
            shortfall = max(expected_total - actual_total, 0)
            arrears_balance += shortfall
        else:
            cash_int = expected_cash_int
            total_princ = min(expected_princ + expected_prepay, balance)
            prepay = expected_prepay
            if arrears_balance > 0:
                paydown = min(arrears_balance, total_princ * 0.1)
                arrears_balance = max(arrears_balance - paydown, 0)

        balance -= total_princ
        cum_principal += total_princ
        
        # Net Cash Flow for Month m (Drawdown minus payments/fees)
        total_payment = cash_int + total_princ + total_fees
        net_cash_disbursed = max(dd_amt - total_payment, 0)
        borrower_net_cf = dd_amt - total_payment
        
        closing = max(balance, 0)
        current_limit = max(facility - cum_principal, 0)
        lvr = (closing / prop_val * 100) if prop_val > 0 else 0

        rows.append({
            "Month": m, "Mode": "Actual" if is_actual else "Forecast",
            "Gross Drawdown": dd_amt,
            "Fees Deducted": total_fees,
            "Net Cash Disbursed": net_cash_disbursed,
            "Opening": closing + total_princ, "Rate": rate_m,
            "Interest": interest, "Interest Paid": cash_int,
            "Interest Capitalised": cap_amt,
            "Principal": total_princ if is_actual else expected_princ,
            "Prepayment": prepay, "Fees": total_fees,
            "Fee Detail": str(month_fees), "Total Payment": total_payment,
            "Closing": closing, "Arrears": arrears_balance,
            "Current Limit": current_limit, "LVR": lvr,
            "Covenant Breach": lvr > max_lvr_covenant,
            "Borrower CF": borrower_net_cf, "Anniversary": (m % 12 == 0)
        })

    return pd.DataFrame(rows)

# ═══════════════════════════════════════════════════════════════
#  NAV
# ═══════════════════════════════════════════════════════════════
st.sidebar.title("🏦 Admin Platform")
nav_page = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "🏢 Borrowers", "📄 Deals Directory", "🧮 Deal Modeler & LMS"]
)

# ═══════════════════════════════════════════════════════════════
#  PAGE 1: DASHBOARD
# ═══════════════════════════════════════════════════════════════
if nav_page == "📊 Dashboard":
    st.title("📊 Portfolio Dashboard")
    try:
        deals_df = run_query("SELECT * FROM deals")
        borrowers_df = run_query("SELECT * FROM borrowers")

        active = deals_df[deals_df["deal_status"] == "Active"] if not deals_df.empty else pd.DataFrame()
        pipeline = deals_df[deals_df["deal_status"] == "Pipeline"] if not deals_df.empty else pd.DataFrame()
        total_aum = active["facility_amount"].sum() if not active.empty else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Active AUM", f"${total_aum:,.0f}")
        c2.metric("Active Deals", len(active))
        c3.metric("Pipeline Deals", len(pipeline))
        c4.metric("Borrowers", len(borrowers_df))

        st.markdown("---")
        if deals_df.empty:
            st.info("No deals yet. Add a Borrower, then create a Deal, then open the Modeler.")
        else:
            q = '''
            SELECT d.id as ID, b.name as Borrower, d.deal_name as Deal,
                   d.facility_amount as Facility, d.deal_status as Status,
                   d.facility_type as Type, d.security_type as Security
            FROM deals d JOIN borrowers b ON d.borrower_id = b.id
            ORDER BY d.id DESC
            '''
            disp = run_query(q)
            disp["Facility"] = disp["Facility"].apply(lambda x: f"${x:,.0f}")
            st.dataframe(disp, hide_index=True, use_container_width=True)
    except Exception as e:
        st.error(f"Dashboard error: {e}")

# ═══════════════════════════════════════════════════════════════
#  PAGE 2: BORROWERS
# ═══════════════════════════════════════════════════════════════
elif nav_page == "🏢 Borrowers":
    st.title("🏢 Borrower Management")
    try:
        with st.expander("➕ Add New Borrower", expanded=True):
            with st.form("new_borrower_form"):
                c1, c2 = st.columns(2)
                b_name = c1.text_input("Borrower Entity Name *")
                b_abn = c2.text_input("ABN / ACN")
                b_group = c1.text_input("Group / Sponsor Name")
                b_ind = c2.selectbox("Industry", ["Residential Dev", "Commercial CRE", "Corporate", "Other"])
                b_risk = c1.selectbox("Risk Grade", ["A", "B", "C", "D", "Watchlist"])
                if st.form_submit_button("Save Borrower"):
                    if b_name.strip():
                        execute_query(
                            "INSERT INTO borrowers (name, abn, group_name, risk_grade, industry, status) VALUES (?,?,?,?,?,?)",
                            (b_name.strip(), b_abn, b_group, b_risk, b_ind, "Active")
                        )
                        st.success(f"Added {b_name}")
                        st.rerun()
                    else:
                        st.warning("Name is required.")

        st.markdown("### Directory")
        b_df = run_query(
            "SELECT id as ID, name as Entity, abn as ABN, group_name as Sponsor, "
            "risk_grade as Grade, industry as Industry, status as Status FROM borrowers ORDER BY id DESC"
        )
        if b_df.empty:
            st.info("No borrowers yet.")
        else:
            st.dataframe(b_df, hide_index=True, use_container_width=True)
    except Exception as e:
        st.error(f"Borrowers error: {e}")

# ═══════════════════════════════════════════════════════════════
#  PAGE 3: DEALS DIRECTORY
# ═══════════════════════════════════════════════════════════════
elif nav_page == "📄 Deals Directory":
    st.title("📄 Deals Directory")
    try:
        borrowers_df = run_query("SELECT id, name FROM borrowers")
        if borrowers_df.empty:
            st.warning("Create a Borrower first.")
        else:
            with st.expander("➕ Create New Deal", expanded=True):
                with st.form("new_deal_form"):
                    b_dict = dict(zip(borrowers_df["name"], borrowers_df["id"]))
                    c1, c2, c3 = st.columns(3)
                    d_borrower = c1.selectbox("Borrower", list(b_dict.keys()))
                    d_name = c2.text_input("Deal / Project Name *")
                    d_status = c3.selectbox("Status", ["Pipeline", "Approved", "Active", "Matured", "Defaulted"])

                    c4, c5, c6 = st.columns(3)
                    d_fac = c4.number_input("Facility ($)", min_value=100000.0, value=10000000.0, step=500000.0)
                    d_type = c5.selectbox("Facility Type", ["Construction", "Bridging", "Term Loan", "Land"])
                    d_sec = c6.selectbox("Security", ["1st Mortgage", "2nd Mortgage", "Unsecured"])

                    c7, c8, c9 = st.columns(3)
                    d_set = c7.date_input("Settlement Date")
                    d_mat = c8.date_input("Maturity Date")
                    d_prop = c9.number_input("Property Value ($)", min_value=0.0, value=15000000.0, step=500000.0)

                    if st.form_submit_button("Create Deal"):
                        if d_name.strip():
                            execute_query(
                                '''INSERT INTO deals
                                   (borrower_id, deal_name, facility_amount, deal_status,
                                    settlement_date, maturity_date, facility_type, security_type, property_value)
                                   VALUES (?,?,?,?,?,?,?,?,?)''',
                                (b_dict[d_borrower], d_name.strip(), d_fac, d_status,
                                 str(d_set), str(d_mat), d_type, d_sec, d_prop)
                            )
                            st.success(f"Deal '{d_name}' created. Open it in Deal Modeler.")
                            st.rerun()
                        else:
                            st.warning("Deal name is required.")

            q = '''
            SELECT d.id as ID, b.name as Borrower, d.deal_name as Deal,
                   d.facility_amount as Facility, d.deal_status as Status,
                   d.facility_type as Type, d.security_type as Security,
                   d.settlement_date as Settlement, d.maturity_date as Maturity
            FROM deals d JOIN borrowers b ON d.borrower_id = b.id
            ORDER BY d.id DESC
            '''
            d_list = run_query(q)
            if d_list.empty:
                st.info("No deals yet.")
            else:
                show = d_list.copy()
                show["Facility"] = show["Facility"].apply(lambda x: f"${x:,.0f}")
                st.dataframe(show, hide_index=True, use_container_width=True)
                st.caption("Tip: Go to Deal Modeler & LMS and select a deal from the dropdown.")
    except Exception as e:
        st.error(f"Deals error: {e}")

# ═══════════════════════════════════════════════════════════════
#  PAGE 4: DEAL MODELER & LMS
# ═══════════════════════════════════════════════════════════════
elif nav_page == "🧮 Deal Modeler & LMS":
    st.title("🧮 Deal Modeler & LMS")

    try:
        db_deals = run_query(
            "SELECT d.id, d.deal_name, b.name, d.facility_amount, d.property_value "
            "FROM deals d JOIN borrowers b ON d.borrower_id = b.id ORDER BY d.id DESC"
        )
        deal_options = ["🛠️ Sandbox / Scratchpad"] + [
            f"ID {r['id']}: {r['deal_name']} ({r['name']})" for _, r in db_deals.iterrows()
        ]
        selected_deal = st.selectbox("📂 Load Deal or use Sandbox", deal_options)

        selected_deal_id = None
        default_facility = 10_000_000.0
        default_prop = 15_000_000.0
        if selected_deal != "🛠️ Sandbox / Scratchpad":
            selected_deal_id = int(selected_deal.split(":")[0].replace("ID ", ""))
            row = db_deals[db_deals["id"] == selected_deal_id].iloc[0]
            default_facility = float(row["facility_amount"] or 10_000_000)
            default_prop = float(row["property_value"] or 15_000_000)
            st.success(f"Loaded: **{selected_deal}**")

        actuals_dict = {}
        if selected_deal_id:
            act_df = run_query("SELECT * FROM actuals WHERE deal_id = ?", (selected_deal_id,))
            for _, r in act_df.iterrows():
                actuals_dict[int(r["month"])] = {
                    "drawdown": r["actual_drawdown"],
                    "interest": r["actual_interest"],
                    "principal": r["actual_principal"],
                    "fees": r["actual_fees"],
                }

        # ── CENTER INPUTS ─────────────────────────────────────
        with st.expander("⚙️ DEAL PARAMETERS & ASSUMPTIONS", expanded=True):
            tab_loan, tab_draw, tab_fees, tab_risk = st.tabs([
                "📝 Core Loan Terms",
                "🏗️ Construction & Drawdowns",
                "💵 Fees & Interest",
                "⚠️ Risk & Covenants"
            ])

            with tab_loan:
                c1, c2, c3, c4 = st.columns(4)
                facility = c1.number_input("Facility Limit ($)", 100_000.0, 500_000_000.0, default_facility, step=100_000.0)
                target_irr = c2.number_input("Target IRR (%)", 1.0, 35.0, 12.0, 0.25) / 100
                tenure = int(c3.number_input("Tenure (months)", 3, 120, 24))
                capitalise_months = int(c4.number_input("PIK Period (months)", 0, tenure, 0))

                c1, c2, c3, c4 = st.columns(4)
                property_value = c1.number_input("Initial Prop Value ($)", 0.0, 1_000_000_000.0, default_prop, step=100_000.0)
                property_growth = c2.number_input("Prop Growth (% p.a.)", -10.0, 20.0, 3.0, 0.5) / 100
                max_lvr_covenant = c3.number_input("Max LVR Covenant (%)", 10.0, 100.0, 75.0, 1.0)
                init_lvr = (facility / property_value * 100) if property_value > 0 else 0
                c4.metric("Starting LVR", f"{init_lvr:.1f}%")

            with tab_draw:
                c1, c2 = st.columns([1, 2])
                construction_end = int(c1.number_input("Construction End (Month)", 0, tenure, 0))
                n_dd = int(c1.number_input("Number of Drawdowns", 1, 10, 1))
                dd_events = []
                rem = facility
                c2.markdown("**Drawdown Schedule**")
                cols = c2.columns(min(n_dd, 4))
                for i in range(n_dd):
                    col = cols[i % len(cols)]
                    dd_m = int(col.number_input(f"DD{i+1} Month", 0, max(tenure - 1, 0), min(i * max(tenure // n_dd, 1), max(tenure - 1, 0)), key=f"ddm{i}"))
                    if i < n_dd - 1:
                        dd_a = float(col.number_input(f"DD{i+1} $", 0.0, float(rem), float(rem / (n_dd - i)), step=100_000.0, key=f"dda{i}"))
                    else:
                        dd_a = float(rem)
                        col.info(f"Final DD: ${dd_a:,.0f}")
                    dd_events.append((dd_m, dd_a))
                    rem -= dd_a

            with tab_fees:
                c1, c2, c3, c4 = st.columns(4)
                est_pct = c1.number_input("Establishment (%)", 0.0, 5.0, 1.5, 0.25) / 100
                line_pct = c2.number_input("Line Fee % p.a. (on undrawn)", 0.0, 5.0, 1.0, 0.25) / 100
                admin_amount = c3.number_input("Admin Fee ($/m)", 0.0, 50_000.0, 1000.0, step=100.0)
                discharge_amt = c4.number_input("Discharge Fee ($)", 0.0, 100_000.0, 2000.0, step=500.0)

                c1, c2, c3, c4 = st.columns(4)
                interest_timing = c1.selectbox(
                    "Interest Timing",
                    ["arrears", "advance", "all_advance"],
                    format_func=lambda x: x.replace("_", " ").title()
                )
                penalty_rate_annual = c2.number_input("Penalty Rate (% p.a.)", 0.0, 30.0, 18.0, 1.0) / 100
                risk_review_amt = c3.number_input("Risk Review Fee ($)", 0.0, 100_000.0, 5000.0, step=500.0)
                restructure_pct = c4.number_input("Restructure Fee (%)", 0.0, 5.0, 1.0, 0.25) / 100

                fee_config = {
                    "establishment_pct": est_pct,
                    "line_fee_pct": line_pct,
                    "admin_type": "fixed",
                    "admin_amount": admin_amount,
                    "admin_pct": 0.0,
                    "discharge_amount": discharge_amt,
                    "risk_review_amount": risk_review_amt,
                    "restructure_pct": restructure_pct,
                }

            with tab_risk:
                c1, c2, c3 = st.columns(3)
                cpr = c1.number_input("CPR Prepayment (% p.a.)", 0.0, 50.0, 0.0, 1.0) / 100
                pd_annual = c2.number_input("PD (% p.a.)", 0.0, 20.0, 2.0, 0.5) / 100
                lgd = c3.number_input("LGD (%)", 0.0, 100.0, 40.0, 5.0) / 100
                c1, c2 = st.columns(2)
                d_month_in = int(c1.number_input("Simulate Default Month (0=off)", 0, tenure, 0))
                r_month_in = int(c2.number_input("Simulate Restructure Month (0=off)", 0, tenure, 0))
                default_month = d_month_in if d_month_in > 0 else None
                restructure_month = r_month_in if r_month_in > 0 else None

        # ── RUN MODEL ─────────────────────────────────────────
        sched = build_schedule(
            facility=facility,
            target_irr=target_irr,
            n_months=tenure,
            interest_timing=interest_timing,
            capitalise_months=capitalise_months,
            drawdown_events=dd_events,
            cpr=cpr,
            fee_config=fee_config,
            default_month=default_month,
            restructure_month=restructure_month,
            restructure_amount=facility / 2,
            construction_end=construction_end,
            property_value=property_value,
            property_growth=property_growth,
            actuals_dict=actuals_dict,
            penalty_rate_annual=penalty_rate_annual,
            max_lvr_covenant=max_lvr_covenant,
        )

        borrower_irr = calc_irr(list(sched["Borrower CF"]))
        total_interest = float(sched["Interest"].sum())
        total_fees = float(sched["Fees"].sum())

        # ── OUTPUTS ───────────────────────────────────────────
        st.markdown("### 📊 Outputs")
        t1, t2, t3, t4, t5 = st.tabs([
            "📋 TLS Ledger", "📈 Economics & Settlement", "💵 Fees", "✏️ Log Actuals", "💾 Export"
        ])

        with t1:
            st.caption("Green Mode = Actual · Red LVR = Covenant breach · Orange Arrears = unpaid")
            cols_show = [
                "Month", "Mode", "Opening", "Gross Drawdown", "Fees Deducted", "Net Cash Disbursed",
                "Interest Paid", "Principal", "Arrears", "Closing", "Current Limit", "LVR", "Covenant Breach"
            ]
            view = sched[cols_show].copy()

            def style_row(row):
                styles = [""] * len(row)
                try:
                    if bool(row["Covenant Breach"]):
                        styles[list(view.columns).index("LVR")] = "background-color:#ffcccc;color:#990000;font-weight:700"
                    if float(row["Arrears"]) > 0:
                        styles[list(view.columns).index("Arrears")] = "background-color:#ffe6cc;color:#cc0000;font-weight:700"
                    if row["Mode"] == "Actual":
                        styles[list(view.columns).index("Mode")] = "background-color:#e6f9e6;color:#006600;font-weight:700"
                except Exception:
                    pass
                return styles

            styler = (
                view.style
                .apply(style_row, axis=1)
                .format({
                    "Opening": "${:,.0f}",
                    "Gross Drawdown": "${:,.0f}",
                    "Fees Deducted": "${:,.0f}",
                    "Net Cash Disbursed": "${:,.0f}",
                    "Interest Paid": "${:,.0f}",
                    "Principal": "${:,.0f}",
                    "Arrears": "${:,.0f}",
                    "Closing": "${:,.0f}",
                    "Current Limit": "${:,.0f}",
                    "LVR": "{:.1f}%",
                })
            )
            st.dataframe(styler, use_container_width=True, height=480)

        with t2:
            st.markdown("#### 📄 Settlement Statement (Month 0 Waterfall)")
            m0_row = sched.iloc[0]
            gross_dd0 = m0_row["Gross Drawdown"]
            fees_deducted0 = m0_row["Fees Deducted"]
            net_disbursed0 = m0_row["Net Cash Disbursed"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Gross Facility / Initial Drawdown", f"${gross_dd0:,.0f}")
            c2.metric("Upfront Fees Deducted at Settlement", f"-${fees_deducted0:,.0f}")
            c3.metric("Net Cash Disbursed to Borrower", f"${net_disbursed0:,.0f}")
            c4.metric("Effective Borrower IRR", f"{borrower_irr*100:.2f}%" if not np.isnan(borrower_irr) else "—")

            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Interest Earned", f"${total_interest:,.0f}")
            c2.metric("Total Fees Earned", f"${total_fees:,.0f}")
            c3.metric("Starting LVR", f"{init_lvr:.1f}%")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sched["Month"], y=sched["Closing"] / 1e6,
                name="Outstanding Balance ($M)", fill="tozeroy", line=dict(color="#1f77b4")
            ))
            fig.add_trace(go.Scatter(
                x=sched["Month"], y=sched["Current Limit"] / 1e6,
                name="Current Limit ($M)", line=dict(color="gray", dash="dash")
            ))
            fig.add_hline(y=facility / 1e6, line_dash="dot", annotation_text="Original Facility")
            fig.update_layout(height=360, title="Facility Balance & Limits Over Time", yaxis_title="$M")
            st.plotly_chart(fig, use_container_width=True)

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=sched["Month"], y=sched["LVR"], name="LVR %", line=dict(color="red")))
            fig2.add_hline(y=max_lvr_covenant, line_dash="dash", line_color="orange",
                           annotation_text=f"Covenant {max_lvr_covenant:.0f}%")
            fig2.update_layout(height=300, title="Dynamic LVR Curve", yaxis_title="LVR %")
            st.plotly_chart(fig2, use_container_width=True)

        with t3:
            st.metric("Total Fees Over Life", f"${total_fees:,.2f}")
            st.dataframe(sched[["Month", "Fees", "Fee Detail"]], use_container_width=True, height=400)

        with t4:
            if not selected_deal_id:
                st.warning("Select a saved deal from the dropdown above to log actuals.")
            else:
                st.write(f"Logging actuals for Deal ID **{selected_deal_id}**")
                with st.form("log_actual_form"):
                    c1, c2, c3, c4, c5 = st.columns(5)
                    act_m = int(c1.number_input("Month", 0, tenure, 1))
                    act_dd = c2.number_input("Drawdown $", 0.0, step=10000.0)
                    act_int = c3.number_input("Interest Paid $", 0.0, step=1000.0)
                    act_prn = c4.number_input("Principal Paid $", 0.0, step=10000.0)
                    act_fee = c5.number_input("Fees Paid $", 0.0, step=500.0)
                    if st.form_submit_button("Save Actual"):
                        execute_query("DELETE FROM actuals WHERE deal_id=? AND month=?", (selected_deal_id, act_m))
                        execute_query(
                            "INSERT INTO actuals (deal_id, month, actual_drawdown, actual_interest, actual_principal, actual_fees) VALUES (?,?,?,?,?,?)",
                            (selected_deal_id, act_m, act_dd, act_int, act_prn, act_fee)
                        )
                        st.success(f"Saved actuals for month {act_m}")
                        st.rerun()

                if actuals_dict:
                    st.markdown("#### Existing Actuals")
                    st.json({str(k): v for k, v in actuals_dict.items()})

        with t5:
            output = io.BytesIO()
            export_df = sched.drop(columns=["Fee Detail"], errors="ignore")
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                export_df.to_excel(writer, sheet_name="TLS", index=False)
                pd.DataFrame({
                    "Metric": ["Gross Facility", "Upfront Fees Deducted", "Net Cash Disbursed (M0)", "Target IRR", "Borrower IRR", "Total Interest", "Total Fees", "Initial LVR"],
                    "Value": [facility, fees_deducted0, net_disbursed0, f"{target_irr*100:.2f}%",
                              f"{borrower_irr*100:.2f}%" if not np.isnan(borrower_irr) else "N/A",
                              total_interest, total_fees, f"{init_lvr:.1f}%"]
                }).to_excel(writer, sheet_name="Summary", index=False)
            st.download_button(
                "⬇️ Download Excel",
                data=output.getvalue(),
                file_name=f"deal_model_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error("Modeler error — details below")
        st.exception(e)

else:
    st.write("Select a page from the sidebar.")
