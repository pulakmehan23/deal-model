import calendar
import hashlib
import json
import sqlite3
from datetime import date, datetime, timedelta
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import brentq

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG & PRESENTATION
# ═══════════════════════════════════════════════════════════════
st.set_page_config(page_title="Private Credit Platform", layout="wide", page_icon="🏦")

st.markdown("""
<style>
    .main .block-container {padding-top:1.2rem; max-width:96%;}
    h1,h2,h3 {color:#17243a; letter-spacing:-0.02em;}
    div[data-testid="stMetric"] {background:#f7f9fc; border:1px solid #e6ebf2;
        border-radius:12px; padding:14px 16px;}
    div[data-testid="stMetricLabel"] {color:#64748b; font-size:0.78rem;}
    div[data-testid="stMetricValue"] {color:#14233b; font-weight:700;}
    .stTabs [data-baseweb="tab-list"] {gap:6px; border-bottom:1px solid #e7ebf1;}
    .stTabs [data-baseweb="tab"] {padding:10px 14px; font-weight:600;}
    .section-note {background:#f2f6fc; border-left:4px solid #4778b8; padding:10px 14px;
        border-radius:6px; color:#31445e; margin:6px 0 14px 0;}
    .breach-alert {background:#fff3cd; border:1px solid #f4cf66; border-radius:8px;
        padding:11px 14px; margin:8px 0; font-weight:650; color:#785b00;}
    .covenant-ok {background:#e9f7ef; border:1px solid #9ed6b1; border-radius:8px;
        padding:11px 14px; margin:8px 0; font-weight:650; color:#176438;}
    [data-testid="stSidebar"] {background:#f7f9fc;}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# DATE HELPERS — all monthly dates are calendar anniversaries
# ═══════════════════════════════════════════════════════════════

def parse_date(value, fallback=None):
    if value is None:
        return fallback
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        if pd.isna(value):
            return fallback
    except (TypeError, ValueError):
        pass
    try:
        return datetime.fromisoformat(str(value)[:10]).date()
    except (TypeError, ValueError):
        return fallback


def add_months(anchor, months):
    """Add calendar months from the original anchor, safely preserving month-end dates."""
    anchor = parse_date(anchor, date.today())
    month_index = anchor.month - 1 + int(months)
    year = anchor.year + month_index // 12
    month = month_index % 12 + 1
    day = min(anchor.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def infer_term_months(settlement, maturity):
    settlement = parse_date(settlement, date.today())
    maturity = parse_date(maturity, add_months(settlement, 12))
    months = (maturity.year - settlement.year) * 12 + maturity.month - settlement.month
    return max(1, min(120, months or 12))


def date_year_fraction(start_date, end_date):
    return max((end_date - start_date).days, 0) / 365.0


def annual_rate_for_days(annual_rate, days):
    if annual_rate <= -1:
        return -1.0
    return (1.0 + annual_rate) ** (max(days, 0) / 365.0) - 1.0


def money(value, decimals=0):
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
        return f"${float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def percent(value, decimals=2):
    try:
        return f"{float(value):,.{decimals}f}%"
    except (TypeError, ValueError):
        return "—"


def safe_json(value, fallback=None):
    if fallback is None:
        fallback = {}
    if isinstance(value, dict):
        return value
    if not value:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback

# ═══════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════
DB = "lms_data.db"


def get_conn():
    conn = sqlite3.connect(DB, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_conn()
    conn.execute("""CREATE TABLE IF NOT EXISTS borrowers (
        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, abn TEXT, group_name TEXT,
        risk_grade TEXT DEFAULT 'Investment Grade', industry TEXT DEFAULT 'Real Estate',
        status TEXT DEFAULT 'Active')""")
    conn.execute("""CREATE TABLE IF NOT EXISTS deals (
        id INTEGER PRIMARY KEY AUTOINCREMENT, borrower_id INTEGER, deal_name TEXT,
        facility_amount REAL, deal_status TEXT DEFAULT 'Pipeline',
        settlement_date TEXT, maturity_date TEXT, facility_type TEXT DEFAULT 'Construction',
        security_type TEXT DEFAULT 'First Mortgage', property_value REAL DEFAULT 0,
        params_json TEXT DEFAULT '{}',
        FOREIGN KEY (borrower_id) REFERENCES borrowers(id))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS actuals (
        id INTEGER PRIMARY KEY AUTOINCREMENT, deal_id INTEGER, month INTEGER,
        actual_date TEXT, actual_drawdown REAL DEFAULT 0,
        drawdown_override INTEGER DEFAULT 0, payment_actuals_logged INTEGER DEFAULT 1,
        actual_interest REAL DEFAULT 0, actual_principal REAL DEFAULT 0,
        actual_fees REAL DEFAULT 0,
        FOREIGN KEY (deal_id) REFERENCES deals(id))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS investors (
        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE,
        investor_type TEXT DEFAULT 'Fund', target_irr REAL DEFAULT 8.0,
        status TEXT DEFAULT 'Active', notes TEXT DEFAULT '')""")
    conn.execute("""CREATE TABLE IF NOT EXISTS deal_investors (
        deal_id INTEGER NOT NULL, investor_id INTEGER NOT NULL,
        senior_default_pct REAL DEFAULT 0, mezz_default_pct REAL DEFAULT 0,
        is_active INTEGER DEFAULT 1,
        PRIMARY KEY (deal_id, investor_id),
        FOREIGN KEY (deal_id) REFERENCES deals(id),
        FOREIGN KEY (investor_id) REFERENCES investors(id))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS investor_transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, deal_id INTEGER NOT NULL,
        transaction_date TEXT NOT NULL, transaction_type TEXT NOT NULL,
        investor_id INTEGER NOT NULL, counterparty_id INTEGER,
        amount REAL NOT NULL, notes TEXT DEFAULT '', source TEXT DEFAULT 'User',
        FOREIGN KEY (deal_id) REFERENCES deals(id),
        FOREIGN KEY (investor_id) REFERENCES investors(id),
        FOREIGN KEY (counterparty_id) REFERENCES investors(id))""")

    investor_cols = {row[1] for row in conn.execute("PRAGMA table_info(investors)").fetchall()}
    if "target_irr" not in investor_cols:
        conn.execute("ALTER TABLE investors ADD COLUMN target_irr REAL DEFAULT 8.0")
    transaction_cols = {row[1] for row in conn.execute("PRAGMA table_info(investor_transactions)").fetchall()}
    if "source" not in transaction_cols:
        conn.execute("ALTER TABLE investor_transactions ADD COLUMN source TEXT DEFAULT 'User'")

    # The three core pools are available on every deal. Optional investors such as
    # CBFM or Syndicate are added in the Investor Portal and attached deal-by-deal.
    for core_name in ("BPCFMC", "BPCFLC", "CBCF"):
        conn.execute("INSERT OR IGNORE INTO investors (name, investor_type, target_irr) VALUES (?, 'Fund', 8.0)", (core_name,))

    # Backwards-compatible migration from the earlier month-number actuals table.
    actual_cols = {row[1] for row in conn.execute("PRAGMA table_info(actuals)").fetchall()}
    if "actual_date" not in actual_cols:
        conn.execute("ALTER TABLE actuals ADD COLUMN actual_date TEXT")
    if "drawdown_override" not in actual_cols:
        conn.execute("ALTER TABLE actuals ADD COLUMN drawdown_override INTEGER DEFAULT 0")
    if "payment_actuals_logged" not in actual_cols:
        conn.execute("ALTER TABLE actuals ADD COLUMN payment_actuals_logged INTEGER DEFAULT 1")
    conn.execute("""CREATE UNIQUE INDEX IF NOT EXISTS ux_actuals_deal_date
                    ON actuals(deal_id, actual_date) WHERE actual_date IS NOT NULL""")
    conn.commit()
    conn.close()


init_db()

# ═══════════════════════════════════════════════════════════════
# FINANCIAL ENGINE — date-based schedules and XIRR
# ═══════════════════════════════════════════════════════════════

def calc_xirr(cashflows):
    """Annual effective XIRR for a sequence of (date, cashflow) pairs."""
    if not cashflows:
        return 0.0
    clean = []
    for flow_date, amount in cashflows:
        flow_date = parse_date(flow_date)
        if flow_date is not None:
            clean.append((flow_date, float(amount)))
    if len(clean) < 2:
        return 0.0

    by_date = {}
    for flow_date, amount in clean:
        by_date[flow_date] = by_date.get(flow_date, 0.0) + amount
    flows = sorted(by_date.items(), key=lambda item: item[0])
    values = [amount for _, amount in flows]
    if all(value >= 0 for value in values) or all(value <= 0 for value in values):
        return 0.0

    start = flows[0][0]

    def npv(rate):
        return sum(amount / ((1.0 + rate) ** date_year_fraction(start, flow_date))
                   for flow_date, amount in flows)

    try:
        return float(brentq(npv, -0.9999, 100.0, maxiter=2000))
    except (ValueError, RuntimeError, OverflowError, ZeroDivisionError):
        return 0.0


def resolve_drawdown_profile(facility_type, requested_profile):
    if requested_profile != "Automatic":
        return requested_profile
    if facility_type == "Construction":
        return "S-curve"
    if facility_type == "Revolving":
        return "Straight-line"
    # Bridging, residential, term and mezzanine default to a single settlement draw.
    return "Full at settlement"


def make_s_curve_weights(count, steepness=5.0):
    count = max(1, int(count))
    if count == 1:
        return np.array([1.0])
    x = np.linspace(-3.0, 3.0, count + 1)
    k = max(float(steepness), 0.1)
    cdf = 1.0 / (1.0 + np.exp(-k * x / 3.0))
    weights = np.diff(cdf)
    if weights.sum() <= 0:
        return np.repeat(1.0 / count, count)
    return weights / weights.sum()


def build_drawdown_template(settlement, term, facility_amount, facility_type,
                            construction_months, profile="Automatic", steepness=5.0):
    settlement = parse_date(settlement, date.today())
    term = max(1, int(term))
    facility_amount = max(0.0, float(facility_amount))
    draw_dates = [add_months(settlement, i) for i in range(term)]
    amounts = np.zeros(term, dtype=float)
    resolved = resolve_drawdown_profile(facility_type, profile)

    if resolved == "Full at settlement":
        amounts[0] = facility_amount
    else:
        count = max(1, min(term, int(construction_months) if int(construction_months) > 0 else 1))
        if resolved == "S-curve":
            weights = make_s_curve_weights(count, steepness)
        else:  # Straight-line
            weights = np.repeat(1.0 / count, count)
        amounts[:count] = facility_amount * weights
        # Remove floating point rounding from the final active draw.
        amounts[count - 1] += facility_amount - float(amounts.sum())

    return pd.DataFrame({"Drawdown Date": draw_dates, "Drawdown Amount": amounts})


def drawdown_signature(deal_id, facility_amount, facility_type, settlement,
                       term, construction_months, profile, steepness):
    payload = "|".join(map(str, [deal_id, round(float(facility_amount), 2), facility_type,
        parse_date(settlement).isoformat(), int(term), int(construction_months),
        profile, round(float(steepness), 2)]))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:14]


def migrate_drawdown_plan(raw_schedule, settlement, term, template_df):
    """Read current date-keyed schedules and legacy month-keyed schedules."""
    plan = template_df.copy()
    raw = safe_json(raw_schedule, {})
    if not isinstance(raw, dict) or not raw:
        return plan
    dates = [parse_date(value) for value in plan["Drawdown Date"].tolist()]
    amounts = {value.isoformat(): 0.0 for value in dates if value is not None}
    for key, value in raw.items():
        try:
            target = parse_date(key)
            if target is None:
                # Legacy Month 1 was the first draw period, beginning on settlement.
                month_num = int(key)
                target = add_months(settlement, max(month_num - 1, 0))
            if isinstance(value, dict):
                value = value.get("gross_draw", value.get("amount", 0.0))
            if target.isoformat() in amounts:
                amounts[target.isoformat()] = max(0.0, float(value))
        except (TypeError, ValueError):
            continue
    plan["Drawdown Amount"] = [amounts.get(value.isoformat(), 0.0) for value in dates]
    return plan


def allocation_column(investor_name):
    return f"{investor_name} allocation"


def default_allocation_amounts(net_amount, loan_ranking, investor_names):
    return {name: max(0.0, float(net_amount)) * standard_split_for(name, loan_ranking) / 100.0
            for name in investor_names}


def saved_drawdown_rows(raw_schedule, settlement):
    raw = safe_json(raw_schedule, {})
    result = {}
    if not isinstance(raw, dict):
        return result
    for key, value in raw.items():
        try:
            draw_date = parse_date(key)
            if draw_date is None:
                draw_date = add_months(settlement, max(int(key) - 1, 0))
            result[draw_date.isoformat()] = value
        except (TypeError, ValueError):
            continue
    return result


def build_drawdown_matrix(template_df, raw_schedule, actual_calls, investor_names,
                          loan_ranking, upfront_total, settlement, use_template=False):
    saved = {} if use_template else saved_drawdown_rows(raw_schedule, settlement)
    rows = []
    for _, template_row in template_df.iterrows():
        draw_date = parse_date(template_row["Drawdown Date"])
        key = draw_date.isoformat()
        prior = saved.get(key)
        if isinstance(prior, dict):
            gross = max(0.0, float(prior.get("gross_draw", prior.get("amount", template_row["Drawdown Amount"])) or 0))
            source = str(prior.get("allocation_source", "Standard"))
            status = str(prior.get("status", "Forecast"))
            prior_allocations = prior.get("allocations", {}) or {}
        elif prior is not None:
            gross = max(0.0, float(prior or 0))
            source = "Standard"
            status = "Forecast"
            prior_allocations = {}
        else:
            gross = max(0.0, float(template_row["Drawdown Amount"]))
            source = "Standard"
            status = "Forecast"
            prior_allocations = {}

        if key in actual_calls and (prior is None or source != "Manual"):
            source = "Investor Ledger"
            status = "Actual"
        if source not in ["Standard", "Investor Ledger", "Manual"]:
            source = "Standard"
        if status not in ["Forecast", "Actual"]:
            status = "Forecast"
        rows.append({"Drawdown Date": draw_date, "Status": status, "Allocation Source": source,
                     "Gross Drawdown": gross, "_prior_allocations": prior_allocations})

    first_draw_index = next((i for i, row in enumerate(rows) if row["Gross Drawdown"] > 0), None)
    for index, row in enumerate(rows):
        upfront = min(float(upfront_total), row["Gross Drawdown"]) if index == first_draw_index else 0.0
        net_to_allocate = max(0.0, row["Gross Drawdown"] - upfront)
        key = row["Drawdown Date"].isoformat()
        if row["Allocation Source"] == "Standard":
            allocations = default_allocation_amounts(net_to_allocate, loan_ranking, investor_names)
        elif row["Allocation Source"] == "Investor Ledger":
            allocations = {name: float(actual_calls.get(key, {}).get(name, 0.0)) for name in investor_names}
        else:
            allocations = {name: float(row["_prior_allocations"].get(name, 0.0) or 0.0)
                           for name in investor_names}
            if not any(allocations.values()) and key in actual_calls:
                allocations = {name: float(actual_calls[key].get(name, 0.0)) for name in investor_names}
        for name in investor_names:
            row[allocation_column(name)] = max(0.0, allocations.get(name, 0.0))
        row["_net_to_allocate"] = net_to_allocate
        row["_prior_allocations"] = None
    return pd.DataFrame(rows)


def effective_drawdown_plan(edited_plan, investor_names, actual_calls, loan_ranking,
                            upfront_total):
    result = {}
    display_rows = []
    first_draw_index = None
    gross_values = []
    for _, row in edited_plan.iterrows():
        try:
            gross = max(0.0, float(row.get("Gross Drawdown", 0) or 0))
        except (TypeError, ValueError):
            gross = 0.0
        gross_values.append(gross)
    first_draw_index = next((i for i, gross in enumerate(gross_values) if gross > 0), None)

    for index, (_, row) in enumerate(edited_plan.iterrows()):
        draw_date = parse_date(row.get("Drawdown Date"))
        if draw_date is None:
            continue
        key = draw_date.isoformat()
        gross = gross_values[index]
        upfront = min(float(upfront_total), gross) if index == first_draw_index else 0.0
        net_to_allocate = max(0.0, gross - upfront)
        source = str(row.get("Allocation Source", "Standard"))
        status = str(row.get("Status", "Forecast"))
        if source == "Standard":
            allocations = default_allocation_amounts(net_to_allocate, loan_ranking, investor_names)
        elif source == "Investor Ledger":
            allocations = {name: float(actual_calls.get(key, {}).get(name, 0.0)) for name in investor_names}
        else:
            allocations = {}
            for name in investor_names:
                try:
                    allocations[name] = max(0.0, float(row.get(allocation_column(name), 0.0) or 0.0))
                except (TypeError, ValueError):
                    allocations[name] = 0.0
        allocated = sum(allocations.values())
        result[key] = {"gross_draw": gross, "status": status,
            "allocation_source": source, "allocations": allocations}
        summary = {"Drawdown Date": draw_date, "Status": status, "Allocation Source": source,
            "Gross Drawdown": gross, "Upfront Fees Netted": upfront,
            "Net Amount to Allocate": net_to_allocate}
        for name in investor_names:
            summary[name] = allocations.get(name, 0.0)
        summary["Allocation Difference"] = net_to_allocate - allocated
        display_rows.append(summary)
    return result, pd.DataFrame(display_rows)


def compute_fees(params):
    facility_amount = float(params.get("facility_amount", 0))
    term = int(params.get("term_months", 12))
    construction_months = int(params.get("construction_months", 6))
    establishment = facility_amount * float(params.get("estab_fee_pct", 1.5)) / 100.0
    line_rate = float(params.get("line_fee_pct", 0.5)) / 100.0
    admin_monthly = float(params.get("admin_fee_monthly", 500))
    legal = float(params.get("legal_fee", 3500))
    risk = float(params.get("risk_review_fee", 2500))
    other_upfront = float(params.get("other_upfront_fees", 0))
    discharge = float(params.get("discharge_fee", 1500))
    restructure = float(params.get("restructure_fee", 0))
    return {
        "establishment": establishment,
        "legal_fee": legal,
        "risk_review": risk,
        "other_upfront": other_upfront,
        "upfront_total": establishment + legal + risk + other_upfront,
        "line_fee_rate": line_rate,
        "line_fee_estimate": facility_amount * line_rate * (construction_months / 12.0),
        "admin_monthly": admin_monthly,
        "admin_total_estimate": admin_monthly * term,
        "discharge": discharge,
        "restructure": restructure,
    }


def read_drawdown_map(params):
    raw = safe_json(params.get("drawdown_schedule", {}), {})
    result = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            try:
                parsed = parse_date(key)
                if parsed:
                    result[parsed.isoformat()] = max(0.0, float(value))
            except (TypeError, ValueError):
                pass
    return result


def build_schedule(params, actuals_map, drawdown_map):
    facility_amount = float(params.get("facility_amount", 0))
    term = max(1, int(params.get("term_months", 12)))
    construction_months = int(params.get("construction_months", 6))
    settlement = parse_date(params.get("settlement_date"), date.today())
    annual_rate = float(params.get("borrower_irr", 12.0)) / 100.0
    cpr = float(params.get("cpr", 0.0)) / 100.0
    property_value = float(params.get("property_value", 0.0))
    property_growth = float(params.get("pv_growth_annual", 3.0)) / 100.0
    covenant_lvr = float(params.get("covenant_lvr", 75.0))
    penalty_rate = float(params.get("penalty_rate", 4.0)) / 100.0
    fees = compute_fees(params)

    rows = []
    balance = 0.0
    cumulative_drawn = 0.0
    cumulative_interest = 0.0
    arrears = 0.0
    first_draw_done = False
    borrower_cashflows = []

    for index in range(term):
        draw_date = add_months(settlement, index)
        payment_date = add_months(settlement, index + 1)
        days = max((payment_date - draw_date).days, 1)
        actual = actuals_map.get(payment_date.isoformat(), {})
        plan_item = drawdown_map.get(draw_date.isoformat(), {})
        if isinstance(plan_item, dict):
            planned_draw = max(0.0, float(plan_item.get("gross_draw", plan_item.get("amount", 0.0))))
            investor_allocations = dict(plan_item.get("allocations", {}))
            allocation_source = str(plan_item.get("allocation_source", "Standard"))
            drawdown_status = str(plan_item.get("status", "Forecast"))
        else:  # Support old month/date -> amount schedule maps.
            planned_draw = max(0.0, float(plan_item or 0.0))
            investor_allocations = {}
            allocation_source = "Standard"
            drawdown_status = "Forecast"
        if actual.get("drawdown_override", False):
            gross_draw = max(0.0, float(actual.get("actual_drawdown", 0.0)))
        else:
            gross_draw = planned_draw

        opening_balance = balance
        upfront_deducted = 0.0
        if gross_draw > 0 and not first_draw_done:
            upfront_deducted = min(fees["upfront_total"], gross_draw)
            first_draw_done = True
        net_draw = gross_draw - upfront_deducted
        balance_after_draw = opening_balance + gross_draw
        cumulative_drawn += gross_draw

        interest_due = balance_after_draw * annual_rate_for_days(annual_rate, days)
        penalty_interest = arrears * annual_rate_for_days(penalty_rate, days) if arrears > 0 else 0.0
        in_construction = index < construction_months
        undrawn = max(facility_amount - cumulative_drawn, 0.0)
        line_fee = undrawn * fees["line_fee_rate"] * (days / 365.0) if in_construction else 0.0
        admin_fee = fees["admin_monthly"]
        discharge_fee = fees["discharge"] if index == term - 1 else 0.0

        smm = 1.0 - (1.0 - cpr) ** (days / 365.0) if cpr > 0 else 0.0
        prepayment = balance_after_draw * smm if not in_construction and balance_after_draw > 0 else 0.0
        if in_construction:
            scheduled_principal = 0.0
        else:
            remaining_payments = max(term - index, 1)
            scheduled_principal = balance_after_draw / remaining_payments
        contract_principal = min(balance_after_draw, scheduled_principal + prepayment)
        total_due = (interest_due + penalty_interest + contract_principal + line_fee
                     + admin_fee + discharge_fee)

        has_payment_actuals = bool(actual.get("payment_actuals_logged", False))
        actual_interest = float(actual.get("actual_interest", 0.0)) if has_payment_actuals else None
        actual_principal = float(actual.get("actual_principal", 0.0)) if has_payment_actuals else None
        actual_fees = float(actual.get("actual_fees", 0.0)) if has_payment_actuals else None
        if has_payment_actuals:
            actual_total = float(actual_interest or 0.0) + float(actual_principal or 0.0) + float(actual_fees or 0.0)
            # total_due already includes this period's penalty interest.
            arrears = max(0.0, arrears + total_due - actual_total)
            principal_applied = min(max(float(actual_principal or 0.0), 0.0), balance_after_draw)
            cash_interest_received = float(actual_interest or 0.0)
            cash_principal_received = float(actual_principal or 0.0)
            cash_fees_received = float(actual_fees or 0.0)
        else:
            principal_applied = contract_principal
            cash_interest_received = interest_due + penalty_interest
            cash_principal_received = contract_principal
            cash_fees_received = line_fee + admin_fee + discharge_fee

        ending_balance = max(0.0, balance_after_draw - principal_applied)
        elapsed_days = max((payment_date - settlement).days, 0)
        current_property_value = (property_value * ((1.0 + property_growth) ** (elapsed_days / 365.0))
                                 if property_value > 0 else 0.0)
        lvr = ending_balance / current_property_value * 100.0 if current_property_value > 0 else 0.0
        breach = lvr > covenant_lvr
        cumulative_interest += interest_due

        if net_draw > 0:
            borrower_cashflows.append((draw_date, -net_draw))
        payment_cashflow = interest_due + penalty_interest + contract_principal + line_fee + admin_fee + discharge_fee
        if payment_cashflow > 0:
            borrower_cashflows.append((payment_date, payment_cashflow))

        rows.append({
            "Drawdown_Date": draw_date,
            "Payment_Date": payment_date,
            "Period_Days": days,
            "Planned_Drawdown": planned_draw,
            "Allocation_Source": allocation_source,
            "Drawdown_Status": drawdown_status,
            "Investor_Allocations": investor_allocations,
            "Drawdown_Gross": gross_draw,
            "Upfront_Fees_Deducted": upfront_deducted,
            "Drawdown_Net": net_draw,
            "Opening_Balance": opening_balance,
            "Balance_After_Draw": balance_after_draw,
            "Interest_Due": interest_due,
            "Penalty_Interest": penalty_interest,
            "Line_Fee": line_fee,
            "Admin_Fee": admin_fee,
            "Discharge_Fee": discharge_fee,
            "Scheduled_Principal": scheduled_principal,
            "Prepayment": prepayment,
            "Contract_Principal_Due": contract_principal,
            "Principal_Applied": principal_applied,
            "Total_Due": total_due,
            "Actual_Interest": actual_interest,
            "Actual_Principal": actual_principal,
            "Actual_Fees": actual_fees,
            "Cash_Interest_Received": cash_interest_received,
            "Cash_Principal_Received": cash_principal_received,
            "Cash_Fees_Received": cash_fees_received,
            "Arrears": arrears,
            "Ending_Balance": ending_balance,
            "Property_Value": current_property_value,
            "LVR_pct": lvr,
            "Covenant_Breach": breach,
            "Cum_Interest": cumulative_interest,
            "Cum_Drawn": cumulative_drawn,
            "Is_Construction": in_construction,
        })
        balance = ending_balance

    return pd.DataFrame(rows), fees, calc_xirr(borrower_cashflows), borrower_cashflows

# ═══════════════════════════════════════════════════════════════
# INVESTOR ALLOCATION & LEDGER
# ═══════════════════════════════════════════════════════════════
def build_investor_ledger(schedule_df, params, investors_df, transactions_df=None):
    """Build projected investor economics, applying actual movements on their transaction dates."""
    transactions_df = transactions_df if transactions_df is not None else pd.DataFrame()
    investor_names = investors_df["name"].astype(str).tolist() if not investors_df.empty else []
    target_rates = {str(row["name"]): float(row.get("target_irr", 8.0)) / 100.0
                    for _, row in investors_df.iterrows()}

    # Keep historical transaction counterparties and any legacy allocation names visible.
    if not transactions_df.empty:
        for column in ["investor_name", "counterparty_name"]:
            if column in transactions_df:
                for name in transactions_df[column].dropna().astype(str).tolist():
                    if name and name.lower() != "nan" and name not in investor_names:
                        investor_names.append(name)
    scheduled_draw_dates = set()
    if not schedule_df.empty:
        for _, schedule_row in schedule_df.iterrows():
            scheduled_draw_dates.add(parse_date(schedule_row.get("Drawdown_Date")).isoformat())
            raw = schedule_row.get("Investor_Allocations", {})
            if isinstance(raw, str):
                raw = safe_json(raw, {})
            if isinstance(raw, dict):
                for name in raw:
                    if str(name) not in investor_names:
                        investor_names.append(str(name))
    for name in investor_names:
        target_rates.setdefault(name, 0.08)

    balances = {name: 0.0 for name in investor_names}
    total_funded = {name: 0.0 for name in investor_names}
    total_interest = {name: 0.0 for name in investor_names}
    total_principal_returned = {name: 0.0 for name in investor_names}
    cashflows = {name: [] for name in investor_names}
    cashflow_events = []
    period_events = [[] for _ in range(len(schedule_df))]
    loan_ranking = str(params.get("loan_ranking", "Senior"))

    def record_flow(name, flow_date, flow_type, amount, signed_amount):
        amount = max(0.0, float(amount))
        if amount <= 0:
            return
        cashflows.setdefault(name, []).append((flow_date, float(signed_amount)))
        cashflow_events.append({"Date": flow_date, "Investor": name, "Flow_Type": flow_type,
                                "Amount": amount, "Signed_Cashflow": float(signed_amount)})

    # Allocate each portal transaction to one settlement-anniversary interval. Boundary-date
    # movements belong to the period ending on that date (except the first settlement date).
    if not transactions_df.empty and not schedule_df.empty:
        for _, tx in transactions_df.iterrows():
            if str(tx.get("source", "User")) == "Drawdown Plan":
                continue  # Already represented by the saved per-draw allocation row.
            tx_type = str(tx.get("transaction_type", ""))
            tx_date = parse_date(tx.get("transaction_date"))
            if tx_date is None:
                continue
            if tx_type == "Capital Call" and tx_date.isoformat() in scheduled_draw_dates:
                continue  # The drawdown row (Investor Ledger or Manual) is the row-level source of truth.
            investor_name = str(tx.get("investor_name", ""))
            counterparty = tx.get("counterparty_name")
            counterparty = str(counterparty) if pd.notna(counterparty) else ""
            amount = max(0.0, float(tx.get("amount", 0.0) or 0.0))
            if not investor_name or investor_name.lower() == "nan" or amount <= 0:
                continue
            if tx_type not in ["Capital Call", "Principal Repayment", "Interest Distribution",
                               "Buyout", "Other Increase", "Other Decrease"]:
                continue
            for period_index, (_, schedule_row) in enumerate(schedule_df.iterrows()):
                draw_date = parse_date(schedule_row["Drawdown_Date"])
                payment_date = parse_date(schedule_row["Payment_Date"])
                in_period = (draw_date < tx_date <= payment_date)
                if period_index == 0 and tx_date == draw_date:
                    in_period = True
                if in_period:
                    event_order = {"Interest Distribution": 0, "Capital Call": 1, "Other Increase": 2,
                                   "Principal Repayment": 3, "Other Decrease": 4, "Buyout": 5}
                    period_events[period_index].append((tx_date, event_order[tx_type], tx_type,
                        investor_name, counterparty, amount))
                    break

    rows = []
    for period_index, (_, schedule_row) in enumerate(schedule_df.iterrows()):
        draw_date = parse_date(schedule_row["Drawdown_Date"])
        payment_date = parse_date(schedule_row["Payment_Date"])
        net_draw = float(schedule_row["Drawdown_Net"])
        raw_allocations = schedule_row.get("Investor_Allocations", {})
        if isinstance(raw_allocations, str):
            raw_allocations = safe_json(raw_allocations, {})
        raw_allocations = raw_allocations if isinstance(raw_allocations, dict) else {}
        source = str(schedule_row.get("Allocation_Source", "Standard"))
        allocations = {name: max(0.0, float(raw_allocations.get(name, 0.0) or 0.0))
                       for name in investor_names}
        if net_draw > 0 and source == "Standard":
            allocations = {name: net_draw * standard_split_for(name, loan_ranking) / 100.0
                           for name in investor_names}

        period_capital_called = {name: allocations.get(name, 0.0) for name in investor_names}
        period_other_increase = {name: 0.0 for name in investor_names}
        period_other_decrease = {name: 0.0 for name in investor_names}
        period_principal_returns = {name: 0.0 for name in investor_names}
        period_interest_actual = {name: 0.0 for name in investor_names}
        actual_interest_names = set()
        period_buyout_in = {name: 0.0 for name in investor_names}
        period_buyout_out = {name: 0.0 for name in investor_names}
        has_actual_principal = False

        for name, amount in allocations.items():
            balances[name] = balances.get(name, 0.0) + amount
            total_funded[name] = total_funded.get(name, 0.0) + amount
            if amount > 0:
                record_flow(name, draw_date, "Capital Call", amount, -amount)

        accrued_interest = {name: 0.0 for name in investor_names}
        cursor = draw_date

        def accrue_interest(until_date):
            nonlocal cursor
            days = max((until_date - cursor).days, 0)
            for investor_name in list(balances):
                accrued_interest[investor_name] = accrued_interest.get(investor_name, 0.0) + balances[investor_name] * annual_rate_for_days(
                    target_rates.get(investor_name, 0.08), days)
            cursor = until_date

        period_events[period_index].sort(key=lambda event: (event[0], event[1]))
        for tx_date, _, tx_type, investor_name, counterparty, amount in period_events[period_index]:
            accrue_interest(tx_date)
            balances.setdefault(investor_name, 0.0)
            total_funded.setdefault(investor_name, 0.0)
            total_interest.setdefault(investor_name, 0.0)
            total_principal_returned.setdefault(investor_name, 0.0)
            accrued_interest.setdefault(investor_name, 0.0)
            period_capital_called.setdefault(investor_name, 0.0)
            period_other_increase.setdefault(investor_name, 0.0)
            period_other_decrease.setdefault(investor_name, 0.0)
            period_principal_returns.setdefault(investor_name, 0.0)
            period_interest_actual.setdefault(investor_name, 0.0)
            period_buyout_in.setdefault(investor_name, 0.0)
            period_buyout_out.setdefault(investor_name, 0.0)
            if tx_type == "Capital Call":
                balances[investor_name] += amount
                total_funded[investor_name] += amount
                period_capital_called[investor_name] += amount
                record_flow(investor_name, tx_date, "Capital Call", amount, -amount)
            elif tx_type == "Interest Distribution":
                actual_interest_names.add(investor_name)
                period_interest_actual[investor_name] += amount
                record_flow(investor_name, tx_date, "Interest Distribution", amount, amount)
            elif tx_type == "Principal Repayment":
                has_actual_principal = True
                returned = min(amount, max(balances[investor_name], 0.0))
                balances[investor_name] = max(0.0, balances[investor_name] - returned)
                period_principal_returns[investor_name] += returned
                record_flow(investor_name, tx_date, "Principal Repayment", returned, returned)
            elif tx_type == "Other Increase":
                balances[investor_name] += amount
                total_funded[investor_name] += amount
                period_other_increase[investor_name] += amount
                record_flow(investor_name, tx_date, "Other Increase", amount, -amount)
            elif tx_type == "Other Decrease":
                decreased = min(amount, max(balances[investor_name], 0.0))
                balances[investor_name] = max(0.0, balances[investor_name] - decreased)
                period_other_decrease[investor_name] += decreased
                record_flow(investor_name, tx_date, "Other Decrease", decreased, decreased)
            elif tx_type == "Buyout":
                if not counterparty or counterparty.lower() == "nan":
                    continue
                balances.setdefault(counterparty, 0.0)
                total_funded.setdefault(counterparty, 0.0)
                total_interest.setdefault(counterparty, 0.0)
                total_principal_returned.setdefault(counterparty, 0.0)
                accrued_interest.setdefault(counterparty, 0.0)
                period_buyout_in.setdefault(counterparty, 0.0)
                period_buyout_out.setdefault(counterparty, 0.0)
                period_capital_called.setdefault(counterparty, 0.0)
                period_other_increase.setdefault(counterparty, 0.0)
                period_other_decrease.setdefault(counterparty, 0.0)
                period_principal_returns.setdefault(counterparty, 0.0)
                period_interest_actual.setdefault(counterparty, 0.0)
                transferred = min(amount, max(balances[investor_name], 0.0))
                balances[investor_name] = max(0.0, balances[investor_name] - transferred)
                balances[counterparty] += transferred
                period_buyout_out[investor_name] += transferred
                period_buyout_in[counterparty] += transferred
                record_flow(investor_name, tx_date, "Buyout In", transferred, transferred)
                record_flow(counterparty, tx_date, "Buyout Out", transferred, -transferred)

        accrue_interest(payment_date)

        interest_by_investor = {}
        for name in list(balances):
            if name in actual_interest_names:
                interest = period_interest_actual.get(name, 0.0)
            else:
                interest = accrued_interest.get(name, 0.0)
                if interest > 0:
                    record_flow(name, payment_date, "Interest Payment", interest, interest)
            interest_by_investor[name] = interest
            total_interest[name] = total_interest.get(name, 0.0) + interest

        if not has_actual_principal:
            total_balance = sum(max(0.0, balance) for balance in balances.values())
            principal_due = min(float(schedule_row.get("Principal_Applied", 0.0)), total_balance)
            if total_balance > 0 and principal_due > 0:
                for name in list(balances):
                    returned = min(max(0.0, balances[name]), principal_due * max(0.0, balances[name]) / total_balance)
                    period_principal_returns[name] = period_principal_returns.get(name, 0.0) + returned
                    balances[name] = max(0.0, balances[name] - returned)
                    if returned > 0:
                        record_flow(name, payment_date, "Scheduled Principal Return", returned, returned)

        for name in list(balances):
            returned = float(period_principal_returns.get(name, 0.0))
            total_principal_returned[name] = total_principal_returned.get(name, 0.0) + returned
            rows.append({
                "Drawdown_Date": draw_date,
                "Payment_Date": payment_date,
                "Investor": name,
                "Capital_Called": period_capital_called.get(name, 0.0),
                "Other_Increase": period_other_increase.get(name, 0.0),
                "Other_Decrease": period_other_decrease.get(name, 0.0),
                "Interest_Earned": interest_by_investor.get(name, 0.0),
                "Principal_Returned": returned,
                "Buyout_In": period_buyout_in.get(name, 0.0),
                "Buyout_Out": period_buyout_out.get(name, 0.0),
                "Ending_Balance": balances.get(name, 0.0),
                "Cumulative_Funded": total_funded.get(name, 0.0),
                "Cumulative_Interest": total_interest.get(name, 0.0),
            })

    ledger_columns = ["Drawdown_Date", "Payment_Date", "Investor", "Capital_Called", "Other_Increase",
        "Other_Decrease", "Interest_Earned", "Principal_Returned", "Buyout_In", "Buyout_Out",
        "Ending_Balance", "Cumulative_Funded", "Cumulative_Interest"]
    ledger_df = pd.DataFrame(rows, columns=ledger_columns)
    summary_rows = []
    for name in investor_names:
        summary_rows.append({
            "Investor": name,
            "Total_Funded": total_funded.get(name, 0.0),
            "Total_Interest_Earned": total_interest.get(name, 0.0),
            "Total_Principal_Returned": total_principal_returned.get(name, 0.0),
            "Outstanding_Balance": balances.get(name, 0.0),
            "Target_XIRR": target_rates.get(name, 0.08) * 100.0,
            "Actual_XIRR": calc_xirr(cashflows.get(name, [])) * 100.0,
        })
    summary_df = pd.DataFrame(summary_rows, columns=["Investor", "Total_Funded", "Total_Interest_Earned",
        "Total_Principal_Returned", "Outstanding_Balance", "Target_XIRR", "Actual_XIRR"])
    cashflow_events_df = pd.DataFrame(cashflow_events, columns=["Date", "Investor", "Flow_Type", "Amount", "Signed_Cashflow"])
    return ledger_df, summary_df, cashflow_events_df

# ═══════════════════════════════════════════════════════════════
# PLATFORM ECONOMICS & SPV CASHFLOWS
# ═══════════════════════════════════════════════════════════════

def build_pnl(schedule_df, fees, investor_summary_df, params):
    borrower_interest = float(schedule_df["Interest_Due"].sum())
    penalty_interest = float(schedule_df["Penalty_Interest"].sum())
    line_fees = float(schedule_df["Line_Fee"].sum())
    admin_fees = float(schedule_df["Admin_Fee"].sum())
    total_income = (borrower_interest + penalty_interest + line_fees + admin_fees
        + fees["establishment"] + fees["legal_fee"] + fees["risk_review"]
        + fees["other_upfront"] + fees["discharge"] + fees["restructure"])
    investor_expense = float(investor_summary_df["Total_Interest_Earned"].sum()) if len(investor_summary_df) else 0.0
    total_funded = float(investor_summary_df["Total_Funded"].sum()) if len(investor_summary_df) else 0.0
    weighted_cost = 0.0
    if total_funded > 0:
        weighted_cost = sum((row["Total_Funded"] / total_funded) * row["Target_XIRR"]
                            for _, row in investor_summary_df.iterrows())
    borrower_rate = float(params.get("borrower_irr", 12.0))
    return {
        "Borrower Interest Income": borrower_interest,
        "Penalty Interest Income": penalty_interest,
        "Establishment Fee": fees["establishment"],
        "Legal Fee": fees["legal_fee"],
        "Risk Review Fee": fees["risk_review"],
        "Line Fees": line_fees,
        "Admin Fees": admin_fees,
        "Discharge Fee": fees["discharge"],
        "Other Upfront Fees": fees["other_upfront"],
        "Restructure Fee": fees["restructure"],
        "Total Income": total_income,
        "Investor Interest Expense": investor_expense,
        "Net Revenue": total_income - investor_expense,
        "Borrower Rate (%)": borrower_rate,
        "Weighted Investor Target (%)": weighted_cost,
        "Platform Spread (bps)": (borrower_rate - weighted_cost) * 100.0,
    }


def build_cashflows(schedule_df, investor_ledger_df, fees, investor_cashflows=None):
    columns = ["Investor_Capital_Called", "Investor_Other_Increase", "Borrower_Interest_Received",
        "Borrower_Principal_Received", "Borrower_Fees_Received", "Total_Inflow",
        "Net_Disbursed_to_Borrower", "Investor_Interest_Paid", "Investor_Principal_Returned",
        "Investor_Other_Decrease", "Total_Outflow", "Net_SPV_Cashflow", "Cumulative_SPV_Cash"]
    events = {}

    def bucket(event_date):
        event_date = parse_date(event_date)
        if event_date not in events:
            events[event_date] = {name: 0.0 for name in columns if name != "Cumulative_SPV_Cash"}
        return events[event_date]

    for _, row in schedule_df.iterrows():
        draw_date = parse_date(row["Drawdown_Date"])
        payment_date = parse_date(row["Payment_Date"])
        draw_event = bucket(draw_date)
        draw_event["Net_Disbursed_to_Borrower"] += float(row["Drawdown_Net"])
        if float(row["Upfront_Fees_Deducted"]) > 0:
            draw_event["Borrower_Fees_Received"] += float(row["Upfront_Fees_Deducted"])

        payment_event = bucket(payment_date)
        payment_event["Borrower_Interest_Received"] += float(row["Cash_Interest_Received"])
        payment_event["Borrower_Principal_Received"] += float(row["Cash_Principal_Received"])
        payment_event["Borrower_Fees_Received"] += float(row["Cash_Fees_Received"])

    if investor_cashflows is not None and not investor_cashflows.empty:
        flow_columns = {
            "Capital Call": "Investor_Capital_Called",
            "Other Increase": "Investor_Other_Increase",
            "Interest Distribution": "Investor_Interest_Paid",
            "Interest Payment": "Investor_Interest_Paid",
            "Principal Repayment": "Investor_Principal_Returned",
            "Scheduled Principal Return": "Investor_Principal_Returned",
            "Other Decrease": "Investor_Other_Decrease",
        }
        for _, flow in investor_cashflows.iterrows():
            column = flow_columns.get(str(flow["Flow_Type"]))
            if column:
                bucket(flow["Date"])[column] += float(flow["Amount"])
    else:
        # Backward-compatible fallback for callers without dated investor flows.
        for _, row in investor_ledger_df.iterrows():
            draw_event = bucket(row["Drawdown_Date"])
            draw_event["Investor_Capital_Called"] += float(row["Capital_Called"])
            payment_event = bucket(row["Payment_Date"])
            payment_event["Investor_Interest_Paid"] += float(row["Interest_Earned"])
            payment_event["Investor_Principal_Returned"] += float(row["Principal_Returned"])
            draw_event["Investor_Other_Increase"] += float(row.get("Other_Increase", 0.0))
            payment_event["Investor_Other_Decrease"] += float(row.get("Other_Decrease", 0.0))

    rows = []
    cumulative = 0.0
    for event_date in sorted(events):
        row = {"Date": event_date, **events[event_date]}
        row["Total_Inflow"] = (row["Investor_Capital_Called"] + row["Investor_Other_Increase"]
            + row["Borrower_Interest_Received"] + row["Borrower_Principal_Received"]
            + row["Borrower_Fees_Received"])
        row["Total_Outflow"] = (row["Net_Disbursed_to_Borrower"] + row["Investor_Interest_Paid"]
            + row["Investor_Principal_Returned"] + row["Investor_Other_Decrease"])
        row["Net_SPV_Cashflow"] = row["Total_Inflow"] - row["Total_Outflow"]
        cumulative += row["Net_SPV_Cashflow"]
        row["Cumulative_SPV_Cash"] = cumulative
        rows.append(row)
    return pd.DataFrame(rows, columns=["Date"] + columns)

# ═══════════════════════════════════════════════════════════════
# DATABASE HELPERS
# ═══════════════════════════════════════════════════════════════

def load_borrowers():
    conn = get_conn()
    df = pd.read_sql("SELECT * FROM borrowers ORDER BY name", conn)
    conn.close()
    return df


def save_borrower(name, abn, group_name, grade, industry):
    conn = get_conn()
    conn.execute("INSERT INTO borrowers (name, abn, group_name, risk_grade, industry) VALUES (?,?,?,?,?)",
                 (name, abn, group_name, grade, industry))
    conn.commit()
    conn.close()


def load_deals():
    conn = get_conn()
    df = pd.read_sql("""SELECT d.*, b.name AS borrower_name FROM deals d
                       LEFT JOIN borrowers b ON d.borrower_id = b.id ORDER BY d.id DESC""", conn)
    conn.close()
    if not df.empty:
        df["loan_ranking"] = df.apply(lambda row: safe_json(row.get("params_json"), {}).get(
            "loan_ranking", "Mezzanine" if row.get("facility_type") == "Mezzanine" else "Senior"), axis=1)
    else:
        df["loan_ranking"] = pd.Series(dtype="object")
    return df


def save_deal(borrower_id, deal_name, facility, status, settlement, maturity,
              facility_type, security_type, property_value, params):
    conn = get_conn()
    conn.execute("""INSERT INTO deals (borrower_id, deal_name, facility_amount, deal_status,
        settlement_date, maturity_date, facility_type, security_type, property_value, params_json)
        VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (borrower_id, deal_name, facility, status, settlement.isoformat(), maturity.isoformat(),
         facility_type, security_type, property_value, json.dumps(params)))
    deal_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.commit()
    conn.close()
    ensure_default_deal_investors(deal_id)
    return deal_id


def standard_split_for(investor_name, loan_ranking):
    if loan_ranking == "Mezzanine":
        return {"BPCFMC": 80.0, "BPCFLC": 20.0}.get(investor_name, 0.0)
    return {"BPCFMC": 50.0, "BPCFLC": 15.0, "CBCF": 35.0}.get(investor_name, 0.0)


def ensure_default_deal_investors(deal_id):
    conn = get_conn()
    core_defaults = {"BPCFMC": (50.0, 80.0), "BPCFLC": (15.0, 20.0), "CBCF": (35.0, 0.0)}
    for name, (senior_pct, mezz_pct) in core_defaults.items():
        investor = conn.execute("SELECT id FROM investors WHERE name=?", (name,)).fetchone()
        if investor:
            conn.execute("""INSERT OR IGNORE INTO deal_investors
                (deal_id, investor_id, senior_default_pct, mezz_default_pct, is_active)
                VALUES (?,?,?,?,1)""", (int(deal_id), int(investor[0]), senior_pct, mezz_pct))
    conn.commit()
    conn.close()


def load_investors(active_only=True):
    conn = get_conn()
    where = "WHERE status='Active'" if active_only else ""
    df = pd.read_sql(f"SELECT * FROM investors {where} ORDER BY name", conn)
    conn.close()
    return df


def load_deal_investors(deal_id):
    ensure_default_deal_investors(deal_id)
    conn = get_conn()
    df = pd.read_sql("""SELECT i.id, i.name, i.investor_type, i.target_irr, i.status,
        di.senior_default_pct, di.mezz_default_pct
        FROM deal_investors di JOIN investors i ON i.id=di.investor_id
        WHERE di.deal_id=? AND di.is_active=1 AND i.status='Active' ORDER BY i.name""",
        conn, params=(int(deal_id),))
    conn.close()
    return df


def save_investor(name, investor_type, target_irr=8.0, notes=""):
    conn = get_conn()
    conn.execute("INSERT INTO investors (name, investor_type, target_irr, notes) VALUES (?,?,?,?)",
                 (name.strip(), investor_type, float(target_irr), notes.strip()))
    conn.commit()
    conn.close()


def update_investor(investor_id, name, investor_type, target_irr, status, notes=""):
    conn = get_conn()
    conn.execute("""UPDATE investors SET name=?, investor_type=?, target_irr=?, status=?, notes=?
        WHERE id=?""", (name.strip(), investor_type, float(target_irr), status, notes.strip(), int(investor_id)))
    conn.commit()
    conn.close()


def save_deal_investors(deal_id, investor_ids):
    conn = get_conn()
    conn.execute("UPDATE deal_investors SET is_active=0 WHERE deal_id=?", (int(deal_id),))
    defaults = {"BPCFMC": (50.0, 80.0), "BPCFLC": (15.0, 20.0), "CBCF": (35.0, 0.0)}
    for investor_id in investor_ids:
        row = conn.execute("SELECT name FROM investors WHERE id=?", (int(investor_id),)).fetchone()
        if not row:
            continue
        senior_pct, mezz_pct = defaults.get(row[0], (0.0, 0.0))
        conn.execute("""INSERT INTO deal_investors
            (deal_id, investor_id, senior_default_pct, mezz_default_pct, is_active)
            VALUES (?,?,?,?,1)
            ON CONFLICT(deal_id, investor_id) DO UPDATE SET is_active=1""",
            (int(deal_id), int(investor_id), senior_pct, mezz_pct))
    conn.commit()
    conn.close()


def save_investor_transaction(deal_id, tx_date, tx_type, investor_id, counterparty_id, amount,
                              notes="", source="User"):
    conn = get_conn()
    conn.execute("""INSERT INTO investor_transactions
        (deal_id, transaction_date, transaction_type, investor_id, counterparty_id, amount, notes, source)
        VALUES (?,?,?,?,?,?,?,?)""",
        (int(deal_id), parse_date(tx_date).isoformat(), tx_type, int(investor_id),
         int(counterparty_id) if counterparty_id is not None else None,
         float(amount), notes.strip(), source))
    conn.commit()
    conn.close()


def load_investor_transactions(deal_id=None):
    conn = get_conn()
    query = """SELECT t.*, d.deal_name, src.name AS investor_name, dst.name AS counterparty_name
        FROM investor_transactions t JOIN deals d ON d.id=t.deal_id
        JOIN investors src ON src.id=t.investor_id
        LEFT JOIN investors dst ON dst.id=t.counterparty_id"""
    if deal_id is None:
        df = pd.read_sql(query + " ORDER BY t.transaction_date DESC, t.id DESC", conn)
    else:
        df = pd.read_sql(query + " WHERE t.deal_id=? ORDER BY t.transaction_date DESC, t.id DESC",
                         conn, params=(int(deal_id),))
    conn.close()
    return df


def capital_calls_by_date(transactions_df):
    result = {}
    if transactions_df.empty:
        return result
    for _, row in transactions_df.iterrows():
        if row["transaction_type"] != "Capital Call" or row.get("source", "User") == "Drawdown Plan":
            continue
        tx_date = parse_date(row["transaction_date"])
        key = tx_date.isoformat()
        result.setdefault(key, {})
        name = row["investor_name"]
        result[key][name] = result[key].get(name, 0.0) + float(row["amount"])
    return result


def update_deal_model(deal_id, params):
    conn = get_conn()
    conn.execute("""UPDATE deals SET facility_amount=?, settlement_date=?, maturity_date=?,
        facility_type=?, property_value=?, params_json=? WHERE id=?""",
        (float(params["facility_amount"]), params["settlement_date"], params["maturity_date"],
         params["facility_type"], float(params["property_value"]), json.dumps(params), int(deal_id)))
    conn.commit()
    conn.close()
    sync_actual_plan_capital_calls(deal_id, params)


def sync_actual_plan_capital_calls(deal_id, params):
    """Mirror saved Actual drawdown rows into the investor movement ledger."""
    plan = safe_json(params.get("drawdown_schedule", {}), {})
    if not isinstance(plan, dict):
        plan = {}
    conn = get_conn()
    # Replace only entries generated by the plan; user-entered portal movements remain untouched.
    conn.execute("DELETE FROM investor_transactions WHERE deal_id=? AND source='Drawdown Plan'",
                 (int(deal_id),))
    investor_ids = {name: int(investor_id) for investor_id, name in
                    conn.execute("SELECT id, name FROM investors").fetchall()}
    existing_user_calls = {(tx_date, investor_name) for tx_date, investor_name in conn.execute("""
        SELECT t.transaction_date, i.name FROM investor_transactions t
        JOIN investors i ON i.id=t.investor_id
        WHERE t.deal_id=? AND t.transaction_type='Capital Call' AND t.source!='Drawdown Plan'""",
        (int(deal_id),)).fetchall()}
    for date_key, item in plan.items():
        if not isinstance(item, dict) or str(item.get("status", "Forecast")) != "Actual":
            continue
        if str(item.get("allocation_source", "")) == "Investor Ledger":
            continue
        tx_date = parse_date(date_key)
        if tx_date is None:
            continue
        allocations = item.get("allocations", {})
        if not isinstance(allocations, dict):
            continue
        for investor_name, amount in allocations.items():
            investor_name = str(investor_name)
            try:
                amount = float(amount)
            except (TypeError, ValueError):
                continue
            if amount <= 0 or investor_name not in investor_ids:
                continue
            if (tx_date.isoformat(), investor_name) in existing_user_calls:
                continue
            conn.execute("""INSERT INTO investor_transactions
                (deal_id, transaction_date, transaction_type, investor_id, counterparty_id,
                 amount, notes, source) VALUES (?,?,?,?,?,?,?,?)""",
                (int(deal_id), tx_date.isoformat(), "Capital Call", investor_ids[investor_name],
                 None, amount, "Synced from actual drawdown plan", "Drawdown Plan"))
    conn.commit()
    conn.close()


def load_actuals(deal_id):
    conn = get_conn()
    df = pd.read_sql("SELECT * FROM actuals WHERE deal_id=? ORDER BY month, id", conn, params=(deal_id,))
    conn.close()
    return df


def actuals_to_map(actuals_df, settlement, term):
    result = {}
    for _, row in actuals_df.iterrows():
        actual_date = parse_date(row.get("actual_date"))
        if actual_date is None:
            # Month-number records from the earlier version map to payment anniversaries.
            try:
                month_num = int(row.get("month", 1))
            except (TypeError, ValueError):
                month_num = 1
            actual_date = add_months(settlement, max(month_num, 1))
        result[actual_date.isoformat()] = {
            "actual_drawdown": float(row.get("actual_drawdown", 0) or 0),
            "drawdown_override": bool(row.get("drawdown_override", 0)),
            "payment_actuals_logged": bool(row.get("payment_actuals_logged", 1)),
            "actual_interest": float(row.get("actual_interest", 0) or 0),
            "actual_principal": float(row.get("actual_principal", 0) or 0),
            "actual_fees": float(row.get("actual_fees", 0) or 0),
        }
    return result


def save_actual(deal_id, period_index, actual_date, actual_drawdown, drawdown_override,
                payment_actuals_logged, interest, principal, fees_paid):
    conn = get_conn()
    existing = conn.execute("SELECT id FROM actuals WHERE deal_id=? AND actual_date=?",
                            (int(deal_id), actual_date.isoformat())).fetchone()
    values = (int(period_index), float(actual_drawdown), int(bool(drawdown_override)),
              int(bool(payment_actuals_logged)), float(interest), float(principal), float(fees_paid))
    if existing:
        conn.execute("""UPDATE actuals SET month=?, actual_drawdown=?, drawdown_override=?,
            payment_actuals_logged=?, actual_interest=?, actual_principal=?, actual_fees=? WHERE id=?""",
            values + (existing[0],))
    else:
        conn.execute("""INSERT INTO actuals (month, actual_drawdown, drawdown_override,
            payment_actuals_logged, actual_interest, actual_principal, actual_fees, deal_id, actual_date)
            VALUES (?,?,?,?,?,?,?,?,?)""", values + (int(deal_id), actual_date.isoformat()))
    conn.commit()
    conn.close()


def init_widget_state(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

# ═══════════════════════════════════════════════════════════════
# PAGE: PORTFOLIO OVERVIEW
# ═══════════════════════════════════════════════════════════════

def page_dashboard():
    st.title("Portfolio Overview")
    st.markdown('<div class="section-note">A quick view of borrowers, facilities, current exposure and pipeline status.</div>',
                unsafe_allow_html=True)
    borrowers = load_borrowers()
    deals = load_deals()
    total_facility = float(deals["facility_amount"].fillna(0).sum()) if len(deals) else 0.0
    active = int((deals["deal_status"] == "Active").sum()) if len(deals) else 0
    pipeline = int((deals["deal_status"] == "Pipeline").sum()) if len(deals) else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Borrowers", len(borrowers))
    c2.metric("Deals", len(deals))
    c3.metric("Total Facilities", money(total_facility))
    c4.metric("Active / Pipeline", f"{active} / {pipeline}")

    if deals.empty:
        st.info("No deals yet. Add a borrower and create a deal to get started.")
        return

    left, right = st.columns(2)
    with left:
        st.subheader("Deals by status")
        status_df = deals["deal_status"].fillna("Unclassified").value_counts().rename_axis("Status").reset_index(name="Deals")
        fig = px.pie(status_df, names="Status", values="Deals", hole=0.52,
                     color_discrete_sequence=["#3976b9", "#5eb785", "#f0b44d", "#d96b68", "#8b79c6"])
        fig.update_layout(height=340, margin=dict(t=15, b=15), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Facility exposure by type")
        exposure = deals.groupby("facility_type", dropna=False)["facility_amount"].sum().reset_index()
        exposure["facility_type"] = exposure["facility_type"].fillna("Unclassified")
        fig = px.bar(exposure, x="facility_type", y="facility_amount", color="facility_type",
                     labels={"facility_type": "Facility type", "facility_amount": "Facility amount"},
                     color_discrete_sequence=px.colors.qualitative.Safe)
        fig.update_layout(height=340, showlegend=False, margin=dict(t=15, b=15))
        fig.update_yaxes(tickprefix="$", tickformat=",.")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent deals")
    recent_cols = ["id", "borrower_name", "deal_name", "facility_type", "facility_amount",
                   "deal_status", "settlement_date", "maturity_date"]
    recent = deals[[column for column in recent_cols if column in deals.columns]].head(10).copy()
    recent = recent.rename(columns={"id": "Deal ID", "borrower_name": "Borrower", "deal_name": "Deal",
        "facility_type": "Facility Type", "facility_amount": "Facility Amount", "deal_status": "Status",
        "settlement_date": "Settlement", "maturity_date": "Maturity"})
    st.dataframe(recent, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════
# PAGE: BORROWERS
# ═══════════════════════════════════════════════════════════════

def page_borrowers():
    st.title("Borrowers")
    st.caption("Maintain borrower and sponsor details in one directory.")
    with st.expander("＋ Add borrower", expanded=False):
        with st.form("add_borrower"):
            c1, c2 = st.columns(2)
            name = c1.text_input("Borrower name *")
            abn = c2.text_input("ABN / ACN")
            c3, c4 = st.columns(2)
            group_name = c3.text_input("Group / sponsor")
            grade = c4.selectbox("Risk grade", ["Investment Grade", "Sub-Investment Grade", "Unrated"])
            industry = st.selectbox("Industry", ["Real Estate", "Infrastructure", "Healthcare", "Agriculture",
                "Manufacturing", "Technology", "Other"])
            if st.form_submit_button("Save borrower", type="primary"):
                if name.strip():
                    save_borrower(name.strip(), abn.strip(), group_name.strip(), grade, industry)
                    st.success(f"Saved borrower: {name.strip()}")
                    st.rerun()
                else:
                    st.error("Borrower name is required.")

    borrowers = load_borrowers()
    if borrowers.empty:
        st.info("No borrowers have been added yet.")
    else:
        query = st.text_input("Search borrowers", placeholder="Name, sponsor, ABN or industry")
        if query.strip():
            mask = borrowers.astype(str).apply(lambda col: col.str.contains(query.strip(), case=False, na=False)).any(axis=1)
            borrowers = borrowers[mask]
        st.dataframe(borrowers, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════
# PAGE: DEALS
# ═══════════════════════════════════════════════════════════════

FACILITY_TYPES = ["Construction", "Bridging", "Residential", "Term Loan", "Revolving"]
SECURITY_TYPES = ["First Mortgage", "Second Mortgage", "Unsecured", "Personal Guarantee", "Other"]


def page_deals():
    st.title("Deals")
    st.caption("Create facilities here, then manage assumptions and servicing in Deal Workspace.")
    borrowers = load_borrowers()
    if borrowers.empty:
        st.warning("Add a borrower before creating a deal.")
        return

    with st.expander("＋ Create deal", expanded=False):
        with st.form("add_deal"):
            c1, c2 = st.columns(2)
            borrower_labels = {f"{row['name']} (ID {int(row['id'])})": int(row["id"])
                               for _, row in borrowers.iterrows()}
            borrower_label = c1.selectbox("Borrower", list(borrower_labels.keys()))
            deal_name = c2.text_input("Deal name *")
            c3, c4 = st.columns(2)
            facility = c3.number_input("Facility amount", min_value=0.0, value=5_000_000.0, step=100_000.0)
            status = c4.selectbox("Status", ["Pipeline", "Active", "Settled", "Closed", "Default"])
            c5, c6 = st.columns(2)
            settlement = c5.date_input("Settlement date", value=date.today())
            maturity = c6.date_input("Maturity date", value=add_months(date.today(), 12))
            c7, c8, c9 = st.columns(3)
            facility_type = c7.selectbox("Facility purpose / type", FACILITY_TYPES)
            loan_ranking = c8.selectbox("Loan ranking", ["Senior", "Mezzanine"],
                help="Ranking controls the default investor split; construction/bridge/residential controls the drawdown profile.")
            security_type = c9.selectbox("Security type", SECURITY_TYPES)
            property_value = st.number_input("Property / security value", min_value=0.0,
                                             value=8_000_000.0, step=100_000.0)
            if st.form_submit_button("Create deal", type="primary"):
                if not deal_name.strip():
                    st.error("Deal name is required.")
                elif maturity <= settlement:
                    st.error("Maturity date must be after settlement date.")
                else:
                    borrower_id = borrower_labels[borrower_label]
                    save_deal(borrower_id, deal_name.strip(), facility, status, settlement, maturity,
                              facility_type, security_type, property_value, {"loan_ranking": loan_ranking})
                    st.success(f"Created deal: {deal_name.strip()}")
                    st.rerun()

    deals = load_deals()
    if deals.empty:
        st.info("No deals have been created yet.")
        return
    status_filter = st.multiselect("Filter by status", sorted(deals["deal_status"].dropna().unique().tolist()),
                                   default=sorted(deals["deal_status"].dropna().unique().tolist()))
    filtered = deals[deals["deal_status"].isin(status_filter)]
    show_cols = ["id", "borrower_name", "deal_name", "facility_type", "loan_ranking", "facility_amount", "deal_status",
                 "settlement_date", "maturity_date", "security_type", "property_value"]
    display = filtered[[column for column in show_cols if column in filtered.columns]].rename(columns={
        "id": "Deal ID", "borrower_name": "Borrower", "deal_name": "Deal", "facility_type": "Facility Type",
        "loan_ranking": "Ranking", "facility_amount": "Facility Amount", "deal_status": "Status", "settlement_date": "Settlement",
        "maturity_date": "Maturity", "security_type": "Security", "property_value": "Security Value"})
    st.dataframe(display, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════
# PAGE: DEAL WORKSPACE
# ═══════════════════════════════════════════════════════════════

def page_modeler():
    st.title("Deal Workspace")
    deals = load_deals()
    if deals.empty:
        st.warning("Create a deal first in Deals.")
        return

    deal_labels = {f"{row['deal_name']} · {row['borrower_name']} · ID {int(row['id'])}": int(row["id"])
                   for _, row in deals.iterrows()}
    selected_label = st.selectbox("Select deal", list(deal_labels.keys()), key="workspace_deal_select")
    deal_id = deal_labels[selected_label]
    deal_row = deals[deals["id"] == deal_id].iloc[0]
    existing_params = safe_json(deal_row.get("params_json", "{}"), {})
    ensure_default_deal_investors(deal_id)
    deal_investors_df = load_deal_investors(deal_id)
    investor_transactions_df = load_investor_transactions(deal_id)
    actual_capital_calls = capital_calls_by_date(investor_transactions_df)

    settlement_default = parse_date(existing_params.get("settlement_date"),
                                    parse_date(deal_row.get("settlement_date"), date.today()))
    maturity_default = parse_date(existing_params.get("maturity_date"),
                                  parse_date(deal_row.get("maturity_date"), add_months(settlement_default, 12)))
    term_default = int(existing_params.get("term_months", infer_term_months(settlement_default, maturity_default)))
    term_default = max(1, min(120, term_default))
    facility_default = float(existing_params.get("facility_amount", deal_row.get("facility_amount", 0)) or 0)
    property_default = float(existing_params.get("property_value", deal_row.get("property_value", 0)) or 0)
    facility_type_raw = str(existing_params.get("facility_type", deal_row.get("facility_type", "Construction")) or "Construction")
    legacy_mezzanine_type = facility_type_raw == "Mezzanine"
    facility_type_default = facility_type_raw if facility_type_raw in FACILITY_TYPES else "Term Loan"
    loan_ranking_default = str(existing_params.get("loan_ranking", "Mezzanine" if legacy_mezzanine_type else "Senior"))
    if loan_ranking_default not in ["Senior", "Mezzanine"]:
        loan_ranking_default = "Senior"

    st.markdown(f"### {deal_row['deal_name']}  ·  {deal_row['borrower_name']}")
    st.caption(f"Deal ID {deal_id}  |  Status: {deal_row['deal_status']}  |  Security: {deal_row['security_type']}")
    st.markdown('<div class="section-note">Edit assumptions or drawdowns once. All schedules, investor allocations, economics and cashflows below recalculate from the current inputs. Save changes from the Overview tab to persist them.</div>',
                unsafe_allow_html=True)

    tabs = st.tabs(["Overview", "Setup & Terms", "Drawdown Plan", "Loan Schedule",
                    "Actuals & Arrears", "Investor Base", "Economics", "SPV Cashflows",
                    "Fees & Settlement", "Export"])

    # ── Setup & terms ──────────────────────────────────────────
    prefix = f"deal_{deal_id}_"
    with tabs[1]:
        setup_tabs = st.tabs(["Facility & Dates", "Returns & Allocation", "Fees & Covenants"])
        with setup_tabs[0]:
            st.subheader("Facility and timeline")
            c1, c2, c3 = st.columns(3)
            init_widget_state(prefix + "facility_type", facility_type_default)
            facility_type = c1.selectbox("Facility type", FACILITY_TYPES, key=prefix + "facility_type")
            init_widget_state(prefix + "facility_amount", facility_default)
            facility_amount = c2.number_input("Facility amount", min_value=0.0, max_value=10_000_000_000.0,
                step=100_000.0, key=prefix + "facility_amount", help="The drawdown plan is checked against this limit.")
            init_widget_state(prefix + "property_value", property_default)
            property_value = c3.number_input("Property / security value", min_value=0.0,
                max_value=10_000_000_000.0, step=100_000.0, key=prefix + "property_value")

            c4, c5, c6 = st.columns(3)
            init_widget_state(prefix + "settlement_date", settlement_default)
            settlement = c4.date_input("Settlement date / anniversary anchor", key=prefix + "settlement_date")
            init_widget_state(prefix + "term_months", term_default)
            term_months = c5.number_input("Term (calendar months)", min_value=1, max_value=120,
                step=1, key=prefix + "term_months")
            maturity_date = add_months(settlement, int(term_months))
            c6.metric("Maturity anniversary", maturity_date.strftime("%d %b %Y"))
            st.caption("Each payment date is generated from the settlement-date anniversary. Month-end dates are preserved (for example, 31 January → 28 February → 31 March).")

            construction_default = int(existing_params.get("construction_months", min(6, int(term_months))))
            construction_default = max(0, min(int(term_months), construction_default))
            init_widget_state(prefix + "construction_months", construction_default)
            if st.session_state[prefix + "construction_months"] > int(term_months):
                st.session_state[prefix + "construction_months"] = int(term_months)
            c7, c8, c9 = st.columns(3)
            construction_months = c7.number_input("Interest-only / construction periods", min_value=0,
                max_value=int(term_months), step=1, key=prefix + "construction_months",
                help="The construction flag controls the drawdown curve and when scheduled principal begins.")
            init_widget_state(prefix + "borrower_irr", float(existing_params.get("borrower_irr", 12.0)))
            borrower_irr = c8.number_input("Borrower rate (% p.a.)", min_value=0.0, max_value=50.0,
                step=0.25, key=prefix + "borrower_irr")
            init_widget_state(prefix + "cpr", float(existing_params.get("cpr", 0.0)))
            cpr = c9.number_input("Annual prepayment rate / CPR (%)", min_value=0.0, max_value=50.0,
                step=0.5, key=prefix + "cpr")

            c10, c11, c12 = st.columns(3)
            init_widget_state(prefix + "drawdown_profile", str(existing_params.get("drawdown_profile", "Automatic")))
            profile_options = ["Automatic", "Full at settlement", "S-curve", "Straight-line"]
            if st.session_state[prefix + "drawdown_profile"] not in profile_options:
                st.session_state[prefix + "drawdown_profile"] = "Automatic"
            drawdown_profile = c10.selectbox("Default drawdown profile", profile_options,
                key=prefix + "drawdown_profile", help="Automatic: Construction uses an S-curve, Revolving uses straight-line, other types draw at settlement.")
            init_widget_state(prefix + "s_curve_steepness", float(existing_params.get("s_curve_steepness", 5.0)))
            s_curve_steepness = c11.slider("S-curve shape", min_value=1.0, max_value=9.0, step=0.5,
                key=prefix + "s_curve_steepness", help="Higher values concentrate construction draws around the middle of the build period.")
            initial_security_type = str(deal_row.get("security_type", "First Mortgage") or "First Mortgage")
            c12.text_input("Security type", value=initial_security_type, disabled=True)

        with setup_tabs[1]:
            st.subheader("Loan ranking and investors")
            init_widget_state(prefix + "loan_ranking", loan_ranking_default)
            if st.session_state[prefix + "loan_ranking"] not in ["Senior", "Mezzanine"]:
                st.session_state[prefix + "loan_ranking"] = loan_ranking_default
            loan_ranking = st.selectbox("Loan ranking", ["Senior", "Mezzanine"],
                key=prefix + "loan_ranking",
                help="Ranking determines the standard split. Facility purpose separately determines the drawdown curve.")
            if loan_ranking == "Senior":
                st.info("Senior default: BPCFMC 50% · BPCFLC 15% · CBCF 35%. Optional investors are 0% until you allocate them on a row.")
            else:
                st.info("Mezzanine default: BPCFMC 80% · BPCFLC 20%. CBCF is not included by default; optional investors can be attached and allocated when needed.")
            if not deal_investors_df.empty:
                inv_view = deal_investors_df[["name", "investor_type", "target_irr"]].rename(columns={
                    "name": "Investor", "investor_type": "Type", "target_irr": "Target XIRR (%)"})
                st.dataframe(inv_view, use_container_width=True, hide_index=True)
            st.caption("Attach or remove optional investors (for example, CBFM or Syndicate) in the Investor Portal. Their allocation columns will then appear in the drawdown plan.")

        with setup_tabs[2]:
            st.subheader("Fees and security assumptions")
            f1, f2, f3, f4 = st.columns(4)
            init_widget_state(prefix + "estab_fee_pct", float(existing_params.get("estab_fee_pct", 1.5)))
            estab_fee_pct = f1.number_input("Establishment fee (%)", min_value=0.0, max_value=10.0,
                step=0.25, key=prefix + "estab_fee_pct")
            init_widget_state(prefix + "line_fee_pct", float(existing_params.get("line_fee_pct", 0.5)))
            line_fee_pct = f2.number_input("Undrawn line fee (% p.a.)", min_value=0.0, max_value=5.0,
                step=0.1, key=prefix + "line_fee_pct")
            init_widget_state(prefix + "admin_fee_monthly", float(existing_params.get("admin_fee_monthly", 500.0)))
            admin_fee_monthly = f3.number_input("Admin fee per payment date", min_value=0.0, max_value=5_000.0,
                step=50.0, key=prefix + "admin_fee_monthly")
            init_widget_state(prefix + "discharge_fee", float(existing_params.get("discharge_fee", 1_500.0)))
            discharge_fee = f4.number_input("Discharge fee", min_value=0.0, max_value=50_000.0,
                step=100.0, key=prefix + "discharge_fee")

            f5, f6, f7, f8 = st.columns(4)
            init_widget_state(prefix + "risk_review_fee", float(existing_params.get("risk_review_fee", 2_500.0)))
            risk_review_fee = f5.number_input("Risk review fee", min_value=0.0, max_value=50_000.0,
                step=100.0, key=prefix + "risk_review_fee")
            init_widget_state(prefix + "legal_fee", float(existing_params.get("legal_fee", 3_500.0)))
            legal_fee = f6.number_input("Legal fee", min_value=0.0, max_value=50_000.0,
                step=100.0, key=prefix + "legal_fee")
            init_widget_state(prefix + "other_upfront_fees", float(existing_params.get("other_upfront_fees", 0.0)))
            other_upfront_fees = f7.number_input("Other upfront fees", min_value=0.0, max_value=100_000.0,
                step=100.0, key=prefix + "other_upfront_fees")
            init_widget_state(prefix + "restructure_fee", float(existing_params.get("restructure_fee", 0.0)))
            restructure_fee = f8.number_input("Restructure fee", min_value=0.0, max_value=100_000.0,
                step=100.0, key=prefix + "restructure_fee")

            c1, c2, c3 = st.columns(3)
            init_widget_state(prefix + "pv_growth_annual", float(existing_params.get("pv_growth_annual", 3.0)))
            pv_growth_annual = c1.number_input("Property growth (% p.a.)", min_value=-10.0, max_value=30.0,
                step=0.5, key=prefix + "pv_growth_annual")
            init_widget_state(prefix + "covenant_lvr", float(existing_params.get("covenant_lvr", 75.0)))
            covenant_lvr = c2.number_input("Covenant LVR (%)", min_value=0.0, max_value=100.0,
                step=1.0, key=prefix + "covenant_lvr")
            init_widget_state(prefix + "penalty_rate", float(existing_params.get("penalty_rate", 4.0)))
            penalty_rate = c3.number_input("Penalty rate (% p.a.)", min_value=0.0, max_value=30.0,
                step=0.5, key=prefix + "penalty_rate")

    # Current inputs become the single source of truth for the live model.
    maturity_date = add_months(settlement, int(term_months))
    params = {
        "facility_amount": float(facility_amount), "facility_type": facility_type,
        "property_value": float(property_value), "settlement_date": settlement.isoformat(),
        "maturity_date": maturity_date.isoformat(), "term_months": int(term_months),
        "construction_months": int(construction_months), "borrower_irr": float(borrower_irr),
        "cpr": float(cpr), "drawdown_profile": drawdown_profile,
        "s_curve_steepness": float(s_curve_steepness), "loan_ranking": loan_ranking,
        "estab_fee_pct": float(estab_fee_pct),
        "line_fee_pct": float(line_fee_pct), "admin_fee_monthly": float(admin_fee_monthly),
        "discharge_fee": float(discharge_fee), "risk_review_fee": float(risk_review_fee),
        "legal_fee": float(legal_fee), "other_upfront_fees": float(other_upfront_fees),
        "restructure_fee": float(restructure_fee), "pv_growth_annual": float(pv_growth_annual),
        "covenant_lvr": float(covenant_lvr), "penalty_rate": float(penalty_rate),
    }

    # ── Editable drawdown and investor allocation matrix ─────────
    signature = drawdown_signature(deal_id, facility_amount, facility_type, settlement,
        term_months, construction_months, drawdown_profile, s_curve_steepness)
    stored_signature = existing_params.get("drawdown_signature")
    template = build_drawdown_template(settlement, term_months, facility_amount, facility_type,
        construction_months, drawdown_profile, s_curve_steepness)
    investor_names = deal_investors_df["name"].tolist()
    if not investor_transactions_df.empty:
        for column in ["investor_name", "counterparty_name"]:
            for investor_name in investor_transactions_df[column].dropna().astype(str).tolist():
                if investor_name not in investor_names:
                    investor_names.append(investor_name)
    upfront_total = compute_fees(params)["upfront_total"]
    force_template_key = prefix + "force_drawdown_template"
    reset_count_key = prefix + "drawdown_reset_count"
    init_widget_state(force_template_key, False)
    init_widget_state(reset_count_key, 0)
    allocation_fingerprint = hashlib.sha1(json.dumps({
        "investors": investor_names, "ranking": loan_ranking,
        "calls": actual_capital_calls}, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:9]

    with tabs[2]:
        st.subheader("Drawdown and investor allocation plan")
        resolved_profile = resolve_drawdown_profile(facility_type, drawdown_profile)
        profile_description = (f"Construction S-curve over up to {max(1, int(construction_months))} draw dates."
            if resolved_profile == "S-curve" else
            f"Straight-line over up to {max(1, int(construction_months))} draw dates."
            if resolved_profile == "Straight-line" else "Full draw at settlement by default.")
        standard_text = ("Senior default: 50% BPCFMC / 15% BPCFLC / 35% CBCF."
            if loan_ranking == "Senior" else "Mezz default: 80% BPCFMC / 20% BPCFLC; CBCF is excluded by default.")
        st.markdown(f'<div class="section-note">{profile_description} {standard_text} Optional investors are included only when attached to this deal. Choose Standard, Investor Ledger or Manual for each row. Manual allocations are editable dollar amounts and should reconcile to the net amount to allocate.</div>',
                    unsafe_allow_html=True)
        st.caption("Investor Ledger rows use Capital Call movements recorded in the Investor Portal. Forecast rows default to the Senior/Mezz split. Choose Manual to override a row; the model checks each row against the net funding amount after upfront fee netting.")

        reset_clicked = st.button("Reset to suggested schedule", key=prefix + "reset_drawdown_button")
        if reset_clicked:
            st.session_state[force_template_key] = True
            st.session_state[reset_count_key] = int(st.session_state[reset_count_key]) + 1
            st.rerun()

        use_template = bool(st.session_state.get(force_template_key, False)) or (
            stored_signature is not None and stored_signature != signature)
        initial_matrix = build_drawdown_matrix(template, existing_params.get("drawdown_schedule", {}),
            actual_capital_calls, investor_names, loan_ranking, upfront_total, settlement, use_template)
        initial_matrix = initial_matrix.drop(columns=["_prior_allocations", "_net_to_allocate"], errors="ignore")
        st.session_state[force_template_key] = False
        editor_key = (prefix + "allocation_matrix_" + signature + "_" + allocation_fingerprint + "_"
                      + str(st.session_state[reset_count_key]))
        column_config = {
            "Drawdown Date": st.column_config.DateColumn("Drawdown date", format="DD MMM YYYY", disabled=True),
            "Status": st.column_config.SelectboxColumn("Actual / forecast", options=["Forecast", "Actual"], required=True),
            "Allocation Source": st.column_config.SelectboxColumn("Allocation source",
                options=["Standard", "Investor Ledger", "Manual"], required=True),
            "Gross Drawdown": st.column_config.NumberColumn("Gross drawdown", min_value=0.0,
                max_value=max(float(facility_amount), 0.0), step=10_000.0, format="$%.0f", required=True),
        }
        for investor_name in investor_names:
            column_config[allocation_column(investor_name)] = st.column_config.NumberColumn(
                investor_name, min_value=0.0, max_value=max(float(facility_amount), 0.0),
                step=10_000.0, format="$%.0f", required=True)
        edited_plan = st.data_editor(initial_matrix, key=editor_key, hide_index=True,
            use_container_width=True, num_rows="fixed", column_config=column_config)
        drawdown_map, plan_summary = effective_drawdown_plan(edited_plan, investor_names,
            actual_capital_calls, loan_ranking, upfront_total)

        gross_total = float(plan_summary["Gross Drawdown"].sum()) if not plan_summary.empty else 0.0
        net_total = float(plan_summary["Net Amount to Allocate"].sum()) if not plan_summary.empty else 0.0
        allocation_total = float(plan_summary[investor_names].sum().sum()) if investor_names and not plan_summary.empty else 0.0
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Gross draws", money(gross_total))
        s2.metric("Net to investors", money(net_total))
        s3.metric("Allocated to investors", money(allocation_total))
        s4.metric("Unallocated / overallocated", money(net_total - allocation_total))
        if gross_total > float(facility_amount) + 0.01:
            st.error("Total gross drawdowns exceed the facility amount.")
        allocation_issues = plan_summary[plan_summary["Allocation Difference"].abs() > 0.01] if not plan_summary.empty else pd.DataFrame()
        if not allocation_issues.empty:
            issue_dates = ", ".join(value.strftime("%d %b %Y") for value in allocation_issues["Drawdown Date"])
            st.warning(f"Investor allocations do not reconcile to net funding on: {issue_dates}. Check the row’s Allocation Source or adjust its investor amounts.")

        summary_format = {column: "${:,.0f}" for column in plan_summary.columns
                          if column not in ["Drawdown Date", "Status", "Allocation Source"]}
        st.dataframe(plan_summary.style.format(summary_format), use_container_width=True, hide_index=True)
        if not plan_summary.empty:
            plan_chart = plan_summary.copy()
            plan_chart["Cumulative Drawn"] = plan_chart["Gross Drawdown"].cumsum()
            fig = go.Figure()
            fig.add_trace(go.Bar(x=plan_chart["Drawdown Date"], y=plan_chart["Gross Drawdown"],
                                 name="Gross draw", marker_color="#4778b8"))
            fig.add_trace(go.Scatter(x=plan_chart["Drawdown Date"], y=plan_chart["Cumulative Drawn"],
                name="Cumulative drawn", mode="lines+markers", yaxis="y2", line=dict(color="#43a581", width=3)))
            fig.update_layout(title="Drawdowns and cumulative facility usage", height=350,
                xaxis_title="Drawdown date", yaxis=dict(title="Gross draw", tickprefix="$"),
                yaxis2=dict(title="Cumulative drawn", overlaying="y", side="right", tickprefix="$"),
                legend=dict(orientation="h", y=1.12), margin=dict(t=55, b=20))
            st.plotly_chart(fig, use_container_width=True)

    params["drawdown_schedule"] = json.dumps(drawdown_map)
    params["drawdown_signature"] = signature

    # Build actuals and all model outputs from the current inputs and edited grid.
    actuals_df = load_actuals(deal_id)
    actuals_map = actuals_to_map(actuals_df, settlement, term_months)
    schedule_df, fees, borrower_xirr, borrower_cashflows = build_schedule(params, actuals_map, drawdown_map)
    investor_ledger_df, investor_summary_df, investor_cashflows = build_investor_ledger(
        schedule_df, params, deal_investors_df, investor_transactions_df)
    pnl = build_pnl(schedule_df, fees, investor_summary_df, params)
    cashflow_df = build_cashflows(schedule_df, investor_ledger_df, fees, investor_cashflows)

    # ── Overview ────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("Live deal overview")
        st.caption("All figures below use the current on-screen assumptions and drawdown plan. Click Save all changes to persist them to the deal record.")
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Facility", money(facility_amount))
        k2.metric("Gross drawn", money(schedule_df["Drawdown_Gross"].sum()))
        k3.metric("Outstanding at maturity", money(schedule_df["Ending_Balance"].iloc[-1]))
        k4.metric("Projected borrower XIRR", percent(borrower_xirr * 100.0))
        k5.metric("Investor target cost", percent(pnl["Weighted Investor Target (%)"]))
        k6.metric("Platform spread", f"{pnl['Platform Spread (bps)']:.0f} bps")
        st.markdown("#### Key dates")
        d1, d2, d3 = st.columns(3)
        d1.metric("Settlement", settlement.strftime("%d %b %Y"))
        d2.metric("Next anniversary", schedule_df["Payment_Date"].iloc[0].strftime("%d %b %Y"))
        d3.metric("Maturity anniversary", maturity_date.strftime("%d %b %Y"))
        st.markdown("#### Save current deal setup")
        st.caption("This saves terms, dates, fees, investor assumptions and the edited drawdown schedule to the selected deal.")
        if st.button("Save all changes", type="primary", key=prefix + "save_all_changes"):
            update_deal_model(deal_id, params)
            st.success("Deal assumptions saved; Actual drawdown allocations have been synced to the Investor Portal.")

        due_breaches = schedule_df[schedule_df["Covenant_Breach"]]
        if not due_breaches.empty:
            st.markdown("<div class='breach-alert'>Covenant warning: projected LVR exceeds the covenant on one or more payment dates. See Loan Schedule for dates and values.</div>", unsafe_allow_html=True)
        elif float(facility_amount) > 0 and float(property_value) > 0:
            st.markdown("<div class='covenant-ok'>No projected covenant breaches under the current assumptions.</div>", unsafe_allow_html=True)

        a, b = st.columns(2)
        with a:
            st.markdown("#### Outstanding balance by anniversary")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=schedule_df["Payment_Date"], y=schedule_df["Ending_Balance"],
                                     mode="lines+markers", name="Loan balance", line=dict(color="#4778b8", width=3)))
            fig.update_layout(height=310, xaxis_title="Payment date", yaxis_title="Outstanding", yaxis_tickprefix="$",
                              margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
        with b:
            st.markdown("#### Income mix")
            income_data = pd.DataFrame({"Type": ["Interest", "Fees", "Investor cost", "Net revenue"],
                "Amount": [float(schedule_df["Interest_Due"].sum() + schedule_df["Penalty_Interest"].sum()),
                    float(pnl["Total Income"] - schedule_df["Interest_Due"].sum() - schedule_df["Penalty_Interest"].sum()),
                    float(pnl["Investor Interest Expense"]), float(pnl["Net Revenue"])]})
            fig = px.bar(income_data, x="Type", y="Amount", color="Type",
                         color_discrete_sequence=["#4778b8", "#43a581", "#d96b68", "#6b83c4"])
            fig.update_layout(height=310, showlegend=False, margin=dict(t=20, b=20))
            fig.update_yaxes(tickprefix="$", tickformat=",.")
            st.plotly_chart(fig, use_container_width=True)

    # ── Borrower schedule and covenant monitoring ───────────────
    with tabs[3]:
        st.subheader("Borrower loan schedule")
        breaches = schedule_df[schedule_df["Covenant_Breach"]]
        if not breaches.empty:
            breach_dates = ", ".join(value.strftime("%d %b %Y") for value in breaches["Payment_Date"])
            st.markdown(f'<div class="breach-alert">Covenant breach projected on: {breach_dates}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="covenant-ok">All projected LVRs are within the selected covenant.</div>', unsafe_allow_html=True)

        display_cols = ["Drawdown_Date", "Payment_Date", "Drawdown_Gross", "Upfront_Fees_Deducted",
            "Drawdown_Net", "Opening_Balance", "Balance_After_Draw", "Interest_Due", "Penalty_Interest",
            "Line_Fee", "Admin_Fee", "Discharge_Fee", "Scheduled_Principal", "Prepayment",
            "Contract_Principal_Due", "Actual_Interest", "Actual_Principal", "Actual_Fees", "Total_Due",
            "Arrears", "Principal_Applied", "Ending_Balance", "Property_Value", "LVR_pct", "Covenant_Breach"]
        display = schedule_df[display_cols].rename(columns={
            "Drawdown_Date": "Drawdown Date", "Payment_Date": "Payment Date",
            "Drawdown_Gross": "Gross Drawdown", "Upfront_Fees_Deducted": "Upfront Fees Netted",
            "Drawdown_Net": "Net Drawdown", "Opening_Balance": "Opening Balance",
            "Balance_After_Draw": "Balance After Draw", "Interest_Due": "Interest Due",
            "Penalty_Interest": "Penalty Interest", "Line_Fee": "Line Fee", "Admin_Fee": "Admin Fee",
            "Discharge_Fee": "Discharge Fee", "Scheduled_Principal": "Scheduled Principal",
            "Prepayment": "Prepayment", "Contract_Principal_Due": "Contract Principal Due",
            "Actual_Interest": "Actual Interest", "Actual_Principal": "Actual Principal",
            "Actual_Fees": "Actual Fees", "Total_Due": "Total Due", "Arrears": "Arrears",
            "Principal_Applied": "Principal Applied", "Ending_Balance": "Ending Balance",
            "Property_Value": "Property Value", "LVR_pct": "LVR (%)", "Covenant_Breach": "Breach"})
        money_columns = [column for column in display.columns if column not in
                         ["Drawdown Date", "Payment Date", "LVR (%)", "Breach"]]
        format_map = {column: "${:,.0f}" for column in money_columns}
        format_map["LVR (%)"] = "{:.1f}%"
        styled = display.style.format(format_map, na_rep="—")
        st.dataframe(styled, use_container_width=True, height=500)

        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=schedule_df["Payment_Date"], y=schedule_df["Ending_Balance"],
                mode="lines+markers", name="Loan balance", line=dict(color="#4778b8", width=3)))
            fig.add_trace(go.Scatter(x=schedule_df["Payment_Date"], y=schedule_df["Property_Value"],
                mode="lines", name="Property value", line=dict(color="#43a581", dash="dash")))
            fig.update_layout(title="Balance vs security value", xaxis_title="Payment date", yaxis_title="$",
                yaxis_tickprefix="$", height=340, margin=dict(t=45, b=20))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            colors = ["#d96b68" if breach else "#43a581" for breach in schedule_df["Covenant_Breach"]]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=schedule_df["Payment_Date"], y=schedule_df["LVR_pct"],
                                 marker_color=colors, name="LVR"))
            fig.add_hline(y=float(covenant_lvr), line_dash="dash", line_color="#b94141",
                          annotation_text=f"Covenant {covenant_lvr:.1f}%")
            fig.update_layout(title="LVR monitoring", xaxis_title="Payment date", yaxis_title="LVR (%)",
                              height=340, margin=dict(t=45, b=20))
            st.plotly_chart(fig, use_container_width=True)

    # ── Actuals and arrears ─────────────────────────────────────
    with tabs[4]:
        st.subheader("Actual payments and arrears")
        st.markdown('<div class="section-note">Select a payment anniversary and log actual collections. Logged principal and drawdown overrides feed through to the outstanding balance and investor principal ledger; shortfalls update arrears and SPV cashflows.</div>',
                    unsafe_allow_html=True)
        payment_dates = [parse_date(value) for value in schedule_df["Payment_Date"].tolist()]
        selected_payment_date = st.selectbox("Payment / anniversary date", payment_dates,
            format_func=lambda value: value.strftime("%d %B %Y"), key=prefix + "actual_date_select")
        selected_row = schedule_df[schedule_df["Payment_Date"] == selected_payment_date].iloc[0]
        existing_actual = actuals_map.get(selected_payment_date.isoformat(), {})
        existing_override = bool(existing_actual.get("drawdown_override", False))
        existing_payment_log = bool(existing_actual.get("payment_actuals_logged", False))
        st.write(f"Draw period starts **{selected_row['Drawdown_Date'].strftime('%d %b %Y')}** · scheduled due **{money(selected_row['Total_Due'])}** · planned draw **{money(selected_row['Planned_Drawdown'])}**")

        with st.form(prefix + "actuals_form"):
            override_key = prefix + "actual_override_" + selected_payment_date.isoformat()
            record_key = prefix + "actual_record_" + selected_payment_date.isoformat()
            draw_key = prefix + "actual_draw_" + selected_payment_date.isoformat()
            interest_key = prefix + "actual_interest_" + selected_payment_date.isoformat()
            principal_key = prefix + "actual_principal_" + selected_payment_date.isoformat()
            fees_key = prefix + "actual_fees_" + selected_payment_date.isoformat()
            init_widget_state(override_key, existing_override)
            init_widget_state(record_key, existing_payment_log)
            init_widget_state(draw_key, float(existing_actual.get("actual_drawdown", selected_row["Planned_Drawdown"]) or 0))
            init_widget_state(interest_key, float(existing_actual.get("actual_interest", 0) or 0))
            init_widget_state(principal_key, float(existing_actual.get("actual_principal", 0) or 0))
            init_widget_state(fees_key, float(existing_actual.get("actual_fees", 0) or 0))
            override_draw = st.checkbox("Override the planned drawdown with an actual draw amount", key=override_key)
            actual_draw = st.number_input("Actual drawdown for this period", min_value=0.0,
                max_value=10_000_000_000.0, step=10_000.0, key=draw_key, disabled=not override_draw)
            record_payments = st.checkbox("Record actual collections for this payment date", key=record_key)
            c1, c2, c3 = st.columns(3)
            actual_interest = c1.number_input("Actual interest received", min_value=0.0,
                max_value=1_000_000_000.0, step=100.0, key=interest_key, disabled=not record_payments)
            actual_principal = c2.number_input("Actual principal received", min_value=0.0,
                max_value=1_000_000_000.0, step=100.0, key=principal_key, disabled=not record_payments)
            actual_fees = c3.number_input("Actual fees received", min_value=0.0,
                max_value=1_000_000_000.0, step=100.0, key=fees_key, disabled=not record_payments)
            submitted = st.form_submit_button("Save actuals", type="primary")
            if submitted:
                if not override_draw and not record_payments:
                    st.error("Choose an actual drawdown override, record collections, or both before saving.")
                else:
                    period_index = int(schedule_df.index[schedule_df["Payment_Date"] == selected_payment_date][0]) + 1
                    save_actual(deal_id, period_index, selected_payment_date, actual_draw, override_draw,
                                record_payments, actual_interest, actual_principal, actual_fees)
                    st.success("Actuals saved. The schedule and related outputs have been recalculated.")
                    st.rerun()

        if not actuals_df.empty:
            st.markdown("#### Logged actuals")
            actual_display = actuals_df.copy()
            actual_display["Date"] = actual_display.apply(
                lambda row: parse_date(row.get("actual_date")) or add_months(settlement, max(int(row.get("month", 1) or 1), 1)), axis=1)
            actual_display = actual_display.rename(columns={
                "actual_drawdown": "Actual Drawdown", "drawdown_override": "Drawdown Override",
                "payment_actuals_logged": "Payment Actuals Logged", "actual_interest": "Actual Interest",
                "actual_principal": "Actual Principal", "actual_fees": "Actual Fees"})
            columns = ["Date", "Actual Drawdown", "Drawdown Override", "Payment Actuals Logged",
                       "Actual Interest", "Actual Principal", "Actual Fees"]
            st.dataframe(actual_display[[column for column in columns if column in actual_display.columns]],
                         use_container_width=True, hide_index=True)

    # ── Investor ledger ─────────────────────────────────────────
    with tabs[5]:
        st.subheader("Investor base and returns")
        st.info("Standard rows use the deal ranking: Senior 50% / 15% / 35% (BPCFMC / BPCFLC / CBCF), or Mezzanine 80% / 20% (BPCFMC / BPCFLC). Investor Ledger and Manual rows use their recorded/entered splits. Buyouts and actual repayments are applied from Investor Portal transactions.")
        summary_format = {"Total_Funded": "${:,.0f}", "Total_Interest_Earned": "${:,.0f}",
            "Total_Principal_Returned": "${:,.0f}", "Outstanding_Balance": "${:,.0f}",
            "Target_XIRR": "{:.2f}%", "Actual_XIRR": "{:.2f}%"}
        st.dataframe(investor_summary_df.style.format(summary_format), use_container_width=True, hide_index=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=investor_summary_df["Investor"], y=investor_summary_df["Target_XIRR"],
                             name="Target XIRR", marker_color="#43a581"))
        fig.add_trace(go.Bar(x=investor_summary_df["Investor"], y=investor_summary_df["Actual_XIRR"],
                             name="Modelled XIRR", marker_color="#4778b8"))
        fig.update_layout(title="Target vs modelled XIRR", barmode="group", yaxis_title="XIRR (%)",
                          height=330, margin=dict(t=45, b=20))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Detailed investor ledger")
        investor_options = investor_ledger_df["Investor"].unique().tolist()
        selected_investors = st.multiselect("Investors", investor_options, default=investor_options,
                                            key=prefix + "investor_filter")
        filtered_ledger = investor_ledger_df[investor_ledger_df["Investor"].isin(selected_investors)].copy()
        ledger_format = {"Capital_Called": "${:,.0f}", "Other_Increase": "${:,.0f}",
            "Other_Decrease": "${:,.0f}", "Interest_Earned": "${:,.0f}",
            "Principal_Returned": "${:,.0f}", "Buyout_In": "${:,.0f}", "Buyout_Out": "${:,.0f}",
            "Ending_Balance": "${:,.0f}",
            "Cumulative_Funded": "${:,.0f}", "Cumulative_Interest": "${:,.0f}"}
        st.dataframe(filtered_ledger.style.format(ledger_format), use_container_width=True, hide_index=True, height=380)
        if selected_investors:
            fig = go.Figure()
            for investor in selected_investors:
                data = filtered_ledger[filtered_ledger["Investor"] == investor]
                fig.add_trace(go.Scatter(x=data["Payment_Date"], y=data["Ending_Balance"],
                    mode="lines+markers", name=investor))
            fig.update_layout(title="Outstanding investor balances", xaxis_title="Payment date",
                yaxis_title="Balance", yaxis_tickprefix="$", height=320, margin=dict(t=45, b=20))
            st.plotly_chart(fig, use_container_width=True)

    # ── Economics ───────────────────────────────────────────────
    pnl_rows = []
    for label, value in pnl.items():
        if "(%)" in label:
            display_value = percent(value)
        elif "(bps)" in label or "bps" in label:
            display_value = f"{value:,.0f} bps"
        else:
            display_value = money(value)
        pnl_rows.append({"Line item": label, "Amount": display_value})
    pnl_display = pd.DataFrame(pnl_rows)
    with tabs[6]:
        st.subheader("Platform economics")
        st.caption("Income and investor cost are modelled from the current contractual schedule. Actual receipts are shown separately in SPV Cashflows.")
        st.dataframe(pnl_display, use_container_width=True, hide_index=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total income", money(pnl["Total Income"]))
        c2.metric("Investor interest expense", money(pnl["Investor Interest Expense"]))
        c3.metric("Net revenue", money(pnl["Net Revenue"]))
        chart_items = ["Borrower Interest Income", "Penalty Interest Income", "Establishment Fee",
            "Legal Fee", "Risk Review Fee", "Line Fees", "Admin Fees", "Discharge Fee",
            "Other Upfront Fees", "Restructure Fee"]
        chart_df = pd.DataFrame({"Income item": chart_items, "Amount": [pnl[key] for key in chart_items]})
        fig = px.bar(chart_df, x="Income item", y="Amount", color="Income item",
                     color_discrete_sequence=px.colors.qualitative.Safe)
        fig.update_layout(title="Income components", showlegend=False, height=360, margin=dict(t=45, b=90))
        fig.update_yaxes(tickprefix="$", tickformat=",.")
        st.plotly_chart(fig, use_container_width=True)

    # ── SPV cashflows ───────────────────────────────────────────
    with tabs[7]:
        st.subheader("SPV cashflows by event date")
        st.caption("Investor calls, repayments, distributions and other movements are recorded on their transaction dates. Forecast interest and scheduled principal remain on settlement anniversaries; logged actual movements replace the affected investor forecast.")
        cashflow_format = {column: "${:,.0f}" for column in cashflow_df.columns if column != "Date"}
        st.dataframe(cashflow_df.style.format(cashflow_format), use_container_width=True, hide_index=True, height=420)
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=cashflow_df["Date"], y=cashflow_df["Total_Inflow"], name="Inflows", marker_color="#43a581"))
            fig.add_trace(go.Bar(x=cashflow_df["Date"], y=-cashflow_df["Total_Outflow"], name="Outflows", marker_color="#d96b68"))
            fig.update_layout(title="SPV inflows and outflows", barmode="relative", xaxis_title="Date",
                              yaxis_title="$", yaxis_tickprefix="$", height=340, margin=dict(t=45, b=20))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cashflow_df["Date"], y=cashflow_df["Cumulative_SPV_Cash"],
                mode="lines+markers", fill="tozeroy", name="Cumulative cash", line=dict(color="#4778b8", width=3)))
            fig.update_layout(title="Cumulative SPV cash position", xaxis_title="Date", yaxis_title="$",
                              yaxis_tickprefix="$", height=340, margin=dict(t=45, b=20))
            st.plotly_chart(fig, use_container_width=True)

    # ── Fees and settlement ─────────────────────────────────────
    with tabs[8]:
        st.subheader("Fees and settlement summary")
        first_draw_rows = schedule_df[schedule_df["Drawdown_Gross"] > 0]
        first_draw = float(first_draw_rows.iloc[0]["Drawdown_Gross"]) if not first_draw_rows.empty else 0.0
        waterfall = [
            ("First gross draw", first_draw),
            ("Establishment fee", -fees["establishment"]),
            ("Legal fee", -fees["legal_fee"]),
            ("Risk review fee", -fees["risk_review"]),
            ("Other upfront fees", -fees["other_upfront"]),
            ("Net cash to borrower", first_draw - min(fees["upfront_total"], first_draw)),
        ]
        left, right = st.columns([1, 1.25])
        with left:
            st.markdown("#### Settlement waterfall")
            st.dataframe(pd.DataFrame([{"Item": label, "Amount": money(amount)} for label, amount in waterfall]),
                         use_container_width=True, hide_index=True)
            st.caption("Upfront fees are deducted from the first draw only. The discharge fee is scheduled at maturity.")
        with right:
            st.markdown("#### Fee summary")
            fee_rows = [
                {"Fee": "Establishment", "Amount": money(fees["establishment"])},
                {"Fee": "Legal", "Amount": money(fees["legal_fee"])},
                {"Fee": "Risk review", "Amount": money(fees["risk_review"])},
                {"Fee": "Other upfront", "Amount": money(fees["other_upfront"])},
                {"Fee": "Total upfront (netted from first draw)", "Amount": money(fees["upfront_total"])},
                {"Fee": "Line fee (modelled from actual undrawn balance)", "Amount": money(schedule_df["Line_Fee"].sum())},
                {"Fee": "Admin fees", "Amount": money(schedule_df["Admin_Fee"].sum())},
                {"Fee": "Discharge", "Amount": money(fees["discharge"])},
                {"Fee": "Restructure", "Amount": money(fees["restructure"])},
            ]
            st.dataframe(pd.DataFrame(fee_rows), use_container_width=True, hide_index=True)
        all_fees = (fees["upfront_total"] + float(schedule_df["Line_Fee"].sum())
                    + float(schedule_df["Admin_Fee"].sum()) + fees["discharge"] + fees["restructure"])
        st.metric("Total modelled fees", money(all_fees))

    # ── Export ──────────────────────────────────────────────────
    with tabs[9]:
        st.subheader("Export model workbook")
        st.caption("The workbook includes the editable drawdown plan, date-based borrower schedule, investor ledgers, economics, actuals and SPV cashflows.")
        if st.button("Generate Excel workbook", type="primary", key=prefix + "make_export"):
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                summary_export = pd.DataFrame([
                    {"Item": "Deal", "Value": deal_row["deal_name"]},
                    {"Item": "Borrower", "Value": deal_row["borrower_name"]},
                    {"Item": "Facility amount", "Value": facility_amount},
                    {"Item": "Settlement date", "Value": settlement},
                    {"Item": "Maturity date", "Value": maturity_date},
                    {"Item": "Facility type", "Value": facility_type},
                    {"Item": "Projected borrower XIRR", "Value": borrower_xirr},
                    {"Item": "Platform spread (bps)", "Value": pnl["Platform Spread (bps)"]},
                ])
                summary_export.to_excel(writer, sheet_name="Deal Summary", index=False)
                plan_summary.to_excel(writer, sheet_name="Drawdown Plan", index=False)
                schedule_df.drop(columns=["Is_Construction"], errors="ignore").to_excel(writer, sheet_name="Borrower Schedule", index=False)
                investor_ledger_df.to_excel(writer, sheet_name="Investor Ledger", index=False)
                investor_summary_df.to_excel(writer, sheet_name="Investor Summary", index=False)
                pnl_display.to_excel(writer, sheet_name="Economics", index=False)
                cashflow_df.to_excel(writer, sheet_name="SPV Cashflows", index=False)
                if not actuals_df.empty:
                    actuals_df.to_excel(writer, sheet_name="Actuals", index=False)
                for sheet in writer.book.worksheets:
                    sheet.freeze_panes = "A2"
                    sheet.auto_filter.ref = sheet.dimensions
            buffer.seek(0)
            st.download_button("Download Excel", data=buffer.getvalue(),
                file_name=f"Deal_{deal_id}_Model_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=prefix + "download_export")

# ═══════════════════════════════════════════════════════════════
# PAGE: INVESTOR PORTAL (INTERNAL)
# ═══════════════════════════════════════════════════════════════

def transaction_balances(transactions_df, as_of=None):
    balances = {}
    names = {}
    cutoff = parse_date(as_of) if as_of is not None else None
    for _, row in transactions_df.iterrows():
        tx_date = parse_date(row.get("transaction_date"))
        if cutoff is not None and (tx_date is None or tx_date > cutoff):
            continue
        investor_id = int(row["investor_id"])
        investor_name = str(row["investor_name"])
        names[investor_id] = investor_name
        tx_type = str(row["transaction_type"])
        amount = float(row["amount"])
        balances[investor_id] = balances.get(investor_id, 0.0)
        if tx_type in ["Capital Call", "Other Increase"]:
            balances[investor_id] += amount
        elif tx_type in ["Principal Repayment", "Other Decrease"]:
            balances[investor_id] -= amount
        elif tx_type == "Buyout":
            balances[investor_id] -= amount
            counterparty = row.get("counterparty_id")
            if pd.notna(counterparty):
                counterparty = int(counterparty)
                names[counterparty] = str(row.get("counterparty_name", "Investor"))
                balances[counterparty] = balances.get(counterparty, 0.0) + amount
    return pd.DataFrame([{"Investor": names.get(investor_id, str(investor_id)),
        "Outstanding Capital (actual ledger)": balance}
        for investor_id, balance in sorted(balances.items(), key=lambda item: names.get(item[0], ""))])


def page_investor_portal():
    st.title("Investor Portal")
    st.caption("Internal investor register, deal assignments and dated movement ledger. Optional investors such as CBFM or Syndicate are attached only to the deals where they participate.")
    investor_tabs = st.tabs(["Investor Register", "Deal Assignments", "Record Movement", "Activity & Balances"])

    # Register investors
    with investor_tabs[0]:
        st.subheader("Investor register")
        with st.expander("＋ Add investor", expanded=False):
            with st.form("add_investor_form"):
                c1, c2, c3 = st.columns(3)
                investor_name = c1.text_input("Investor name *")
                investor_type = c2.selectbox("Investor type", ["Fund", "CBFM", "Syndicate", "Institutional", "Individual", "Other"])
                investor_target = c3.number_input("Target XIRR (% p.a.)", min_value=0.0, max_value=50.0, value=8.0, step=0.25)
                investor_notes = st.text_input("Notes")
                if st.form_submit_button("Save investor", type="primary"):
                    if not investor_name.strip():
                        st.error("Investor name is required.")
                    else:
                        try:
                            save_investor(investor_name, investor_type, investor_target, investor_notes)
                            st.success(f"Added {investor_name.strip()}.")
                            st.rerun()
                        except sqlite3.IntegrityError:
                            st.error("An investor with that name already exists.")
        investors = load_investors(active_only=False)
        if investors.empty:
            st.info("No investors found.")
        else:
            st.dataframe(investors[["id", "name", "investor_type", "target_irr", "status", "notes"]].rename(columns={
                "id": "ID", "name": "Investor", "investor_type": "Type", "target_irr": "Target XIRR (%)",
                "status": "Status", "notes": "Notes"}), use_container_width=True, hide_index=True)
            manage_id = st.selectbox("Manage investor", investors["id"].astype(int).tolist(),
                format_func=lambda value: f"{investors.loc[investors.id == int(value), 'name'].iloc[0]} · ID {int(value)}",
                key="manage_investor_id")
            manage_row = investors[investors["id"] == manage_id].iloc[0]
            is_core = str(manage_row["name"]) in ["BPCFMC", "BPCFLC", "CBCF"]
            type_options = ["Fund", "CBFM", "Syndicate", "Institutional", "Individual", "Other"]
            if str(manage_row["investor_type"]) not in type_options:
                type_options.append(str(manage_row["investor_type"]))
            with st.form(f"edit_investor_form_{int(manage_id)}"):
                ec1, ec2, ec3 = st.columns(3)
                edited_name = ec1.text_input("Investor name", value=str(manage_row["name"]),
                    disabled=is_core, key=f"edit_investor_name_{int(manage_id)}")
                edited_type = ec2.selectbox("Investor type", type_options,
                    index=type_options.index(str(manage_row["investor_type"])),
                    key=f"edit_investor_type_{int(manage_id)}")
                edited_target = ec3.number_input("Target XIRR (% p.a.)", min_value=0.0, max_value=50.0,
                    value=float(manage_row["target_irr"] or 8.0), step=0.25,
                    key=f"edit_investor_target_{int(manage_id)}")
                status_options = ["Active", "Inactive"]
                current_status = str(manage_row["status"] or "Active")
                if current_status not in status_options:
                    status_options.append(current_status)
                edited_status = st.selectbox("Status", status_options,
                    index=status_options.index(current_status), disabled=is_core,
                    key=f"edit_investor_status_{int(manage_id)}")
                edited_notes = st.text_area("Notes", value=str(manage_row["notes"] or ""),
                    key=f"edit_investor_notes_{int(manage_id)}")
                if st.form_submit_button("Save investor details"):
                    if not edited_name.strip():
                        st.error("Investor name is required.")
                    else:
                        try:
                            update_investor(manage_id, edited_name, edited_type, edited_target,
                                            edited_status, edited_notes)
                            st.success("Investor details updated.")
                            st.rerun()
                        except sqlite3.IntegrityError:
                            st.error("An investor with that name already exists.")
            if is_core:
                st.caption("Core pool names and active status are fixed to preserve standard allocation rules.")

    deals = load_deals()
    active_investors = load_investors(active_only=True)
    if deals.empty:
        with investor_tabs[1]:
            st.info("Create a deal before assigning investors.")
        with investor_tabs[2]:
            st.info("Create a deal before recording investor movements.")
        with investor_tabs[3]:
            st.info("No deal activity to show yet.")
        return

    deal_labels = {f"{row['deal_name']} · {row['borrower_name']} · ID {int(row['id'])}": int(row["id"])
                   for _, row in deals.iterrows()}
    label_by_deal = {value: key for key, value in deal_labels.items()}
    investor_labels = {int(row["id"]): str(row["name"]) for _, row in active_investors.iterrows()}

    # Attach optional investor entities to a deal
    with investor_tabs[1]:
        st.subheader("Deal investor assignments")
        assignment_label = st.selectbox("Select deal", list(deal_labels.keys()), key="investor_assign_deal")
        assignment_deal_id = deal_labels[assignment_label]
        current_assignment = load_deal_investors(assignment_deal_id)
        current_ids = [int(value) for value in current_assignment["id"].tolist()]
        assignment_key = f"investor_assignment_{assignment_deal_id}"
        if assignment_key not in st.session_state:
            st.session_state[assignment_key] = current_ids
        selected_ids = st.multiselect("Investors attached to this deal", list(investor_labels.keys()),
            key=assignment_key, format_func=lambda value: investor_labels.get(int(value), str(value)))
        st.caption("Core funds are attached automatically. Add optional investors here; their allocation columns will appear in the deal’s drawdown matrix.")
        if st.button("Save deal assignments", type="primary", key=f"save_assignments_{assignment_deal_id}"):
            save_deal_investors(assignment_deal_id, selected_ids)
            st.success("Deal investor assignments saved.")
            st.rerun()
        if not current_assignment.empty:
            assignment_view = current_assignment[["name", "investor_type", "target_irr", "senior_default_pct", "mezz_default_pct"]].rename(columns={
                "name": "Investor", "investor_type": "Type", "target_irr": "Target XIRR (%)",
                "senior_default_pct": "Senior default (%)", "mezz_default_pct": "Mezz default (%)"})
            st.dataframe(assignment_view, use_container_width=True, hide_index=True)

    # Record actual movements; buyouts are entered as a single total with a user-defined seller split.
    with investor_tabs[2]:
        st.subheader("Record an investor movement")
        movement_label = st.selectbox("Deal", list(deal_labels.keys()), key="movement_deal")
        movement_deal_id = deal_labels[movement_label]
        attached = load_deal_investors(movement_deal_id)
        if attached.empty:
            st.warning("Attach at least one investor to this deal first.")
        else:
            attached_ids = [int(value) for value in attached["id"].tolist()]
            attached_names = {int(row["id"]): str(row["name"]) for _, row in attached.iterrows()}
            move_type = st.selectbox("Movement type", ["Capital Call", "Principal Repayment", "Interest Distribution",
                "Buyout", "Other Increase", "Other Decrease"], key=f"movement_type_{movement_deal_id}")
            movement_date = st.date_input("Movement date", value=date.today(), key=f"movement_date_{movement_deal_id}")
            if move_type == "Buyout":
                buyer_id = st.selectbox("Buying investor", attached_ids,
                    format_func=lambda value: attached_names.get(int(value), str(value)), key=f"buyout_buyer_{movement_deal_id}")
                sellers = [value for value in attached_ids if value != buyer_id]
                if not sellers:
                    st.warning("Attach at least two investors to record a buyout.")
                else:
                    buyout_total = st.number_input("Total buyout amount", min_value=0.0,
                        max_value=1_000_000_000.0, value=0.0, step=10_000.0, key=f"buyout_amount_{movement_deal_id}")
                    shares_df = pd.DataFrame({"Selling Investor": [attached_names[value] for value in sellers],
                        "Share (%)": [0.0 for _ in sellers]})
                    shares_key = f"buyout_shares_{movement_deal_id}_{buyer_id}"
                    shares = st.data_editor(shares_df, key=shares_key, hide_index=True, use_container_width=True,
                        num_rows="fixed", column_config={
                            "Selling Investor": st.column_config.TextColumn("Selling investor", disabled=True),
                            "Share (%)": st.column_config.NumberColumn("Share of buyout (%)", min_value=0.0,
                                max_value=100.0, step=5.0, format="%.1f", required=True),
                        })
                    share_total = float(pd.to_numeric(shares["Share (%)"], errors="coerce").fillna(0).sum())
                    st.caption(f"Seller shares allocated: {share_total:.1f}% (must total 100%). Enter your agreed buyout ratio; a separate ledger transfer will be recorded for each seller.")
                    buyout_notes = st.text_input("Buyout notes", key=f"buyout_notes_{movement_deal_id}")
                    if st.button("Record buyout", type="primary", key=f"save_buyout_{movement_deal_id}"):
                        if buyout_total <= 0:
                            st.error("Enter a buyout amount greater than zero.")
                        elif abs(share_total - 100.0) > 0.01:
                            st.error("Seller shares must add up to 100%.")
                        else:
                            prior_transactions = load_investor_transactions(movement_deal_id)
                            prior_balances_df = transaction_balances(prior_transactions, as_of=movement_date)
                            prior_balances = dict(zip(prior_balances_df.get("Investor", []),
                                prior_balances_df.get("Outstanding Capital (actual ledger)", [])))
                            requested = []
                            invalid_seller = None
                            for _, share_row in shares.iterrows():
                                share_pct = float(share_row["Share (%)"] or 0)
                                if share_pct <= 0:
                                    continue
                                seller_name = str(share_row["Selling Investor"])
                                transfer_amount = buyout_total * share_pct / 100.0
                                if transfer_amount > float(prior_balances.get(seller_name, 0.0)) + 0.01:
                                    invalid_seller = seller_name
                                    break
                                seller_id = next(value for value in sellers if attached_names[value] == seller_name)
                                requested.append((seller_id, transfer_amount))
                            if invalid_seller:
                                st.error(f"Buyout exceeds {invalid_seller}'s recorded capital balance. Log the relevant capital call first or reduce the seller share.")
                            else:
                                for seller_id, transfer_amount in requested:
                                    save_investor_transaction(movement_deal_id, movement_date, "Buyout",
                                        seller_id, buyer_id, transfer_amount, buyout_notes)
                                st.success("Buyout transfer recorded.")
                                st.rerun()
            else:
                with st.form(f"movement_form_{movement_deal_id}"):
                    tx_investor = st.selectbox("Investor", attached_ids,
                        format_func=lambda value: attached_names.get(int(value), str(value)))
                    tx_amount = st.number_input("Amount", min_value=0.0, max_value=1_000_000_000.0,
                        value=0.0, step=10_000.0)
                    tx_notes = st.text_input("Notes / reference")
                    if st.form_submit_button("Save movement", type="primary"):
                        if tx_amount <= 0:
                            st.error("Enter an amount greater than zero.")
                        elif move_type in ["Principal Repayment", "Other Decrease"]:
                            prior_transactions = load_investor_transactions(movement_deal_id)
                            prior_balances_df = transaction_balances(prior_transactions, as_of=movement_date)
                            prior_balances = dict(zip(prior_balances_df.get("Investor", []),
                                prior_balances_df.get("Outstanding Capital (actual ledger)", [])))
                            investor_name = attached_names[int(tx_investor)]
                            if tx_amount > float(prior_balances.get(investor_name, 0.0)) + 0.01:
                                st.error(f"Repayment exceeds {investor_name}'s recorded capital balance.")
                            else:
                                save_investor_transaction(movement_deal_id, movement_date, move_type,
                                    tx_investor, None, tx_amount, tx_notes)
                                st.success("Investor movement recorded.")
                                st.rerun()
                        else:
                            save_investor_transaction(movement_deal_id, movement_date, move_type,
                                tx_investor, None, tx_amount, tx_notes)
                            st.success("Investor movement recorded.")
                            st.rerun()

    # Ledger and capital balances derived from actual movement records
    with investor_tabs[3]:
        st.subheader("Investor activity and actual capital balances")
        activity_filter = st.selectbox("Deal filter", ["All deals"] + list(deal_labels.keys()),
                                       key="investor_activity_filter")
        if activity_filter == "All deals":
            transactions = load_investor_transactions()
        else:
            transactions = load_investor_transactions(deal_labels[activity_filter])
        if transactions.empty:
            st.info("No investor movements recorded yet. Use Record Movement to log calls, repayments, distributions or buyouts.")
        else:
            def transaction_from(row):
                if row["transaction_type"] == "Capital Call":
                    return "Investor"
                if row["transaction_type"] == "Buyout":
                    return row["investor_name"]
                return "SPV"
            def transaction_to(row):
                if row["transaction_type"] == "Capital Call":
                    return "SPV"
                if row["transaction_type"] == "Buyout":
                    return row["counterparty_name"]
                return row["investor_name"]
            activity = pd.DataFrame({
                "Date": transactions["transaction_date"],
                "Deal": transactions["deal_name"],
                "Movement": transactions["transaction_type"],
                "From": transactions.apply(transaction_from, axis=1),
                "To": transactions.apply(transaction_to, axis=1),
                "Amount": transactions["amount"],
                "Source": transactions["source"],
                "Notes": transactions["notes"],
            })
            st.dataframe(activity.style.format({"Amount": "${:,.0f}"}), use_container_width=True, hide_index=True)
            st.markdown("#### Principal exposure by investor")
            balances = transaction_balances(transactions)
            if balances.empty:
                st.info("No capital balances to report.")
            else:
                st.dataframe(balances.style.format({"Outstanding Capital (actual ledger)": "${:,.0f}"}),
                             use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════
# NAVIGATION
# ═══════════════════════════════════════════════════════════════
PAGES = {
    "📊 Overview": page_dashboard,
    "🏢 Borrowers": page_borrowers,
    "📄 Deals": page_deals,
    "🧮 Deal Workspace": page_modeler,
    "👥 Investor Portal": page_investor_portal,
}

with st.sidebar:
    st.markdown("# 🏦 Private Credit")
    st.caption("Portfolio and loan servicing")
    st.markdown("---")
    selection = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
    st.markdown("---")
    st.caption("Date-based model · Investor allocations · Live ledger")

try:
    PAGES[selection]()
except Exception as error:
    st.error(f"The page could not be loaded: {error}")
    st.exception(error)
