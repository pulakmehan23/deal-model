import calendar
import hashlib
import json
import sqlite3
from datetime import date, datetime, timedelta
from io import BytesIO
from pathlib import Path

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


def initial_advance_interest_amount(facility_amount, annual_rate, settlement, months):
    settlement = parse_date(settlement, date.today())
    months = max(0, int(months))
    return sum(max(0.0, float(facility_amount)) * annual_rate_for_days(
        float(annual_rate), max((add_months(settlement, month_index + 1)
        - add_months(settlement, month_index)).days, 1)) for month_index in range(months))


def capitalised_interest_for_period(principal, start_date, end_date, annual_rate, receipt_events=None):
    """Accrue ACT/365 capitalised interest on the changing balance between dated principal receipts."""
    start_date = parse_date(start_date)
    end_date = parse_date(end_date, start_date)
    if start_date is None or end_date is None or end_date <= start_date:
        return 0.0
    balance = max(0.0, float(principal))
    cursor = start_date
    total_interest = 0.0
    events = [event for event in (receipt_events or []) if isinstance(event, dict)]
    events.sort(key=lambda event: parse_date(event.get("receipt_date"), end_date))
    for event in events:
        event_date = parse_date(event.get("receipt_date"), end_date)
        if event_date < start_date or event_date > end_date:
            continue
        if event_date > cursor and balance > 0:
            accrued = balance * annual_rate_for_days(annual_rate, (event_date - cursor).days)
            balance += accrued
            total_interest += accrued
        principal_received = max(0.0, float(event.get("principal", 0.0) or 0.0))
        balance = max(0.0, balance - principal_received)
        cursor = max(cursor, event_date)
    if end_date > cursor and balance > 0:
        accrued = balance * annual_rate_for_days(annual_rate, (end_date - cursor).days)
        total_interest += accrued
    return total_interest


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


def safe_rate_pct(value, fallback=8.0):
    try:
        rate = float(value)
        if not np.isfinite(rate):
            raise ValueError("non-finite return")
    except (TypeError, ValueError):
        rate = float(fallback)
    if not np.isfinite(rate):
        rate = 8.0
    return max(0.0, min(50.0, rate))


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


FACILITY_TYPES = ["Construction", "Bridging", "Residual Stock"]


def normalize_facility_type(value, fallback="Bridging"):
    """Normalize legacy facility labels to the three supported facility types."""
    label = str(value or "").strip().lower()
    aliases = {
        "construction": "Construction",
        "bridge": "Bridging",
        "bridging": "Bridging",
        "residual stock": "Residual Stock",
    }
    return aliases.get(label, fallback)


PRINCIPAL_REPAYMENT_MODES = ["Standard amortising", "Bullet at maturity", "Custom schedule"]
PRINCIPAL_REPAYMENT_BASES = ["Dollar amount", "% of original tranche"]


def default_principal_repayment_mode(facility_type):
    return "Bullet at maturity" if normalize_facility_type(facility_type) == "Bridging" else "Standard amortising"


def parse_principal_repayment_rows(raw_schedule):
    """Normalize saved custom principal repayments to dated amount/basis records."""
    if isinstance(raw_schedule, str):
        raw_schedule = safe_json(raw_schedule, [])
    if isinstance(raw_schedule, dict):
        source_rows = []
        for due_date, value in raw_schedule.items():
            if isinstance(value, dict):
                source_rows.append({"due_date": due_date, **value})
            else:
                source_rows.append({"due_date": due_date, "amount": value, "basis": "Dollar amount"})
    elif isinstance(raw_schedule, list):
        source_rows = raw_schedule
    else:
        source_rows = []

    rows = []
    for item in source_rows:
        if not isinstance(item, dict):
            continue
        due_date = parse_date(item.get("due_date", item.get("date", item.get("Due Date"))))
        if due_date is None:
            continue
        amount_value = item.get("amount", item.get("principal_due", item.get("Amount", 0.0)))
        try:
            amount = float(amount_value)
            if not np.isfinite(amount) or amount <= 0:
                continue
        except (TypeError, ValueError):
            continue
        raw_basis = str(item.get("basis", item.get("amount_basis", item.get("Amount Basis", "Dollar amount"))))
        basis = ("% of original tranche" if raw_basis.strip().lower() in
            ["% of original tranche", "percent", "percentage", "%", "pct"] else "Dollar amount")
        rows.append({"due_date": due_date.isoformat(), "amount": amount, "basis": basis})
    return sorted(rows, key=lambda row: (row["due_date"], row["basis"], row["amount"]))

# ═══════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════
# Keep the database beside this source file, independent of the process working directory.
DB = Path(__file__).resolve().with_name("lms_data.db")


def get_conn():
    DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB), timeout=30, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def init_db():
    conn = get_conn()
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = FULL")
    conn.execute("""CREATE TABLE IF NOT EXISTS borrowers (
        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, abn TEXT, group_name TEXT,
        risk_grade TEXT DEFAULT 'Investment Grade', industry TEXT DEFAULT 'Real Estate',
        status TEXT DEFAULT 'Active')""")
    conn.execute("""CREATE TABLE IF NOT EXISTS deals (
        id INTEGER PRIMARY KEY AUTOINCREMENT, borrower_id INTEGER, deal_name TEXT,
        facility_name TEXT DEFAULT 'Tranche 1', parent_deal_id INTEGER,
        facility_amount REAL, deal_status TEXT DEFAULT 'Pipeline',
        settlement_date TEXT, maturity_date TEXT, facility_type TEXT DEFAULT 'Construction',
        security_type TEXT DEFAULT 'First Mortgage', property_value REAL DEFAULT 0,
        params_json TEXT DEFAULT '{}',
        FOREIGN KEY (borrower_id) REFERENCES borrowers(id),
        FOREIGN KEY (parent_deal_id) REFERENCES deals(id))""")
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
        senior_target_irr REAL DEFAULT 8.0, mezz_target_irr REAL DEFAULT 8.0,
        status TEXT DEFAULT 'Active', notes TEXT DEFAULT '')""")
    conn.execute("""CREATE TABLE IF NOT EXISTS deal_investors (
        deal_id INTEGER NOT NULL, investor_id INTEGER NOT NULL,
        senior_default_pct REAL DEFAULT 0, mezz_default_pct REAL DEFAULT 0,
        target_irr REAL, senior_target_irr REAL, mezz_target_irr REAL,
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
    conn.execute("""CREATE TABLE IF NOT EXISTS borrower_receipts (
        id INTEGER PRIMARY KEY AUTOINCREMENT, parent_deal_id INTEGER NOT NULL,
        receipt_date TEXT NOT NULL, amount REAL NOT NULL,
        interest_through_date TEXT, is_payoff INTEGER DEFAULT 0,
        notes TEXT DEFAULT '', created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (parent_deal_id) REFERENCES deals(id))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS borrower_receipt_allocations (
        receipt_id INTEGER NOT NULL, facility_deal_id INTEGER NOT NULL,
        period_date TEXT NOT NULL, fees REAL DEFAULT 0, penalty_interest REAL DEFAULT 0,
        interest REAL DEFAULT 0, principal REAL DEFAULT 0, override_flag INTEGER DEFAULT 0,
        PRIMARY KEY (receipt_id, facility_deal_id),
        FOREIGN KEY (receipt_id) REFERENCES borrower_receipts(id) ON DELETE CASCADE,
        FOREIGN KEY (facility_deal_id) REFERENCES deals(id))""")


    deal_cols = {row[1] for row in conn.execute("PRAGMA table_info(deals)").fetchall()}
    if "facility_name" not in deal_cols:
        conn.execute("ALTER TABLE deals ADD COLUMN facility_name TEXT DEFAULT 'Tranche 1'")
    if "parent_deal_id" not in deal_cols:
        conn.execute("ALTER TABLE deals ADD COLUMN parent_deal_id INTEGER")
    conn.execute("UPDATE deals SET facility_name='Tranche 1' WHERE facility_name IS NULL OR TRIM(facility_name)=''")
    for legacy_id, legacy_name in conn.execute("SELECT id, facility_name FROM deals").fetchall():
        legacy_parts = str(legacy_name or "").strip().split()
        if (len(legacy_parts) == 2 and legacy_parts[0].lower() == "facility"
                and legacy_parts[1].isdigit()):
            conn.execute("UPDATE deals SET facility_name=? WHERE id=?",
                (f"Tranche {int(legacy_parts[1])}", int(legacy_id)))
    conn.execute("UPDATE deals SET parent_deal_id=id WHERE parent_deal_id IS NULL")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_deals_parent ON deals(parent_deal_id)")

    deal_investor_cols = {row[1] for row in conn.execute("PRAGMA table_info(deal_investors)").fetchall()}
    if "target_irr" not in deal_investor_cols:
        conn.execute("ALTER TABLE deal_investors ADD COLUMN target_irr REAL")
    if "senior_target_irr" not in deal_investor_cols:
        conn.execute("ALTER TABLE deal_investors ADD COLUMN senior_target_irr REAL")
    if "mezz_target_irr" not in deal_investor_cols:
        conn.execute("ALTER TABLE deal_investors ADD COLUMN mezz_target_irr REAL")

    investor_cols = {row[1] for row in conn.execute("PRAGMA table_info(investors)").fetchall()}
    if "target_irr" not in investor_cols:
        conn.execute("ALTER TABLE investors ADD COLUMN target_irr REAL DEFAULT 8.0")
    if "senior_target_irr" not in investor_cols:
        conn.execute("ALTER TABLE investors ADD COLUMN senior_target_irr REAL")
    if "mezz_target_irr" not in investor_cols:
        conn.execute("ALTER TABLE investors ADD COLUMN mezz_target_irr REAL")
    conn.execute("""UPDATE investors SET
        senior_target_irr=COALESCE(senior_target_irr, target_irr, 8.0),
        mezz_target_irr=COALESCE(mezz_target_irr, target_irr, 8.0)""")
    conn.execute("""UPDATE deal_investors SET
        senior_target_irr=COALESCE(senior_target_irr,
            (SELECT senior_target_irr FROM investors WHERE investors.id=deal_investors.investor_id), 8.0),
        mezz_target_irr=COALESCE(mezz_target_irr,
            (SELECT mezz_target_irr FROM investors WHERE investors.id=deal_investors.investor_id), 8.0)""")
    transaction_cols = {row[1] for row in conn.execute("PRAGMA table_info(investor_transactions)").fetchall()}
    if "source" not in transaction_cols:
        conn.execute("ALTER TABLE investor_transactions ADD COLUMN source TEXT DEFAULT 'User'")

    # The three core pools are available on every deal. Optional investors such as
    # CBFM or Syndicate are added in the Investor Portal and attached deal-by-deal.
    for core_name in ("BPCFMC", "BPCFLC", "CBCF"):
        conn.execute("""INSERT OR IGNORE INTO investors
            (name, investor_type, target_irr, senior_target_irr, mezz_target_irr)
            VALUES (?, 'Fund', 8.0, 8.0, 8.0)""", (core_name,))

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

    # Seed one inspectable Bridging example exactly once. The marker/name lookup means
    # restarts and repeat Streamlit reruns never duplicate it or alter existing demo rows.
    demo_borrower_name = "DEMO ONLY — Returns & Tranches Borrower"
    demo_deal_name = "DEMO — Bridging $10m Bullet Facility"
    demo_settlement = date(2026, 11, 13)
    demo_maturity = date(2027, 11, 13)
    borrower_row = conn.execute("SELECT id FROM borrowers WHERE name=? ORDER BY id LIMIT 1",
                                (demo_borrower_name,)).fetchone()
    if borrower_row is None:
        conn.execute("""INSERT INTO borrowers (name, abn, group_name, risk_grade, industry, status)
            VALUES (?, '', 'Demo Portfolio', 'Investment Grade', 'Real Estate', 'Active')""",
            (demo_borrower_name,))
        borrower_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
    else:
        borrower_id = int(borrower_row[0])
    # The exact demo name is the durable idempotency key as well as the seed_key;
    # the workspace editor intentionally writes its canonical assumptions back and
    # may not retain metadata fields from the initial row.
    seeded = conn.execute(
        "SELECT 1 FROM deals WHERE borrower_id=? AND deal_name=? AND facility_name='Tranche 1' LIMIT 1",
        (borrower_id, demo_deal_name)).fetchone() is not None
    if not seeded:
        demo_params = {
            "seed_key": "demo_bridging_10m_bullet_v1",
            "facility_amount": 10_000_000.0, "facility_type": "Bridging",
            "property_value": 15_000_000.0,
            "settlement_date": demo_settlement.isoformat(),
            "maturity_date": demo_maturity.isoformat(), "term_months": 12,
            "loan_ranking": "Senior", "borrower_irr": 12.0, "cpr": 0.0,
            "principal_repayment_mode": "Bullet at maturity",
            "principal_repayment_schedule": [],
            "interest_treatment": "Serviced", "interest_advance_months": 1,
            "construction_months": 0, "interest_only_months": 0,
            "drawdown_profile": "Automatic", "s_curve_steepness": 5.0,
            "estab_fee_pct": 2.0, "line_fee_pct": 0.5,
            "admin_fee_monthly": 0.0, "discharge_fee": 0.0,
            "risk_review_fee": 0.0, "legal_fee": 0.0,
            "other_upfront_fees": 0.0, "restructure_fee": 0.0,
            "pv_growth_annual": 3.0, "covenant_lvr": 75.0,
            "default_investor_target_irr": 8.0,
            "drawdown_schedule": json.dumps({demo_settlement.isoformat(): {
                "gross_draw": 9_800_000.0, "allocation_source": "Standard",
                "status": "Forecast", "allocations": {}}}),
        }
        conn.execute("""INSERT INTO deals (borrower_id, deal_name, facility_name, parent_deal_id,
            facility_amount, deal_status, settlement_date, maturity_date, facility_type,
            security_type, property_value, params_json)
            VALUES (?, ?, 'Tranche 1', NULL, ?, 'Active', ?, ?, 'Bridging',
                    'First Mortgage', ?, ?)""",
            (borrower_id, demo_deal_name, 10_000_000.0, demo_settlement.isoformat(),
             demo_maturity.isoformat(), 15_000_000.0, json.dumps(demo_params)))
        demo_deal_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        conn.execute("UPDATE deals SET parent_deal_id=? WHERE id=?",
                     (demo_deal_id, demo_deal_id))
        core_defaults = {"BPCFMC": (50.0, 80.0), "BPCFLC": (15.0, 20.0), "CBCF": (35.0, 0.0)}
        for core_name, (senior_pct, mezz_pct) in core_defaults.items():
            investor = conn.execute("SELECT id, target_irr, senior_target_irr, mezz_target_irr FROM investors WHERE name=?",
                                    (core_name,)).fetchone()
            if investor:
                conn.execute("""INSERT OR IGNORE INTO deal_investors
                    (deal_id, investor_id, senior_default_pct, mezz_default_pct, target_irr,
                     senior_target_irr, mezz_target_irr, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 1)""",
                    (demo_deal_id, int(investor[0]), senior_pct, mezz_pct,
                     float(investor[1] if investor[1] is not None else 8.0),
                     safe_rate_pct(investor[2], 8.0), safe_rate_pct(investor[3], 8.0)))

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
    facility_type = normalize_facility_type(facility_type)
    requested_profile = str(requested_profile or "Automatic")
    # Staged construction profiles (including S-curve) are only valid for Construction.
    if facility_type != "Construction":
        return "Full at settlement"
    if requested_profile == "Automatic":
        return "S-curve"
    if requested_profile in ["Full at settlement", "S-curve", "Straight-line"]:
        return requested_profile
    return "S-curve"


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
    payload = "|".join(map(str, [deal_id, round(float(facility_amount), 2), normalize_facility_type(facility_type),
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
        investor_funding_required = max(0.0, row["Gross Drawdown"] + upfront)
        key = row["Drawdown Date"].isoformat()
        if row["Allocation Source"] == "Standard":
            allocations = default_allocation_amounts(investor_funding_required, loan_ranking, investor_names)
        elif row["Allocation Source"] == "Investor Ledger":
            allocations = {name: float(actual_calls.get(key, {}).get(name, 0.0)) for name in investor_names}
        else:
            allocations = {name: float(row["_prior_allocations"].get(name, 0.0) or 0.0)
                           for name in investor_names}
            if not any(allocations.values()) and key in actual_calls:
                allocations = {name: float(actual_calls[key].get(name, 0.0)) for name in investor_names}
        for name in investor_names:
            row[allocation_column(name)] = max(0.0, allocations.get(name, 0.0))
        row["_net_to_allocate"] = investor_funding_required
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
        investor_funding_required = max(0.0, gross + upfront)
        source = str(row.get("Allocation Source", "Standard"))
        status = str(row.get("Status", "Forecast"))
        if source == "Standard":
            allocations = default_allocation_amounts(investor_funding_required, loan_ranking, investor_names)
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
            "Gross Drawdown": gross, "Upfront Costs Funded": upfront,
            "Investor Funding Required": investor_funding_required}
        for name in investor_names:
            summary[name] = allocations.get(name, 0.0)
        summary["Allocation Difference"] = investor_funding_required - allocated
        display_rows.append(summary)
    return result, pd.DataFrame(display_rows)


def compute_fees(params):
    facility_amount = float(params.get("facility_amount", 0))
    term = int(params.get("term_months", 12))
    construction_months = int(params.get("construction_months", 6))
    facility_type = normalize_facility_type(params.get("facility_type", "Construction"))
    is_construction = facility_type == "Construction"
    establishment = facility_amount * float(params.get("estab_fee_pct", 1.5)) / 100.0
    # Undrawn line fees only apply to Construction facilities.
    line_rate = float(params.get("line_fee_pct", 0.5)) / 100.0 if is_construction else 0.0
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
    construction_months = max(0, int(params.get("construction_months", 6)))
    interest_only_months = max(0, min(term, int(params.get(
        "interest_only_months", params.get("construction_months", 6)))))
    settlement = parse_date(params.get("settlement_date"), date.today())
    annual_rate = float(params.get("borrower_irr", 12.0)) / 100.0
    cpr = float(params.get("cpr", 0.0)) / 100.0
    property_value = float(params.get("property_value", 0.0))
    property_growth = float(params.get("pv_growth_annual", 3.0)) / 100.0
    covenant_lvr = float(params.get("covenant_lvr", 75.0))
    facility_type = normalize_facility_type(params.get("facility_type", "Construction"))
    default_repayment_mode = default_principal_repayment_mode(facility_type)
    principal_repayment_mode = str(params.get("principal_repayment_mode", default_repayment_mode))
    if principal_repayment_mode not in PRINCIPAL_REPAYMENT_MODES:
        principal_repayment_mode = default_repayment_mode
    custom_principal_by_date = {}
    if principal_repayment_mode == "Custom schedule":
        valid_payment_dates = {add_months(settlement, month).isoformat() for month in range(1, term + 1)}
        for repayment in parse_principal_repayment_rows(params.get("principal_repayment_schedule", [])):
            due_date = repayment["due_date"]
            if due_date not in valid_payment_dates:
                continue
            amount = float(repayment["amount"])
            if repayment["basis"] == "% of original tranche":
                amount = facility_amount * amount / 100.0
            custom_principal_by_date[due_date] = custom_principal_by_date.get(due_date, 0.0) + max(0.0, amount)
    fees = compute_fees(params)
    raw_interest_treatment = str(params.get("interest_treatment", "Serviced")).strip().lower()
    interest_treatment = "Capitalised" if raw_interest_treatment in ["capitalised", "capitalized"] else "Serviced"
    # Only capitalised fees/interest consume the peak-balance cap. For the
    # Capitalised scenario, reserve scheduled charges before allocating interest capacity.
    projected_fee_cap_reserve = ((fees["admin_total_estimate"] + fees["line_fee_estimate"]
        + fees["discharge"]) if interest_treatment == "Capitalised" else 0.0)
    advance_setting = params.get("interest_advance_months", 1)
    all_interest_in_advance = str(advance_setting).strip().lower() == "all"
    try:
        advance_months = term if all_interest_in_advance else max(1, min(term, int(advance_setting)))
    except (TypeError, ValueError):
        advance_months = 1
    capitalisation_interval = advance_months if interest_treatment == "Capitalised" else 1
    first_planned_draw = 0.0
    for plan_key, plan_value in sorted(drawdown_map.items()):
        if isinstance(plan_value, dict):
            candidate_draw = max(0.0, float(plan_value.get("gross_draw", plan_value.get("amount", 0.0)) or 0.0))
        else:
            candidate_draw = max(0.0, float(plan_value or 0.0))
        if candidate_draw > 0:
            first_planned_draw = candidate_draw
            break
    initial_advance_interest = (initial_advance_interest_amount(
        first_planned_draw or facility_amount, annual_rate, settlement, advance_months)
        if interest_treatment == "Serviced" else 0.0)
    net_advance_interest_from_proceeds = (interest_treatment == "Serviced"
        and all_interest_in_advance
        and facility_type in ["Bridging", "Residual Stock"])

    rows = []
    balance = 0.0
    cumulative_drawn = 0.0
    cumulative_upfront_funded = 0.0
    cumulative_cap_fee_charges = 0.0
    cumulative_interest_charged = 0.0
    cumulative_borrower_principal_repaid = 0.0
    cumulative_upfront_fees_repaid = 0.0
    cumulative_interest = 0.0
    capitalised_interest_pending = 0.0
    arrears = 0.0
    first_draw_done = False
    borrower_cashflows = []
    payoff_dates = []
    for actual_item in actuals_map.values():
        for event in actual_item.get("_receipt_events", []) if isinstance(actual_item, dict) else []:
            if isinstance(event, dict) and bool(event.get("is_payoff", False)):
                payoff_date = parse_date(event.get("receipt_date"))
                if payoff_date is not None:
                    payoff_dates.append(payoff_date)
    facility_payoff_date = min(payoff_dates) if payoff_dates else None

    for index in range(term):
        draw_date = add_months(settlement, index)
        after_payoff = facility_payoff_date is not None and draw_date > facility_payoff_date
        payment_date = add_months(settlement, index + 1)
        days = max((payment_date - draw_date).days, 1)
        actual = actuals_map.get(payment_date.isoformat(), {})
        raw_period_events = actual.get("_receipt_events", []) if isinstance(actual, dict) else []
        period_receipt_events = ([event for event in raw_period_events if isinstance(event, dict)]
                                 if isinstance(raw_period_events, list) else [])
        period_payoff_dates = [parse_date(event.get("receipt_date"), payment_date)
                               for event in period_receipt_events if bool(event.get("is_payoff", False))]
        period_payoff_date = min(period_payoff_dates) if period_payoff_dates else None
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
        if after_payoff:
            planned_draw = 0.0
            investor_allocations = {}
            allocation_source = "Standard"
            drawdown_status = "Forecast"
            gross_draw = 0.0
        elif actual.get("drawdown_override", False):
            gross_draw = max(0.0, float(actual.get("actual_drawdown", 0.0)))
        else:
            gross_draw = planned_draw

        opening_balance = balance
        actual_draw_override = bool(actual.get("drawdown_override", False))
        if interest_treatment == "Serviced":
            # Cash-serviced interest and cash-billed fees do not enlarge peak outstanding.
            cap_room_before_draw = max(0.0, facility_amount - opening_balance)
        else:
            fees_reserved_for_draw = max(projected_fee_cap_reserve, cumulative_cap_fee_charges)
            cap_room_before_draw = max(0.0, facility_amount - cumulative_drawn
                - cumulative_upfront_funded - fees_reserved_for_draw - cumulative_interest_charged)
        if not actual_draw_override:
            gross_draw = min(gross_draw, cap_room_before_draw)
        upfront_deducted = 0.0
        is_first_draw = gross_draw > 0 and not first_draw_done
        if is_first_draw:
            if interest_treatment == "Serviced":
                upfront_room = max(0.0, facility_amount - opening_balance - gross_draw)
            else:
                upfront_room = max(0.0, facility_amount - cumulative_drawn - gross_draw
                    - cumulative_upfront_funded - max(projected_fee_cap_reserve, cumulative_cap_fee_charges)
                    - cumulative_interest_charged)
            upfront_deducted = min(fees["upfront_total"], upfront_room)
            first_draw_done = True
        investor_funding = max(0.0, gross_draw + upfront_deducted)
        advance_interest_withheld = (min(initial_advance_interest, investor_funding)
            if index == 0 and is_first_draw and net_advance_interest_from_proceeds else 0.0)
        net_draw = max(0.0, gross_draw - advance_interest_withheld)
        balance_after_draw = opening_balance + gross_draw + upfront_deducted
        cumulative_drawn += gross_draw
        cumulative_upfront_funded += upfront_deducted

        # Capitalised-treatment fees consume capacity inside the peak/outstanding cap.
        arrears_opening = arrears
        in_construction = facility_type == "Construction" and index < construction_months
        in_interest_only = index < interest_only_months
        undrawn = max(0.0, facility_amount - cumulative_drawn - cumulative_upfront_funded
            - cumulative_cap_fee_charges - cumulative_interest_charged)
        line_fee = (undrawn * fees["line_fee_rate"] * (days / 365.0)
                    if in_construction and not after_payoff else 0.0)
        admin_fee = fees["admin_monthly"] if not after_payoff else 0.0
        discharge_fee = fees["discharge"] if index == term - 1 and not after_payoff else 0.0
        current_cap_fee_charges = line_fee + admin_fee + discharge_fee

        if after_payoff:
            interest_accrued = 0.0
        elif interest_treatment == "Capitalised":
            if period_payoff_date:
                # A payoff terminates accrual on its actual date. Partial principal receipts
                # are otherwise applied to forecast balances at the payment anniversary.
                interest_accrued = capitalised_interest_for_period(
                    balance_after_draw, draw_date, min(payment_date, period_payoff_date),
                    annual_rate, period_receipt_events)
            else:
                # Keep forecast-period interest through the scheduled anniversary; actual
                # principal receipts reduce the balance for the next forecast period.
                interest_accrued = balance_after_draw * annual_rate_for_days(annual_rate, days)
        else:
            interest_principal = (max(0.0, cumulative_drawn)
                if index < advance_months else balance_after_draw)
            interest_accrued = interest_principal * annual_rate_for_days(annual_rate, days)
        gross_interest_accrued = max(0.0, float(interest_accrued))
        interest_cap_shortfall = 0.0
        if interest_treatment == "Capitalised" and not bool(actual.get("payment_actuals_logged", False)):
            fees_reserved_within_cap = max(projected_fee_cap_reserve,
                cumulative_cap_fee_charges + current_cap_fee_charges)
            available_for_interest = max(0.0, facility_amount - cumulative_drawn
                - cumulative_upfront_funded - fees_reserved_within_cap - cumulative_interest_charged)
            interest_accrued = min(interest_accrued, available_for_interest)
            interest_cap_shortfall = max(0.0, gross_interest_accrued - interest_accrued)
        if after_payoff or interest_treatment == "Capitalised":
            interest_due = 0.0
        elif index == 0:
            interest_due = initial_advance_interest
        elif index < advance_months:
            interest_due = 0.0
        else:
            interest_due = interest_accrued
        if interest_treatment == "Capitalised":
            cumulative_interest_charged += interest_accrued
        cumulative_cap_fee_charges += current_cap_fee_charges
        if interest_treatment == "Capitalised":
            capitalised_interest_pending += interest_accrued
            capitalize_now = ((index + 1) % capitalisation_interval == 0
                or index == term - 1 or period_payoff_date is not None)
            capitalised_interest = capitalised_interest_pending if capitalize_now else 0.0
            if capitalize_now:
                capitalised_interest_pending = 0.0
        else:
            capitalised_interest = 0.0
        balance_after_capitalisation = balance_after_draw + capitalised_interest
        principal_base = balance_after_capitalisation

        smm = 1.0 - (1.0 - cpr) ** (days / 365.0) if cpr > 0 else 0.0
        prepayment = principal_base * smm if not after_payoff and not in_interest_only and principal_base > 0 else 0.0
        if after_payoff or principal_repayment_mode == "Bullet at maturity":
            scheduled_principal = 0.0
        elif principal_repayment_mode == "Custom schedule":
            custom_due = custom_principal_by_date.get(payment_date.isoformat(), 0.0)
            scheduled_principal = min(max(0.0, principal_base - prepayment), custom_due)
        elif in_interest_only:
            scheduled_principal = 0.0
        else:
            remaining_payments = max(term - index, 1)
            scheduled_principal = principal_base / remaining_payments
        maturity_repayment = 0.0
        if index == term - 1 and not after_payoff and not period_payoff_dates:
            maturity_repayment = max(0.0, principal_base - min(principal_base, scheduled_principal + prepayment))
        contract_principal = min(principal_base,
            scheduled_principal + prepayment + maturity_repayment)

        has_payment_actuals = bool(actual.get("payment_actuals_logged", False))
        actual_interest = max(0.0, float(actual.get("actual_interest", 0.0) or 0.0)) if has_payment_actuals else None
        actual_principal = max(0.0, float(actual.get("actual_principal", 0.0) or 0.0)) if has_payment_actuals else None
        actual_fees = max(0.0, float(actual.get("actual_fees", 0.0) or 0.0)) if has_payment_actuals else None
        actual_penalty_interest = (max(0.0, float(actual.get("actual_penalty_interest", 0.0) or 0.0))
                                   if has_payment_actuals else None)
        servicing_interest_due = interest_due
        if has_payment_actuals:
            # Serviced payoff interest is charged through the payoff date, preserving any prepaid minimum.
            events = list(period_receipt_events)
            accrual_end = min([payment_date] + period_payoff_dates) if period_payoff_dates else payment_date
            if period_payoff_dates and interest_treatment == "Serviced":
                days_to_payoff = max((accrual_end - draw_date).days, 0)
                payoff_principal = facility_amount if index < advance_months else balance_after_draw
                partial_accrual = payoff_principal * annual_rate_for_days(annual_rate, days_to_payoff)
                required_through_payoff = max(initial_advance_interest, cumulative_interest + partial_accrual)
                baseline_before_period = (0.0 if index == 0
                    else max(initial_advance_interest, cumulative_interest))
                servicing_interest_due = max(0.0, required_through_payoff - baseline_before_period)
                interest_accrued = partial_accrual
                interest_due = servicing_interest_due
            cash_servicing_interest_due = max(0.0, servicing_interest_due - advance_interest_withheld)
            # Charge ACT/365 interest on overdue serviced interest only, stopping at receipts.
            accrual_balance = arrears_opening + cash_servicing_interest_due
            accrual_cursor = draw_date
            penalty_interest = 0.0
            event_interest_total = sum(max(0.0, float(event.get("interest", 0.0) or 0.0)) for event in events)
            manual_remainder = max(0.0, float(actual_interest or 0.0) - event_interest_total)
            if manual_remainder > 0.01:
                events.append({"receipt_date": payment_date, "interest": manual_remainder})
            if not events:
                events = [{"receipt_date": payment_date, "interest": float(actual_interest or 0.0)}]
            events.sort(key=lambda event: parse_date(event.get("receipt_date"), payment_date))
            for event in events:
                event_date = parse_date(event.get("receipt_date"), payment_date)
                event_date = max(draw_date, min(event_date, accrual_end))
                if event_date > accrual_cursor and accrual_balance > 0:
                    penalty_interest += accrual_balance * annual_rate_for_days(
                        annual_rate, (event_date - accrual_cursor).days)
                accrual_cursor = max(accrual_cursor, event_date)
                accrual_balance = max(0.0, accrual_balance - max(0.0, float(event.get("interest", 0.0) or 0.0)))
                if accrual_cursor >= accrual_end:
                    break
            if accrual_end > accrual_cursor and accrual_balance > 0:
                penalty_interest += accrual_balance * annual_rate_for_days(
                    annual_rate, (accrual_end - accrual_cursor).days)
            # Penalty charges are separate; prepaid interest withheld at settlement is already paid.
            arrears = max(0.0, arrears_opening + cash_servicing_interest_due - float(actual_interest or 0.0))
            principal_applied = min(float(actual_principal or 0.0), principal_base)
            cash_interest_received = float(actual_interest or 0.0) + float(actual_penalty_interest or 0.0)
            cash_principal_received = float(actual_principal or 0.0)
            cash_fees_received = float(actual_fees or 0.0)
        else:
            penalty_interest = arrears_opening * annual_rate_for_days(annual_rate, days) if arrears_opening > 0 else 0.0
            cash_servicing_interest_due = max(0.0, servicing_interest_due - advance_interest_withheld)
            principal_applied = contract_principal
            cash_interest_advance = (cash_servicing_interest_due
                if interest_treatment == "Serviced" else 0.0)
            cash_interest_received = cash_interest_advance + penalty_interest
            cash_principal_received = contract_principal
            cash_fees_received = line_fee + admin_fee + discharge_fee
        if has_payment_actuals:
            cash_interest_advance = 0.0
            cash_principal_received = float(actual_principal or 0.0)
            cash_capitalised_interest_received = 0.0
            cumulative_borrower_principal_repaid += cash_principal_received
            cash_upfront_fee_received = 0.0
        else:
            borrower_principal_remaining = max(0.0,
                cumulative_drawn - cumulative_borrower_principal_repaid)
            cash_principal_received = min(contract_principal, borrower_principal_remaining)
            after_principal = max(0.0, contract_principal - cash_principal_received)
            upfront_fee_remaining = max(0.0,
                cumulative_upfront_funded - cumulative_upfront_fees_repaid)
            cash_upfront_fee_received = min(after_principal, upfront_fee_remaining)
            cash_capitalised_interest_received = max(0.0,
                after_principal - cash_upfront_fee_received)
            cumulative_borrower_principal_repaid += cash_principal_received
            cumulative_upfront_fees_repaid += cash_upfront_fee_received
            cash_fees_received += cash_upfront_fee_received
            cash_interest_received += cash_capitalised_interest_received
        # Advances are collected on the draw date, not again on the anniversary payment date.
        # Total_Due is the amount due on Payment_Date; Cash_Interest_Advance is shown separately.
        interest_due_on_payment_date = max(0.0, cash_servicing_interest_due - cash_interest_advance)
        total_due = (arrears_opening + interest_due_on_payment_date + penalty_interest + contract_principal
                     + line_fee + admin_fee + discharge_fee)

        ending_balance = max(0.0, balance_after_capitalisation - principal_applied)
        elapsed_days = max((payment_date - settlement).days, 0)
        current_property_value = (property_value * ((1.0 + property_growth) ** (elapsed_days / 365.0))
                                 if property_value > 0 else 0.0)
        lvr = ending_balance / current_property_value * 100.0 if current_property_value > 0 else 0.0
        breach = lvr > covenant_lvr
        cumulative_interest += interest_accrued

        if net_draw > 0:
            borrower_cashflows.append((draw_date, -net_draw))
        if has_payment_actuals:
            actual_events = actual.get("_receipt_events", [])
            event_total = 0.0
            for event in actual_events if isinstance(actual_events, list) else []:
                event_date = parse_date(event.get("receipt_date"), payment_date)
                event_amount = sum(max(0.0, float(event.get(key, 0.0) or 0.0))
                                   for key in ("interest", "penalty_interest", "principal", "fees"))
                if event_amount > 0:
                    borrower_cashflows.append((event_date, event_amount))
                    event_total += event_amount
            manual_total = max(0.0, sum(max(0.0, float(value or 0.0)) for value in
                (actual_interest, actual_penalty_interest, actual_principal, actual_fees)) - event_total)
            if manual_total > 0:
                borrower_cashflows.append((payment_date, manual_total))
        else:
            if cash_interest_advance > 0:
                borrower_cashflows.append((draw_date, cash_interest_advance))
            payment_cashflow = penalty_interest + contract_principal + line_fee + admin_fee + discharge_fee
            if payment_cashflow > 0:
                borrower_cashflows.append((payment_date, payment_cashflow))

        rows.append({
            "Drawdown_Date": draw_date,
            "Interest_Due_Date": draw_date,
            "Payment_Date": payment_date,
            "Period_Days": days,
            "Planned_Drawdown": planned_draw,
            "Allocation_Source": allocation_source,
            "Drawdown_Status": drawdown_status,
            "Investor_Allocations": investor_allocations,
            "Drawdown_Gross": gross_draw,
            "Upfront_Fees_Deducted": upfront_deducted,
            "Advance_Interest_Withheld": advance_interest_withheld,
            "Investor_Funding": investor_funding,
            "Drawdown_Net": net_draw,
            "Opening_Balance": opening_balance,
            "Balance_After_Draw": balance_after_draw,
            "Capitalised_Interest": capitalised_interest,
            "Balance_After_Capitalisation": balance_after_capitalisation,
            "Interest_Accrued": interest_accrued,
            "Gross_Interest_Accrued": gross_interest_accrued,
            "Interest_Cap_Shortfall": interest_cap_shortfall,
            "Interest_Due": interest_due,
            "Penalty_Interest": penalty_interest,
            "Line_Fee": line_fee,
            "Admin_Fee": admin_fee,
            "Discharge_Fee": discharge_fee,
            "Scheduled_Principal": scheduled_principal,
            "Prepayment": prepayment,
            "Maturity_Repayment": maturity_repayment,
            "Contract_Principal_Due": contract_principal,
            "Principal_Applied": principal_applied,
            "Total_Due": total_due,
            "Arrears_Opening": arrears_opening,
            "Servicing_Interest_Due": servicing_interest_due,
            "Total_Interest_Due": arrears_opening + cash_servicing_interest_due + penalty_interest,
            "Actual_Interest": actual_interest,
            "Actual_Penalty_Interest": actual_penalty_interest,
            "Actual_Principal": actual_principal,
            "Actual_Fees": actual_fees,
            "Cash_Interest_Received": cash_interest_received,
            "Cash_Penalty_Interest_Received": float(actual_penalty_interest or 0.0) if has_payment_actuals else penalty_interest,
            "Cash_Interest_Advance": cash_interest_advance,
            "Cash_Capitalised_Interest_Received": cash_capitalised_interest_received,
            "Cash_Arrears_Interest": float(actual_penalty_interest or 0.0) if has_payment_actuals else penalty_interest,
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
            "Is_Interest_Only": in_interest_only,
        })
        balance = ending_balance

    return pd.DataFrame(rows), fees, calc_xirr(borrower_cashflows), borrower_cashflows

# ═══════════════════════════════════════════════════════════════
# INVESTOR ALLOCATION & LEDGER
# ═══════════════════════════════════════════════════════════════
def build_investor_ledger(schedule_df, params, investors_df, transactions_df=None,
                          borrower_receipt_allocations_df=None):
    """Build investor economics using each investor's rate for the tranche ranking."""
    transactions_df = transactions_df if transactions_df is not None else pd.DataFrame()
    borrower_receipt_allocations_df = (borrower_receipt_allocations_df
        if borrower_receipt_allocations_df is not None else pd.DataFrame())
    receipt_dates = set()
    receipt_events_by_period = {}
    if not borrower_receipt_allocations_df.empty:
        for _, receipt_row in borrower_receipt_allocations_df.iterrows():
            receipt_date = parse_date(receipt_row.get("receipt_date"))
            period_date = parse_date(receipt_row.get("period_date"))
            if receipt_date is not None:
                receipt_dates.add(receipt_date.isoformat())
            if period_date is not None:
                receipt_events_by_period.setdefault(period_date.isoformat(), []).append(receipt_row)
    investor_names = investors_df["name"].astype(str).tolist() if not investors_df.empty else []
    loan_ranking = str(params.get("loan_ranking", "Senior"))
    return_column = "senior_target_irr" if loan_ranking == "Senior" else "mezz_target_irr"
    try:
        default_target_pct = float(params.get("default_investor_target_irr", 8.0))
        if not np.isfinite(default_target_pct):
            default_target_pct = 8.0
    except (TypeError, ValueError):
        default_target_pct = 8.0
    return_rates = params.get("investor_return_rates", {})
    return_rates = return_rates if isinstance(return_rates, dict) else {}
    return_overrides = params.get("investor_return_overrides", {})
    return_overrides = return_overrides if isinstance(return_overrides, dict) else {}
    target_rates = {}
    if not investors_df.empty:
        for _, investor_row in investors_df.iterrows():
            name = str(investor_row["name"])
            investor_id = investor_row.get("id", None)
            override = return_overrides.get(str(investor_id), return_overrides.get(investor_id, {}))
            override = override if isinstance(override, dict) else {}
            override_rate = override.get("senior" if loan_ranking == "Senior" else "mezzanine")
            deal_return_column = "deal_senior_target_irr" if loan_ranking == "Senior" else "deal_mezz_target_irr"
            row_rate = investor_row.get(deal_return_column,
                investor_row.get(return_column, investor_row.get("target_irr", default_target_pct)))
            candidate = return_rates.get(name, override_rate if override_rate is not None else row_rate)
            try:
                target_pct = float(candidate if candidate is not None else default_target_pct)
                if not np.isfinite(target_pct):
                    target_pct = default_target_pct
            except (TypeError, ValueError):
                target_pct = default_target_pct
            target_rates[name] = target_pct / 100.0

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
    default_target_rate = default_target_pct / 100.0
    for name in investor_names:
        target_rates.setdefault(name, default_target_rate)

    raw_interest_treatment = str(params.get("interest_treatment", "Serviced")).strip().lower()
    investor_interest_capitalised = raw_interest_treatment in {"capitalised", "capitalized"}
    try:
        capitalisation_interval = int(params.get("interest_advance_months", 1))
    except (TypeError, ValueError):
        # "All" is a valid Serviced-interest setting; if legacy data combines it
        # with Capitalised treatment, roll investor interest up at maturity.
        capitalisation_interval = max(1, len(schedule_df))
    capitalisation_interval = max(1, min(max(1, len(schedule_df)), capitalisation_interval))

    balances = {name: 0.0 for name in investor_names}
    # Original investor principal is tracked separately from capitalised return so a
    # maturity payoff can report principal back and interest paid as distinct amounts.
    principal_basis = {name: 0.0 for name in investor_names}
    capitalised_interest_pending = {name: 0.0 for name in investor_names}
    total_funded = {name: 0.0 for name in investor_names}
    total_interest = {name: 0.0 for name in investor_names}
    total_principal_returned = {name: 0.0 for name in investor_names}
    cashflows = {name: [] for name in investor_names}
    cashflow_events = []
    period_events = [[] for _ in range(len(schedule_df))]

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
            if tx_type in ["Principal Repayment", "Interest Distribution"] and tx_date.isoformat() in receipt_dates:
                continue  # Borrower receipt allocations already generate these investor-side actuals.
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
        investor_funding = float(schedule_row.get("Investor_Funding", net_draw) or 0.0)
        raw_allocations = schedule_row.get("Investor_Allocations", {})
        if isinstance(raw_allocations, str):
            raw_allocations = safe_json(raw_allocations, {})
        raw_allocations = raw_allocations if isinstance(raw_allocations, dict) else {}
        source = str(schedule_row.get("Allocation_Source", "Standard"))
        allocations = {name: max(0.0, float(raw_allocations.get(name, 0.0) or 0.0))
                       for name in investor_names}
        if investor_funding > 0 and source == "Standard":
            allocations = {name: investor_funding * standard_split_for(name, loan_ranking) / 100.0
                           for name in investor_names}

        period_capital_called = {name: allocations.get(name, 0.0) for name in investor_names}
        period_other_increase = {name: 0.0 for name in investor_names}
        period_other_decrease = {name: 0.0 for name in investor_names}
        period_principal_returns = {name: 0.0 for name in investor_names}
        period_interest_actual = {name: 0.0 for name in investor_names}
        period_interest_paid = {name: 0.0 for name in investor_names}
        period_capitalised_interest = {name: 0.0 for name in investor_names}
        actual_interest_names = set()
        period_buyout_in = {name: 0.0 for name in investor_names}
        period_buyout_out = {name: 0.0 for name in investor_names}
        has_actual_principal = False

        def register_principal_repayment(name, flow_date, amount, flow_type):
            """Separate returned contributed capital from repayment of capitalised return."""
            amount = max(0.0, float(amount or 0.0))
            if amount <= 0:
                return
            principal_basis.setdefault(name, 0.0)
            if investor_interest_capitalised:
                principal_component = min(amount, max(0.0, principal_basis[name]))
                interest_component = max(0.0, amount - principal_component)
                principal_basis[name] = max(0.0, principal_basis[name] - principal_component)
                if principal_component > 0:
                    period_principal_returns[name] = period_principal_returns.get(name, 0.0) + principal_component
                    record_flow(name, flow_date, flow_type, principal_component, principal_component)
                if interest_component > 0:
                    period_interest_paid[name] = period_interest_paid.get(name, 0.0) + interest_component
                    record_flow(name, flow_date, "Interest Distribution", interest_component, interest_component)
            else:
                period_principal_returns[name] = period_principal_returns.get(name, 0.0) + amount
                record_flow(name, flow_date, flow_type, amount, amount)

        for name, amount in allocations.items():
            balances[name] = balances.get(name, 0.0) + amount
            principal_basis[name] = principal_basis.get(name, 0.0) + amount
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
                    target_rates.get(investor_name, default_target_rate), days)
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
                period_interest_paid[investor_name] = period_interest_paid.get(investor_name, 0.0) + amount
                record_flow(investor_name, tx_date, "Interest Distribution", amount, amount)
            elif tx_type == "Principal Repayment":
                has_actual_principal = True
                returned = min(amount, max(balances[investor_name], 0.0))
                balances[investor_name] = max(0.0, balances[investor_name] - returned)
                register_principal_repayment(investor_name, tx_date, returned, "Principal Repayment")
            elif tx_type == "Other Increase":
                balances[investor_name] += amount
                principal_basis[investor_name] = principal_basis.get(investor_name, 0.0) + amount
                total_funded[investor_name] += amount
                period_other_increase[investor_name] += amount
                record_flow(investor_name, tx_date, "Other Increase", amount, -amount)
            elif tx_type == "Other Decrease":
                decreased = min(amount, max(balances[investor_name], 0.0))
                balances[investor_name] = max(0.0, balances[investor_name] - decreased)
                principal_basis[investor_name] = max(0.0, principal_basis.get(investor_name, 0.0) - decreased)
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
                principal_basis.setdefault(counterparty, 0.0)
                transferred = min(amount, max(balances[investor_name], 0.0))
                principal_transferred = min(transferred, max(0.0, principal_basis.get(investor_name, 0.0)))
                principal_basis[investor_name] = max(0.0, principal_basis.get(investor_name, 0.0) - principal_transferred)
                principal_basis[counterparty] += principal_transferred
                balances[investor_name] = max(0.0, balances[investor_name] - transferred)
                balances[counterparty] += transferred
                period_buyout_out[investor_name] += transferred
                period_buyout_in[counterparty] += transferred
                record_flow(investor_name, tx_date, "Buyout In", transferred, transferred)
                record_flow(counterparty, tx_date, "Buyout Out", transferred, -transferred)

        accrue_interest(payment_date)

        interest_by_investor = {}
        schedule_actual_interest = schedule_row.get("Actual_Interest")
        has_schedule_actual_interest = pd.notna(schedule_actual_interest)
        period_receipts = receipt_events_by_period.get(payment_date.isoformat(), [])
        schedule_actual_penalty = schedule_row.get("Actual_Penalty_Interest")
        if has_schedule_actual_interest:
            # Investor yield is accrued at their ranking-specific target return. In Serviced
            # mode, cash paid is capped by borrower interest actually collected; in
            # Capitalised mode, actual borrower interest is not passed through as cash.
            for name in list(balances):
                interest_by_investor[name] = accrued_interest.get(name, 0.0)

            if not investor_interest_capitalised:
                actual_pool = max(0.0, float(schedule_actual_interest))
                if pd.notna(schedule_actual_penalty):
                    actual_pool += max(0.0, float(schedule_actual_penalty))
                explicit = {name: max(0.0, period_interest_actual.get(name, 0.0))
                            for name in actual_interest_names}
                remaining_pool = max(0.0, actual_pool - sum(explicit.values()))
                eligible = [name for name in balances if name not in explicit]
                weights = {name: max(0.0, accrued_interest.get(name, 0.0)) for name in eligible}
                weight_total = sum(weights.values())
                if weight_total <= 0:
                    weights = {name: max(0.0, balances.get(name, 0.0)) for name in eligible}
                    weight_total = sum(weights.values())
                receipt_amounts = [max(0.0, float(row.get("interest", 0.0) or 0.0))
                    + max(0.0, float(row.get("penalty_interest", 0.0) or 0.0)) for row in period_receipts]
                receipt_interest_total = sum(receipt_amounts)
                receipt_pool = min(receipt_interest_total, remaining_pool)
                manual_pool = max(0.0, remaining_pool - receipt_pool)
                for name in eligible:
                    investor_share = weights.get(name, 0.0) / weight_total if weight_total > 0 else 0.0
                    distribution = remaining_pool * investor_share
                    if distribution <= 0:
                        continue
                    period_interest_paid[name] = period_interest_paid.get(name, 0.0) + distribution
                    if period_receipts and receipt_interest_total > 0 and remaining_pool > 0:
                        for receipt_row, receipt_interest in zip(period_receipts, receipt_amounts):
                            amount = distribution * receipt_pool / remaining_pool * receipt_interest / receipt_interest_total
                            record_flow(name, parse_date(receipt_row.get("receipt_date"), payment_date),
                                        "Interest Distribution", amount, amount)
                        if manual_pool > 0:
                            amount = distribution * manual_pool / remaining_pool
                            record_flow(name, payment_date, "Interest Distribution", amount, amount)
                    else:
                        record_flow(name, payment_date, "Interest Distribution", distribution, distribution)
                # Explicit Investor Portal distributions were already recorded on their dates.
        else:
            for name in list(balances):
                if investor_interest_capitalised:
                    # Keep accruing the investor target return, but do not create a cash payment.
                    interest_by_investor[name] = accrued_interest.get(name, 0.0)
                elif name in actual_interest_names:
                    interest_by_investor[name] = period_interest_actual.get(name, 0.0)
                else:
                    interest = accrued_interest.get(name, 0.0)
                    interest_by_investor[name] = interest
                    if interest > 0:
                        record_flow(name, payment_date, "Interest Payment", interest, interest)
                        period_interest_paid[name] = period_interest_paid.get(name, 0.0) + interest

        # Capitalised treatment carries investor accrual into their balance at the same
        # interval selected for borrower capitalisation. No forecast cash interest is paid.
        if investor_interest_capitalised:
            for name in list(balances):
                capitalised_interest_pending.setdefault(name, 0.0)
                earned = max(0.0, float(interest_by_investor.get(name, 0.0) or 0.0))
                paid = max(0.0, float(period_interest_paid.get(name, 0.0) or 0.0))
                capitalised_interest_pending[name] += max(0.0, earned - paid)
            period_payoff = any(
                bool(row.get("is_payoff", False))
                for row in period_receipts
                if pd.notna(row.get("is_payoff", False))
            )
            capitalize_now = (
                (period_index + 1) % capitalisation_interval == 0
                or period_index == len(schedule_df) - 1
                or period_payoff
            )
            if capitalize_now:
                for name in list(balances):
                    amount = max(0.0, capitalised_interest_pending.get(name, 0.0))
                    balances[name] += amount
                    period_capitalised_interest[name] = amount
                    capitalised_interest_pending[name] = 0.0

        for name, interest in interest_by_investor.items():
            total_interest[name] = total_interest.get(name, 0.0) + interest

        if has_schedule_actual_interest:
            receipt_principal_total = sum(max(0.0, float(row.get("principal", 0.0) or 0.0))
                                          for row in period_receipts)
            interest_due_for_schedule = max(0.0, float(schedule_row.get("Total_Interest_Due", 0.0) or 0.0))
            actual_principal_value = schedule_row.get("Actual_Principal")
            actual_principal_amount = (max(0.0, float(actual_principal_value))
                                       if pd.notna(actual_principal_value) else receipt_principal_total)
            if period_receipts and actual_principal_amount > 0:
                total_balance = sum(max(0.0, balance) for balance in balances.values())
                receipt_events = sorted(period_receipts, key=lambda row: (
                    parse_date(row.get("receipt_date"), payment_date),
                    -int(row.get("priority", 99) or 99)))
                remaining_principal = actual_principal_amount
                for receipt_row in receipt_events:
                    receipt_date = parse_date(receipt_row.get("receipt_date"), payment_date)
                    facility_principal = max(0.0, float(receipt_row.get("principal", 0.0) or 0.0))
                    if facility_principal <= 0 or total_balance <= 0:
                        continue
                    cap = min(facility_principal, remaining_principal, total_balance)
                    for name in list(balances):
                        returned = min(max(0.0, balances[name]), cap * max(0.0, balances[name]) / total_balance)
                        if returned <= 0:
                            continue
                        balances[name] = max(0.0, balances[name] - returned)
                        register_principal_repayment(name, receipt_date, returned, "Principal Repayment")
                    total_balance = sum(max(0.0, balance) for balance in balances.values())
                    remaining_principal -= cap
                if remaining_principal > 0.01:
                    total_balance = sum(max(0.0, balance) for balance in balances.values())
                    cap = min(remaining_principal, total_balance)
                    if total_balance > 0 and cap > 0:
                        for name in list(balances):
                            returned = min(max(0.0, balances[name]), cap * max(0.0, balances[name]) / total_balance)
                            if returned <= 0:
                                continue
                            balances[name] = max(0.0, balances[name] - returned)
                            register_principal_repayment(name, payment_date, returned, "Principal Repayment")
            elif actual_principal_amount > 0:
                principal_due = min(actual_principal_amount, sum(max(0.0, balance) for balance in balances.values()))
                total_balance = sum(max(0.0, balance) for balance in balances.values())
                if total_balance > 0:
                    for name in list(balances):
                        returned = min(max(0.0, balances[name]), principal_due * max(0.0, balances[name]) / total_balance)
                        balances[name] = max(0.0, balances[name] - returned)
                        if returned > 0:
                            register_principal_repayment(name, payment_date, returned, "Principal Repayment")
        if not has_actual_principal and not has_schedule_actual_interest:
            total_balance = sum(max(0.0, balance) for balance in balances.values())
            principal_due = min(float(schedule_row.get("Principal_Applied", 0.0)), total_balance)
            if total_balance > 0 and principal_due > 0:
                for name in list(balances):
                    returned = min(max(0.0, balances[name]), principal_due * max(0.0, balances[name]) / total_balance)
                    balances[name] = max(0.0, balances[name] - returned)
                    if returned > 0:
                        register_principal_repayment(name, payment_date, returned, "Scheduled Principal Return")

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
                "Interest_Paid": period_interest_paid.get(name, 0.0),
                "Capitalised_Interest": period_capitalised_interest.get(name, 0.0),
                "Principal_Returned": returned,
                "Buyout_In": period_buyout_in.get(name, 0.0),
                "Buyout_Out": period_buyout_out.get(name, 0.0),
                "Ending_Balance": balances.get(name, 0.0),
                "Cumulative_Funded": total_funded.get(name, 0.0),
                "Cumulative_Interest": total_interest.get(name, 0.0),
            })

    ledger_columns = ["Drawdown_Date", "Payment_Date", "Investor", "Capital_Called", "Other_Increase",
        "Other_Decrease", "Interest_Earned", "Interest_Paid", "Capitalised_Interest",
        "Principal_Returned", "Buyout_In", "Buyout_Out", "Ending_Balance",
        "Cumulative_Funded", "Cumulative_Interest"]
    ledger_df = pd.DataFrame(rows, columns=ledger_columns)
    summary_rows = []
    for name in investor_names:
        summary_rows.append({
            "Investor": name,
            "Total_Funded": total_funded.get(name, 0.0),
            "Total_Interest_Earned": total_interest.get(name, 0.0),
            "Total_Principal_Returned": total_principal_returned.get(name, 0.0),
            "Outstanding_Balance": balances.get(name, 0.0),
            "Target_XIRR": target_rates.get(name, default_target_rate) * 100.0,
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
    if "Capitalised_Interest" in schedule_df.columns:
        borrower_interest += float(schedule_df["Capitalised_Interest"].sum())
    penalty_interest = float(schedule_df["Penalty_Interest"].sum())
    line_fees = float(schedule_df["Line_Fee"].sum())
    admin_fees = float(schedule_df["Admin_Fee"].sum())
    total_income = (borrower_interest + penalty_interest + line_fees + admin_fees
        + fees["establishment"] + fees["legal_fee"] + fees["risk_review"]
        + fees["other_upfront"] + fees["discharge"] + fees["restructure"])
    investor_expense = float(investor_summary_df["Total_Interest_Earned"].sum()) if len(investor_summary_df) else 0.0
    upfront_cost_expense = (float(schedule_df["Upfront_Fees_Deducted"].sum())
        if "Upfront_Fees_Deducted" in schedule_df.columns else float(fees["upfront_total"]))
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
        "Upfront Cost Expense": upfront_cost_expense,
        "Net Revenue": total_income - investor_expense - upfront_cost_expense,
        "Borrower Rate (%)": borrower_rate,
        "Weighted Investor Target (%)": weighted_cost,
        "Platform Spread (bps)": (borrower_rate - weighted_cost) * 100.0,
    }


def build_cashflows(schedule_df, investor_ledger_df, fees, investor_cashflows=None,
                    borrower_receipt_allocations_df=None):
    columns = ["Investor_Capital_Called", "Investor_Other_Increase", "Borrower_Interest_Received",
        "Borrower_Principal_Received", "Borrower_Fees_Received", "Total_Inflow",
        "Net_Disbursed_to_Borrower", "Upfront_Costs_Paid", "Investor_Interest_Paid", "Investor_Principal_Returned",
        "Investor_Other_Decrease", "Total_Outflow", "Net_SPV_Cashflow", "Cumulative_SPV_Cash"]
    events = {}
    borrower_receipt_allocations_df = (borrower_receipt_allocations_df
        if borrower_receipt_allocations_df is not None else pd.DataFrame())
    receipts_by_period = {}
    if not borrower_receipt_allocations_df.empty:
        for _, receipt_row in borrower_receipt_allocations_df.iterrows():
            period = parse_date(receipt_row.get("period_date"))
            if period is not None:
                receipts_by_period.setdefault(period.isoformat(), []).append(receipt_row)

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
            draw_event["Upfront_Costs_Paid"] += float(row["Upfront_Fees_Deducted"])

        period_receipts = receipts_by_period.get(payment_date.isoformat(), [])
        actual_fields = [row.get(name) for name in
            ("Actual_Interest", "Actual_Penalty_Interest", "Actual_Principal", "Actual_Fees")]
        has_actuals = bool(period_receipts) or any(pd.notna(value) for value in actual_fields)
        if has_actuals:
            receipt_totals = {"interest": 0.0, "penalty_interest": 0.0, "principal": 0.0, "fees": 0.0}
            for receipt_row in period_receipts:
                receipt_date = parse_date(receipt_row.get("receipt_date"), payment_date)
                interest = max(0.0, float(receipt_row.get("interest", 0.0) or 0.0))
                penalty = max(0.0, float(receipt_row.get("penalty_interest", 0.0) or 0.0))
                principal = max(0.0, float(receipt_row.get("principal", 0.0) or 0.0))
                receipt_fees = max(0.0, float(receipt_row.get("fees", 0.0) or 0.0))
                event = bucket(receipt_date)
                event["Borrower_Interest_Received"] += interest + penalty
                event["Borrower_Principal_Received"] += principal
                event["Borrower_Fees_Received"] += receipt_fees
                receipt_totals["interest"] += interest
                receipt_totals["penalty_interest"] += penalty
                receipt_totals["principal"] += principal
                receipt_totals["fees"] += receipt_fees
            actual_values = {
                "interest": row.get("Actual_Interest"),
                "penalty_interest": row.get("Actual_Penalty_Interest"),
                "principal": row.get("Actual_Principal"),
                "fees": row.get("Actual_Fees"),
            }
            residual = {key: max(0.0, (float(value) if pd.notna(value) else 0.0) - receipt_totals[key])
                        for key, value in actual_values.items()}
            event = bucket(payment_date)
            event["Borrower_Interest_Received"] += residual["interest"] + residual["penalty_interest"]
            event["Borrower_Principal_Received"] += residual["principal"]
            event["Borrower_Fees_Received"] += residual["fees"]
        else:
            # Contractual interest in advance is a dated cashflow on the draw/anniversary date.
            interest_date = parse_date(row.get("Interest_Due_Date"), draw_date)
            bucket(interest_date)["Borrower_Interest_Received"] += float(row.get("Cash_Interest_Advance", 0.0) or 0.0)
            payment_event = bucket(payment_date)
            payment_event["Borrower_Interest_Received"] += float(row.get("Cash_Arrears_Interest", 0.0) or 0.0)
            payment_event["Borrower_Interest_Received"] += float(row.get("Cash_Capitalised_Interest_Received", 0.0) or 0.0)
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
            payment_event["Investor_Interest_Paid"] += float(row.get("Interest_Paid", 0.0) or 0.0)
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
        row["Total_Outflow"] = (row["Net_Disbursed_to_Borrower"] + row["Upfront_Costs_Paid"]
            + row["Investor_Interest_Paid"]
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
                       LEFT JOIN borrowers b ON d.borrower_id = b.id
                       ORDER BY COALESCE(d.parent_deal_id, d.id), d.id""", conn)
    conn.close()
    if not df.empty:
        df["loan_ranking"] = df.apply(lambda row: safe_json(row.get("params_json"), {}).get(
            "loan_ranking", "Mezzanine" if row.get("facility_type") == "Mezzanine" else "Senior"), axis=1)
        df["facility_type"] = df["facility_type"].map(
            lambda value: normalize_facility_type(value, fallback="Bridging"))
    else:
        df["loan_ranking"] = pd.Series(dtype="object")
    return df


def save_deal(borrower_id, deal_name, facility, status, settlement, maturity,
              facility_type, security_type, property_value, params,
              facility_name="Tranche 1", parent_deal_id=None):
    facility_type = normalize_facility_type(facility_type)
    params = dict(params or {})
    params["facility_type"] = facility_type
    params.setdefault("principal_repayment_mode", default_principal_repayment_mode(facility_type))
    params.setdefault("principal_repayment_schedule", [])
    conn = get_conn()
    conn.execute("""INSERT INTO deals (borrower_id, deal_name, facility_name, parent_deal_id,
        facility_amount, deal_status, settlement_date, maturity_date, facility_type,
        security_type, property_value, params_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (int(borrower_id), deal_name, facility_name.strip(), parent_deal_id,
         float(facility), status, settlement.isoformat(), maturity.isoformat(), facility_type,
         security_type, float(property_value), json.dumps(params)))
    deal_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
    if parent_deal_id is None:
        conn.execute("UPDATE deals SET parent_deal_id=? WHERE id=?", (deal_id, deal_id))
    conn.commit()
    conn.close()
    ensure_default_deal_investors(deal_id)
    return deal_id


def save_additional_facility(parent_deal_id, facility_name, facility_amount, status, settlement,
                             maturity, facility_type, security_type, property_value, params):
    conn = get_conn()
    parent = conn.execute("SELECT borrower_id, deal_name FROM deals WHERE id=?", (int(parent_deal_id),)).fetchone()
    conn.close()
    if not parent:
        raise ValueError("Parent deal was not found.")
    new_id = save_deal(parent[0], parent[1], facility_amount, status, settlement, maturity,
        facility_type, security_type, property_value, params, facility_name, int(parent_deal_id))
    # New tranches inherit the deal-specific return defaults from Tranche 1,
    # while keeping their own independent assignment rows and later overrides.
    conn = get_conn()
    conn.execute("""UPDATE deal_investors SET
        senior_target_irr=COALESCE((SELECT parent.senior_target_irr FROM deal_investors AS parent
            WHERE parent.deal_id=? AND parent.investor_id=deal_investors.investor_id AND parent.is_active=1),
            senior_target_irr),
        mezz_target_irr=COALESCE((SELECT parent.mezz_target_irr FROM deal_investors AS parent
            WHERE parent.deal_id=? AND parent.investor_id=deal_investors.investor_id AND parent.is_active=1),
            mezz_target_irr)
        WHERE deal_id=?""", (int(parent_deal_id), int(parent_deal_id), int(new_id)))
    conn.commit()
    conn.close()
    return new_id


def standard_split_for(investor_name, loan_ranking):
    if loan_ranking == "Mezzanine":
        return {"BPCFMC": 80.0, "BPCFLC": 20.0}.get(investor_name, 0.0)
    return {"BPCFMC": 50.0, "BPCFLC": 15.0, "CBCF": 35.0}.get(investor_name, 0.0)


def ensure_default_deal_investors(deal_id):
    conn = get_conn()
    core_defaults = {"BPCFMC": (50.0, 80.0), "BPCFLC": (15.0, 20.0), "CBCF": (35.0, 0.0)}
    for name, (senior_pct, mezz_pct) in core_defaults.items():
        investor = conn.execute("""SELECT id, target_irr, senior_target_irr, mezz_target_irr
            FROM investors WHERE name=?""", (name,)).fetchone()
        if investor:
            conn.execute("""INSERT OR IGNORE INTO deal_investors
                (deal_id, investor_id, senior_default_pct, mezz_default_pct, target_irr,
                 senior_target_irr, mezz_target_irr, is_active)
                VALUES (?,?,?,?,?,?,?,1)""", (int(deal_id), int(investor[0]), senior_pct, mezz_pct,
                float(investor[1] if investor[1] is not None else 8.0),
                safe_rate_pct(investor[2], 8.0), safe_rate_pct(investor[3], 8.0)))
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
    df = pd.read_sql("""SELECT i.id, i.name, i.investor_type,
        COALESCE(di.target_irr, i.target_irr) AS target_irr,
        COALESCE(i.senior_target_irr, i.target_irr, 8.0) AS senior_target_irr,
        COALESCE(i.mezz_target_irr, i.target_irr, 8.0) AS mezz_target_irr,
        COALESCE(di.senior_target_irr, i.senior_target_irr, i.target_irr, 8.0) AS deal_senior_target_irr,
        COALESCE(di.mezz_target_irr, i.mezz_target_irr, i.target_irr, 8.0) AS deal_mezz_target_irr,
        i.status, di.senior_default_pct, di.mezz_default_pct
        FROM deal_investors di JOIN investors i ON i.id=di.investor_id
        WHERE di.deal_id=? AND di.is_active=1 AND i.status='Active' ORDER BY i.name""",
        conn, params=(int(deal_id),))
    conn.close()
    return df


def save_investor(name, investor_type, target_irr=8.0, notes="",
                  senior_target_irr=None, mezz_target_irr=None):
    senior_rate = float(target_irr if senior_target_irr is None else senior_target_irr)
    mezz_rate = float(target_irr if mezz_target_irr is None else mezz_target_irr)
    conn = get_conn()
    conn.execute("""INSERT INTO investors
        (name, investor_type, target_irr, senior_target_irr, mezz_target_irr, notes)
        VALUES (?,?,?,?,?,?)""",
        (name.strip(), investor_type, float(target_irr), senior_rate, mezz_rate, notes.strip()))
    conn.commit()
    conn.close()


def update_investor(investor_id, name, investor_type, target_irr, status, notes="",
                    senior_target_irr=None, mezz_target_irr=None):
    senior_rate = float(target_irr if senior_target_irr is None else senior_target_irr)
    mezz_rate = float(target_irr if mezz_target_irr is None else mezz_target_irr)
    conn = get_conn()
    conn.execute("""UPDATE investors SET name=?, investor_type=?, target_irr=?,
        senior_target_irr=?, mezz_target_irr=?, status=?, notes=? WHERE id=?""",
        (name.strip(), investor_type, float(target_irr), senior_rate, mezz_rate,
         status, notes.strip(), int(investor_id)))
    conn.commit()
    conn.close()


def save_deal_investors(deal_id, investor_ids, target_returns=None,
                        senior_target_returns=None, mezz_target_returns=None):
    conn = get_conn()
    deal_id = int(deal_id)
    conn.execute("UPDATE deal_investors SET is_active=0 WHERE deal_id=?", (deal_id,))
    defaults = {"BPCFMC": (50.0, 80.0), "BPCFLC": (15.0, 20.0), "CBCF": (35.0, 0.0)}
    target_returns = target_returns or {}
    senior_target_returns = senior_target_returns or {}
    mezz_target_returns = mezz_target_returns or {}
    parent_row = conn.execute("SELECT parent_deal_id FROM deals WHERE id=?", (deal_id,)).fetchone()
    parent_deal_id = int(parent_row[0]) if parent_row and parent_row[0] is not None else deal_id
    for investor_id in investor_ids:
        investor_id = int(investor_id)
        investor = conn.execute("""SELECT name, target_irr, senior_target_irr, mezz_target_irr
            FROM investors WHERE id=?""", (investor_id,)).fetchone()
        if not investor:
            continue
        name, global_target, global_senior, global_mezz = investor
        senior_pct, mezz_pct = defaults.get(name, (0.0, 0.0))
        existing = conn.execute("""SELECT target_irr, senior_target_irr, mezz_target_irr
            FROM deal_investors WHERE deal_id=? AND investor_id=?""",
            (deal_id, investor_id)).fetchone()
        parent_defaults = None
        if parent_deal_id != deal_id:
            parent_defaults = conn.execute("""SELECT senior_target_irr, mezz_target_irr
                FROM deal_investors WHERE deal_id=? AND investor_id=? AND is_active=1""",
                (parent_deal_id, investor_id)).fetchone()
        target = target_returns.get(investor_id)
        if target is None:
            target = existing[0] if existing and existing[0] is not None else global_target
        try:
            target = float(target if target is not None else 8.0)
            if not np.isfinite(target):
                target = 8.0
        except (TypeError, ValueError):
            target = 8.0
        senior_value = senior_target_returns.get(investor_id)
        if senior_value is None:
            senior_value = (existing[1] if existing and existing[1] is not None else
                parent_defaults[0] if parent_defaults and parent_defaults[0] is not None else global_senior)
        mezz_value = mezz_target_returns.get(investor_id)
        if mezz_value is None:
            mezz_value = (existing[2] if existing and existing[2] is not None else
                parent_defaults[1] if parent_defaults and parent_defaults[1] is not None else global_mezz)
        senior_value = safe_rate_pct(senior_value, 8.0)
        mezz_value = safe_rate_pct(mezz_value, 8.0)
        conn.execute("""INSERT INTO deal_investors
            (deal_id, investor_id, senior_default_pct, mezz_default_pct, target_irr,
             senior_target_irr, mezz_target_irr, is_active)
            VALUES (?,?,?,?,?,?,?,1)
            ON CONFLICT(deal_id, investor_id) DO UPDATE SET is_active=1,
                target_irr=excluded.target_irr,
                senior_target_irr=excluded.senior_target_irr,
                mezz_target_irr=excluded.mezz_target_irr""",
            (deal_id, investor_id, senior_pct, mezz_pct, float(target), senior_value, mezz_value))
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
    query = """SELECT t.*, d.deal_name, d.facility_name, d.parent_deal_id,
        src.name AS investor_name, dst.name AS counterparty_name
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
    params = dict(params or {})
    params["facility_type"] = normalize_facility_type(params.get("facility_type", "Bridging"))
    params.setdefault("principal_repayment_mode",
        default_principal_repayment_mode(params["facility_type"]))
    params.setdefault("principal_repayment_schedule", [])
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


def save_borrower_receipt(parent_deal_id, receipt_date, amount, interest_through_date,
                          is_payoff, notes, allocations):
    conn = get_conn()
    cursor = conn.execute("""INSERT INTO borrower_receipts
        (parent_deal_id, receipt_date, amount, interest_through_date, is_payoff, notes)
        VALUES (?,?,?,?,?,?)""", (int(parent_deal_id), parse_date(receipt_date).isoformat(),
        float(amount), parse_date(interest_through_date).isoformat() if interest_through_date else None,
        int(bool(is_payoff)), str(notes or "").strip()))
    receipt_id = int(cursor.lastrowid)
    for _, row in allocations.iterrows():
        conn.execute("""INSERT INTO borrower_receipt_allocations
            (receipt_id, facility_deal_id, period_date, fees, penalty_interest, interest, principal, override_flag)
            VALUES (?,?,?,?,?,?,?,?)""", (receipt_id, int(row["facility_deal_id"]),
            parse_date(row["period_date"]).isoformat(), float(row.get("fees", 0.0) or 0.0),
            float(row.get("penalty_interest", 0.0) or 0.0), float(row.get("interest", 0.0) or 0.0),
            float(row.get("principal", 0.0) or 0.0), int(bool(row.get("override_flag", False)))))
    conn.commit()
    conn.close()
    return receipt_id


def load_borrower_receipts(parent_deal_id=None):
    conn = get_conn()
    query = """SELECT r.*, d.deal_name FROM borrower_receipts r
        JOIN deals d ON d.id=r.parent_deal_id"""
    if parent_deal_id is None:
        df = pd.read_sql(query + " ORDER BY r.receipt_date DESC, r.id DESC", conn)
    else:
        df = pd.read_sql(query + " WHERE r.parent_deal_id=? ORDER BY r.receipt_date DESC, r.id DESC",
                         conn, params=(int(parent_deal_id),))
    conn.close()
    return df


def load_receipt_allocations(facility_deal_id=None, parent_deal_id=None):
    conn = get_conn()
    query = """SELECT a.*, r.parent_deal_id, r.receipt_date, r.amount AS receipt_amount,
        r.interest_through_date, r.is_payoff, r.notes, d.deal_name, d.facility_name,
        d.params_json FROM borrower_receipt_allocations a
        JOIN borrower_receipts r ON r.id=a.receipt_id
        JOIN deals d ON d.id=a.facility_deal_id"""
    if facility_deal_id is not None:
        df = pd.read_sql(query + " WHERE a.facility_deal_id=? ORDER BY r.receipt_date, r.id",
                         conn, params=(int(facility_deal_id),))
    elif parent_deal_id is not None:
        df = pd.read_sql(query + " WHERE r.parent_deal_id=? ORDER BY r.receipt_date, r.id",
                         conn, params=(int(parent_deal_id),))
    else:
        df = pd.read_sql(query + " ORDER BY r.receipt_date, r.id", conn)
    conn.close()
    return df


def merge_receipt_allocations_into_actuals(actuals_map, allocations_df):
    result = {key: dict(value) for key, value in actuals_map.items()}
    if allocations_df.empty:
        return result
    for _, row in allocations_df.iterrows():
        period_date = parse_date(row.get("period_date"))
        if period_date is None:
            continue
        key = period_date.isoformat()
        item = result.setdefault(key, {"actual_drawdown": 0.0, "drawdown_override": False,
            "payment_actuals_logged": False, "actual_interest": 0.0, "actual_penalty_interest": 0.0,
            "actual_principal": 0.0, "actual_fees": 0.0, "_receipt_events": []})
        item["payment_actuals_logged"] = True
        item["actual_interest"] = float(item.get("actual_interest", 0.0) or 0.0) + float(row.get("interest", 0.0) or 0.0)
        item["actual_penalty_interest"] = float(item.get("actual_penalty_interest", 0.0) or 0.0) + float(row.get("penalty_interest", 0.0) or 0.0)
        item["actual_principal"] = float(item.get("actual_principal", 0.0) or 0.0) + float(row.get("principal", 0.0) or 0.0)
        item["actual_fees"] = float(item.get("actual_fees", 0.0) or 0.0) + float(row.get("fees", 0.0) or 0.0)
        item.setdefault("_receipt_events", []).append({
            "receipt_date": parse_date(row.get("receipt_date"), period_date),
            "interest": float(row.get("interest", 0.0) or 0.0),
            "penalty_interest": float(row.get("penalty_interest", 0.0) or 0.0),
            "principal": float(row.get("principal", 0.0) or 0.0),
            "fees": float(row.get("fees", 0.0) or 0.0),
            "is_payoff": bool(row.get("is_payoff", False)),
        })
    return result


def last_settlement_anniversary(settlement, on_or_before):
    settlement, on_or_before = parse_date(settlement), parse_date(on_or_before)
    months = (on_or_before.year - settlement.year) * 12 + on_or_before.month - settlement.month
    candidate = add_months(settlement, months)
    if candidate > on_or_before:
        candidate = add_months(settlement, months - 1)
    return candidate


def default_facility_model_params(facility_row):
    params = safe_json(facility_row.get("params_json", "{}"), {})
    settlement = parse_date(params.get("settlement_date"), parse_date(facility_row.get("settlement_date"), date.today()))
    maturity = parse_date(params.get("maturity_date"), parse_date(facility_row.get("maturity_date"), add_months(settlement, 12)))
    term = max(1, min(120, int(params.get("term_months", infer_term_months(settlement, maturity)))))
    facility_type = normalize_facility_type(params.get("facility_type",
        facility_row.get("facility_type", "Bridging")))
    principal_repayment_mode = str(params.get("principal_repayment_mode",
        default_principal_repayment_mode(facility_type)))
    if principal_repayment_mode not in PRINCIPAL_REPAYMENT_MODES:
        principal_repayment_mode = default_principal_repayment_mode(facility_type)
    params.update({
        "facility_amount": float(params.get("facility_amount", facility_row.get("facility_amount", 0)) or 0),
        "facility_type": facility_type,
        "property_value": float(params.get("property_value", facility_row.get("property_value", 0)) or 0),
        "settlement_date": settlement.isoformat(), "maturity_date": add_months(settlement, term).isoformat(),
        "term_months": term, "borrower_irr": float(params.get("borrower_irr", 12.0)),
        "construction_months": max(0, int(params.get("construction_months", min(6, term)))),
        "interest_only_months": max(0, min(term, int(params.get("interest_only_months",
            params.get("construction_months", min(6, term)))))),
        "drawdown_profile": str(params.get("drawdown_profile", "Automatic")),
        "s_curve_steepness": float(params.get("s_curve_steepness", 5.0)),
        "principal_repayment_mode": principal_repayment_mode,
        "principal_repayment_schedule": parse_principal_repayment_rows(
            params.get("principal_repayment_schedule", [])),
        "loan_ranking": str(params.get("loan_ranking", facility_row.get("loan_ranking", "Senior"))),
        "interest_treatment": str(params.get("interest_treatment", "Serviced")),
        "interest_advance_months": params.get("interest_advance_months", 1),
    })
    if not params.get("drawdown_schedule"):
        template = build_drawdown_template(settlement, term, params["facility_amount"],
            params["facility_type"], params["construction_months"], params["drawdown_profile"],
            params["s_curve_steepness"])
        plan = {}
        upfront = compute_fees(params)["upfront_total"]
        first_draw = next((index for index, amount in enumerate(template["Drawdown Amount"].tolist()) if amount > 0), None)
        for index, row in template.iterrows():
            gross = float(row["Drawdown Amount"])
            funded_upfront = min(upfront, gross) if index == first_draw else 0.0
            allocations = default_allocation_amounts(gross + funded_upfront,
                params["loan_ranking"], ["BPCFMC", "BPCFLC", "CBCF"])
            plan[parse_date(row["Drawdown Date"]).isoformat()] = {"gross_draw": gross,
                "allocation_source": "Standard", "status": "Forecast", "allocations": allocations}
        params["drawdown_schedule"] = json.dumps(plan)
    return params


def contractual_interest_through(schedule_df, params, cutoff_date):
    """Accrue ACT/365 contractual interest through an actual date."""
    settlement = parse_date(params.get("settlement_date"), date.today())
    cutoff = parse_date(cutoff_date, settlement)
    if cutoff <= settlement or schedule_df.empty:
        return 0.0
    raw_treatment = str(params.get("interest_treatment", "Serviced")).strip().lower()
    capitalised = raw_treatment in ["capitalised", "capitalized"]
    advance_setting = params.get("interest_advance_months", 1)
    try:
        advance_months = (int(params.get("term_months", len(schedule_df)))
            if str(advance_setting).strip().lower() == "all"
            else max(1, min(int(params.get("term_months", len(schedule_df))), int(advance_setting))))
    except (TypeError, ValueError):
        advance_months = 1
    facility_amount = max(0.0, float(params.get("facility_amount", 0.0) or 0.0))
    rate = float(params.get("borrower_irr", 12.0)) / 100.0
    total = 0.0
    for index, (_, row) in enumerate(schedule_df.iterrows()):
        start = parse_date(row.get("Drawdown_Date"))
        end = parse_date(row.get("Payment_Date"))
        if start is None or end is None or cutoff <= start:
            continue
        segment_end = min(end, cutoff)
        days = max((segment_end - start).days, 0)
        if capitalised:
            principal = max(0.0, float(row.get("Balance_After_Draw", 0.0) or 0.0))
        else:
            principal = (facility_amount if index < advance_months
                else max(0.0, float(row.get("Balance_After_Draw", 0.0) or 0.0)))
        total += principal * annual_rate_for_days(rate, days)
        if cutoff <= end:
            break
    return total


def arrears_penalty_through(period_row, params, previous_period, cutoff_date, interest_due=None):
    """ACT/365 charge on overdue interest through the receipt/payoff date."""
    start = parse_date(period_row.get("Drawdown_Date"))
    cutoff = parse_date(cutoff_date, start)
    if start is None or cutoff is None or cutoff <= start:
        return 0.0
    cutoff = min(cutoff, parse_date(period_row.get("Payment_Date"), cutoff))
    rate = float(params.get("borrower_irr", 12.0)) / 100.0
    scheduled_interest = (float(period_row.get("Interest_Due", 0.0) or 0.0)
                          if interest_due is None else float(interest_due))
    scheduled_interest = max(0.0, scheduled_interest
        - float(period_row.get("Advance_Interest_Withheld", 0.0) or 0.0))
    outstanding = max(0.0, float(period_row.get("Arrears_Opening", 0.0) or 0.0) + scheduled_interest)
    cursor = start
    accrued = 0.0
    if not previous_period.empty:
        receipt_dates = pd.to_datetime(previous_period["receipt_date"], errors="coerce").dt.date
        previous_period = previous_period[receipt_dates <= cutoff].copy()
    rows = previous_period.sort_values("receipt_date") if not previous_period.empty else previous_period
    for _, receipt in rows.iterrows():
        event_date = parse_date(receipt.get("receipt_date"), cutoff)
        if event_date > cutoff:
            continue
        event_date = max(start, min(event_date, cutoff))
        if event_date > cursor and outstanding > 0:
            accrued += outstanding * annual_rate_for_days(rate, (event_date - cursor).days)
        cursor = max(cursor, event_date)
        outstanding = max(0.0, outstanding - max(0.0, float(receipt.get("interest", 0.0) or 0.0)))
    if cutoff > cursor and outstanding > 0:
        accrued += outstanding * annual_rate_for_days(rate, (cutoff - cursor).days)
    paid_penalty = float(previous_period["penalty_interest"].fillna(0).sum()) if not previous_period.empty else 0.0
    return max(0.0, accrued - paid_penalty)


def suggest_receipt_allocations(parent_deal_id, receipt_date, amount,
                                interest_through_date=None, is_payoff=False):
    deals = load_deals()
    facilities = deals[deals["parent_deal_id"] == int(parent_deal_id)].copy()
    facility_rows = []
    receipt_date = parse_date(receipt_date, date.today())
    for _, facility in facilities.iterrows():
        params = default_facility_model_params(facility)
        settlement = parse_date(params["settlement_date"])
        term = int(params["term_months"])
        previous = load_receipt_allocations(facility_deal_id=int(facility["id"]))
        if not previous.empty:
            previous_dates = pd.to_datetime(previous["receipt_date"], errors="coerce").dt.date
            previous = previous[previous_dates <= receipt_date].copy()
        actuals_map = actuals_to_map(load_actuals(int(facility["id"])), settlement, term)
        actuals_map = merge_receipt_allocations_into_actuals(actuals_map, previous)
        draw_map = safe_json(params.get("drawdown_schedule", {}), {})
        schedule, _, _, _ = build_schedule(params, actuals_map, draw_map)
        before = schedule[schedule["Drawdown_Date"] <= receipt_date]
        current = schedule.iloc[0] if before.empty else before.iloc[-1]
        period_date = parse_date(current["Payment_Date"])
        current_index = int(schedule.index[schedule["Payment_Date"] == period_date][0])
        period_previous = previous[pd.to_datetime(previous["period_date"], errors="coerce").dt.date == period_date] if not previous.empty else previous
        rank = str(params.get("loan_ranking", "Senior"))
        prior_fees = float(period_previous["fees"].fillna(0).sum()) if not period_previous.empty else 0.0
        prior_period_principal = float(period_previous["principal"].fillna(0).sum()) if not period_previous.empty else 0.0
        fees_due = max(0.0, float(current["Line_Fee"] + current["Admin_Fee"])
                       + (compute_fees(params)["discharge"] if is_payoff else float(current["Discharge_Fee"]))
                       - prior_fees)
        raw_treatment = str(params.get("interest_treatment", "Serviced")).strip().lower()
        capitalised = raw_treatment in ["capitalised", "capitalized"]
        penalty_cutoff = receipt_date
        payoff_period_interest_due = 0.0
        if is_payoff:
            cutoff = min(parse_date(interest_through_date, receipt_date), receipt_date)
            cutoff = max(cutoff, settlement)
            penalty_cutoff = cutoff
            minimum_interest = 0.0 if capitalised else float(schedule.iloc[0]["Interest_Due"])
            required_contract_interest = (0.0 if capitalised else max(minimum_interest,
                contractual_interest_through(schedule, params, cutoff)))
            prior_accrued = float(schedule.iloc[:current_index]["Interest_Accrued"].sum()) if current_index > 0 else 0.0
            baseline_before_period = 0.0 if current_index == 0 else max(minimum_interest, prior_accrued)
            payoff_period_interest_due = (0.0 if capitalised else
                max(0.0, required_contract_interest - baseline_before_period))
            if not previous.empty:
                paid_dates = pd.to_datetime(previous["receipt_date"], errors="coerce").dt.date
                paid_before_cutoff = previous[paid_dates <= cutoff]
                prior_contract_interest = float(paid_before_cutoff["interest"].fillna(0).sum())
            else:
                prior_contract_interest = 0.0
            prepaid_interest_credit = 0.0
            if not capitalised and "Advance_Interest_Withheld" in schedule:
                draw_dates = pd.to_datetime(schedule["Drawdown_Date"], errors="coerce").dt.date
                prepaid_interest_credit = float(schedule.loc[draw_dates <= cutoff,
                    "Advance_Interest_Withheld"].fillna(0).sum())
            interest_due = (0.0 if capitalised else max(0.0,
                required_contract_interest - prior_contract_interest - prepaid_interest_credit))
        else:
            cutoff = max(settlement, last_settlement_anniversary(settlement, receipt_date))
            due_rows = schedule[pd.to_datetime(schedule["Interest_Due_Date"]).dt.date <= cutoff]
            required_contract_interest = 0.0 if capitalised else float(due_rows["Interest_Due"].fillna(0).sum())
            prior_contract_interest = float(previous["interest"].fillna(0).sum()) if not previous.empty else 0.0
            prepaid_interest_credit = 0.0
            if not capitalised and "Advance_Interest_Withheld" in schedule:
                draw_dates = pd.to_datetime(schedule["Drawdown_Date"], errors="coerce").dt.date
                prepaid_interest_credit = float(schedule.loc[draw_dates <= cutoff,
                    "Advance_Interest_Withheld"].fillna(0).sum())
            interest_due = (0.0 if capitalised else max(0.0,
                required_contract_interest - prior_contract_interest - prepaid_interest_credit))
        penalty_due = arrears_penalty_through(current, params, period_previous, penalty_cutoff,
            payoff_period_interest_due if is_payoff else None)
        if capitalised:
            accrual_cutoff = penalty_cutoff
            current_draw_date = parse_date(current["Drawdown_Date"], settlement)
            starting_balance = max(0.0, float(current["Balance_After_Draw"]))
            event_rows = period_previous.copy()
            if not event_rows.empty:
                event_dates = pd.to_datetime(event_rows["receipt_date"], errors="coerce").dt.date
                event_rows = event_rows[event_dates <= accrual_cutoff]
            receipt_events = event_rows.to_dict("records") if not event_rows.empty else []
            capitalised_accrual = capitalised_interest_for_period(starting_balance,
                current_draw_date, accrual_cutoff, float(params.get("borrower_irr", 12.0)) / 100.0,
                receipt_events)
            try:
                interval_setting = params.get("interest_advance_months", 1)
                capitalization_interval = (term if str(interval_setting).strip().lower() == "all"
                    else max(1, min(term, int(interval_setting))))
            except (TypeError, ValueError):
                capitalization_interval = 1
            uncapitalised_start = (current_index // capitalization_interval) * capitalization_interval
            uncapitalised_before_current = float(schedule.iloc[uncapitalised_start:current_index][
                "Interest_Accrued"].fillna(0.0).sum())
            principal_received_before_cutoff = sum(max(0.0, float(row.get("principal", 0.0) or 0.0))
                for row in receipt_events)
            principal_available = max(0.0,
                starting_balance + uncapitalised_before_current + capitalised_accrual
                - principal_received_before_cutoff)
        else:
            principal_available = max(0.0, float(current["Balance_After_Draw"]) - prior_period_principal)
        facility_rows.append({"facility_deal_id": int(facility["id"]), "Facility": str(facility["facility_name"]),
            "Ranking": rank, "period_date": period_date, "fees_available": fees_due,
            "penalty_available": penalty_due, "interest_available": interest_due,
            "principal_available": principal_available, "fees": 0.0, "penalty_interest": 0.0,
            "interest": 0.0, "principal": 0.0})

    remaining = max(0.0, float(amount))
    # Clear fees and arrears charges first, then follow the deal's senior-first facility waterfall.
    for component, cap_key in [("fees", "fees_available"), ("penalty_interest", "penalty_available")]:
        capacities = [max(0.0, row[cap_key]) for row in facility_rows]
        total_cap = sum(capacities)
        take = min(remaining, total_cap)
        if total_cap > 0 and take > 0:
            for row, capacity in zip(facility_rows, capacities):
                row[component] = take * capacity / total_cap
            remaining -= take
    for rank in ["Senior", "Mezzanine"]:
        for component, cap_key in [("interest", "interest_available"), ("principal", "principal_available")]:
            group = [row for row in facility_rows if str(row["Ranking"]).strip().lower() == rank.lower()]
            capacities = [max(0.0, row[cap_key] - row[component]) for row in group]
            total_cap = sum(capacities)
            take = min(remaining, total_cap)
            if total_cap > 0 and take > 0:
                for row, capacity in zip(group, capacities):
                    row[component] += take * capacity / total_cap
                remaining -= take
    return pd.DataFrame(facility_rows), remaining


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
            "actual_penalty_interest": 0.0,
            "actual_principal": float(row.get("actual_principal", 0) or 0),
            "actual_fees": float(row.get("actual_fees", 0) or 0),
            "_receipt_events": [],
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
    root_deals = deals[deals["parent_deal_id"] == deals["id"]] if len(deals) else deals
    active = int((root_deals["deal_status"] == "Active").sum()) if len(root_deals) else 0
    pipeline = int((root_deals["deal_status"] == "Pipeline").sum()) if len(root_deals) else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Borrowers", len(borrowers))
    c2.metric("Deals", len(root_deals))
    c3.metric("Total Tranche Exposure", money(total_facility))
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
    recent_cols = ["id", "borrower_name", "deal_name", "facility_name", "facility_type", "facility_amount",
                   "deal_status", "settlement_date", "maturity_date"]
    recent = deals[[column for column in recent_cols if column in deals.columns]].head(10).copy()
    recent = recent.rename(columns={"id": "Tranche ID", "borrower_name": "Borrower", "deal_name": "Deal",
        "facility_name": "Tranche", "facility_type": "Facility Type", "facility_amount": "Tranche Amount", "deal_status": "Status",
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

SECURITY_TYPES = ["First Mortgage", "Second Mortgage", "Unsecured", "Personal Guarantee", "Other"]


def next_tranche_number(deals_df, parent_deal_id):
    """Return the next sequential tranche number for a deal, with no configured ceiling."""
    if deals_df is None or deals_df.empty:
        return 1
    members = deals_df[deals_df["parent_deal_id"] == int(parent_deal_id)]
    if members.empty:
        return 1
    numbers = []
    for label in members.get("facility_name", pd.Series(dtype=str)).fillna("").astype(str):
        parts = label.strip().split()
        if len(parts) >= 2 and parts[-2].lower() in {"tranche", "facility"} and parts[-1].isdigit():
            numbers.append(int(parts[-1]))
    return max(max(numbers, default=0), len(members)) + 1


def page_deals():
    st.title("Deals")
    st.caption("Create a deal with Senior Tranche 1, then add Senior or Mezzanine tranches one at a time. Manage each tranche in Deal Workspace.")
    borrowers = load_borrowers()
    if borrowers.empty:
        st.warning("Add a borrower before creating a deal.")
        return

    with st.expander("＋ Create deal · Senior Tranche 1", expanded=False):
        with st.form("add_deal"):
            c1, c2 = st.columns(2)
            borrower_labels = {f"{row['name']} (ID {int(row['id'])})": int(row["id"])
                               for _, row in borrowers.iterrows()}
            borrower_label = c1.selectbox("Borrower", list(borrower_labels.keys()))
            deal_name = c2.text_input("Deal name *")
            c3, c4 = st.columns(2)
            facility = c3.number_input("Senior Tranche 1 amount", min_value=0.0,
                value=5_000_000.0, step=100_000.0)
            status = c4.selectbox("Status", ["Pipeline", "Active", "Settled", "Closed", "Default"])
            c5, c6 = st.columns(2)
            settlement = c5.date_input("Settlement date", value=date.today())
            maturity = c6.date_input("Maturity date", value=add_months(date.today(), 12))
            c7, c8 = st.columns(2)
            facility_type = c7.selectbox("Facility purpose / type", FACILITY_TYPES)
            security_type = c8.selectbox("Security type", SECURITY_TYPES)
            property_value = st.number_input("Property / security value", min_value=0.0,
                                             value=8_000_000.0, step=100_000.0)
            st.info("Tranche 1 is created as Senior. Later tranches can each be tagged Senior or Mezzanine.")
            if st.form_submit_button("Create deal with Senior Tranche 1", type="primary"):
                if not deal_name.strip():
                    st.error("Deal name is required.")
                elif maturity <= settlement:
                    st.error("Maturity date must be after settlement date.")
                else:
                    borrower_id = borrower_labels[borrower_label]
                    save_deal(borrower_id, deal_name.strip(), facility, status, settlement, maturity,
                              facility_type, security_type, property_value,
                              {"loan_ranking": "Senior", "interest_treatment": "Serviced",
                               "interest_advance_months": 1, "default_investor_target_irr": 8.0},
                              "Tranche 1")
                    st.success(f"Created deal: {deal_name.strip()} with Senior Tranche 1.")
                    st.rerun()

    deals = load_deals()
    roots = deals[deals["parent_deal_id"] == deals["id"]] if not deals.empty else deals
    if not roots.empty:
        root_labels = {f"{row['deal_name']} · {row['borrower_name']} · Deal ID {int(row['id'])}": int(row["id"])
                       for _, row in roots.iterrows()}
        with st.expander("＋ Add next tranche", expanded=False):
            root_label = st.selectbox("Deal", list(root_labels.keys()), key="add_tranche_parent")
            root_id = root_labels[root_label]
            root_row = roots[roots["id"] == root_id].iloc[0]
            next_number = next_tranche_number(deals, root_id)
            tranche_name = f"Tranche {next_number}"
            st.info(f"Next in sequence: {tranche_name}. Each tranche keeps its own terms, ranking, investor returns and allocations.")
            with st.form(f"add_tranche_form_{root_id}"):
                f1, f2, f3 = st.columns(3)
                tranche_widget_suffix = f"{root_id}_{next_number}"
                f1.text_input("Tranche name", value=tranche_name, disabled=True,
                    key=f"tranche_name_{tranche_widget_suffix}")
                new_facility_amount = f2.number_input("Tranche amount", min_value=0.0,
                    value=1_000_000.0, step=100_000.0,
                    key=f"tranche_amount_{tranche_widget_suffix}")
                status_options = ["Pipeline", "Active", "Settled", "Closed", "Default"]
                root_status = str(root_row["deal_status"])
                new_status = f3.selectbox("Status", status_options,
                    index=status_options.index(root_status) if root_status in status_options else 0,
                    key=f"tranche_status_{tranche_widget_suffix}")
                f4, f5, f6 = st.columns(3)
                new_settlement = f4.date_input("Settlement date", value=parse_date(root_row["settlement_date"], date.today()),
                    key=f"tranche_settlement_{tranche_widget_suffix}")
                new_maturity = f5.date_input("Maturity date", value=parse_date(root_row["maturity_date"], add_months(date.today(), 12)),
                    key=f"tranche_maturity_{tranche_widget_suffix}")
                root_facility_type = str(root_row["facility_type"])
                new_facility_type = f6.selectbox("Facility purpose / type", FACILITY_TYPES,
                    index=FACILITY_TYPES.index(root_facility_type) if root_facility_type in FACILITY_TYPES else 0,
                    key=f"tranche_type_{tranche_widget_suffix}")
                f7, f8 = st.columns(2)
                new_ranking = f7.selectbox("Tranche ranking", ["Senior", "Mezzanine"], index=0,
                    key=f"tranche_ranking_{tranche_widget_suffix}")
                root_security = str(root_row["security_type"])
                new_security = f8.selectbox("Security type", SECURITY_TYPES,
                    index=SECURITY_TYPES.index(root_security) if root_security in SECURITY_TYPES else 0,
                    key=f"tranche_security_{tranche_widget_suffix}")
                new_property_value = st.number_input("Property / security value", min_value=0.0,
                    value=float(root_row["property_value"] or 0), step=100_000.0,
                    key=f"tranche_property_{tranche_widget_suffix}")
                if st.form_submit_button(f"Add {tranche_name}", type="primary"):
                    duplicate = ((deals["parent_deal_id"] == root_id) &
                        (deals["facility_name"].fillna("").astype(str).str.lower() == tranche_name.lower())).any()
                    if duplicate:
                        st.error(f"{tranche_name} already exists. Refresh and add the next tranche in sequence.")
                    elif new_maturity <= new_settlement:
                        st.error("Tranche maturity must be after its settlement date.")
                    else:
                        new_id = save_additional_facility(root_id, tranche_name,
                            new_facility_amount, new_status, new_settlement, new_maturity,
                            new_facility_type, new_security, new_property_value,
                            {"loan_ranking": new_ranking, "interest_treatment": "Serviced",
                             "interest_advance_months": 1, "default_investor_target_irr": 8.0})
                        st.success(f"Added {tranche_name} with {new_ranking} ranking (ID {new_id}).")
                        st.rerun()

    if deals.empty:
        st.info("No deals have been created yet.")
        return
    status_filter = st.multiselect("Filter by status", sorted(deals["deal_status"].dropna().unique().tolist()),
                                   default=sorted(deals["deal_status"].dropna().unique().tolist()))
    filtered = deals[deals["deal_status"].isin(status_filter)]
    show_cols = ["id", "borrower_name", "deal_name", "facility_name", "facility_type", "loan_ranking", "facility_amount", "deal_status",
                 "settlement_date", "maturity_date", "security_type", "property_value"]
    display = filtered[[column for column in show_cols if column in filtered.columns]].rename(columns={
        "id": "Tranche ID", "borrower_name": "Borrower", "deal_name": "Deal", "facility_name": "Tranche", "facility_type": "Facility Type",
        "loan_ranking": "Ranking", "facility_amount": "Tranche Amount", "deal_status": "Status", "settlement_date": "Settlement",
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

    deal_labels = {f"{row['deal_name']} · {row['facility_name']} · {row['borrower_name']} · Tranche ID {int(row['id'])}": int(row["id"])
                   for _, row in deals.iterrows()}
    selected_label = st.selectbox("Select deal / tranche", list(deal_labels.keys()), key="workspace_deal_select")
    deal_id = deal_labels[selected_label]
    deal_row = deals[deals["id"] == deal_id].iloc[0]
    existing_params = safe_json(deal_row.get("params_json", "{}"), {})
    ensure_default_deal_investors(deal_id)
    deal_investors_df = load_deal_investors(deal_id)
    stored_return_overrides = safe_json(existing_params.get("investor_return_overrides", {}), {})
    if not isinstance(stored_return_overrides, dict):
        stored_return_overrides = {}
    legacy_shared_return_mode = "investor_target_irr" in existing_params
    if legacy_shared_return_mode:
        stored_return_overrides = {}  # The superseded facility-wide rate must not override deal defaults.
    try:
        return_override_version = int(existing_params.get("investor_return_overrides_version", 0) or 0)
    except (TypeError, ValueError):
        return_override_version = 0
    legacy_return_snapshot = return_override_version < 2 and not legacy_shared_return_mode
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
    legacy_mezzanine_type = facility_type_raw.strip().lower() == "mezzanine"
    facility_type_default = normalize_facility_type(facility_type_raw)
    legacy_facility_type_unknown = normalize_facility_type(facility_type_raw, fallback=None) is None
    loan_ranking_default = str(existing_params.get("loan_ranking", "Mezzanine" if legacy_mezzanine_type else "Senior"))
    if loan_ranking_default not in ["Senior", "Mezzanine"]:
        loan_ranking_default = "Senior"

    st.markdown(f"### {deal_row['deal_name']}  ·  {deal_row['facility_name']}  ·  {deal_row['borrower_name']}")
    st.caption(f"Tranche ID {deal_id}  |  Ranking: {loan_ranking_default}  |  Status: {deal_row['deal_status']}  |  Security: {deal_row['security_type']}")
    if existing_params.get("seed_key") == "demo_bridging_10m_bullet_v1":
        st.info("Illustrative demo only—not a confirmed transaction. Dates, rates, fees and other assumptions are editable in Setup & Terms.")
    if legacy_facility_type_unknown and not legacy_mezzanine_type:
        st.warning(f"Legacy facility type ‘{facility_type_raw}’ is no longer available. This deal is being treated as Bridging; choose Construction or Residual Stock if that is more appropriate, then save.")
    st.markdown('<div class="section-note">Edit assumptions or drawdowns once. Schedules, investor allocations, economics and cashflows recalculate from the current inputs; workspace changes autosave to the persistent database.</div>',
                unsafe_allow_html=True)

    tabs = st.tabs(["Overview", "Setup & Terms", "Drawdown Plan", "Loan Schedule",
                    "Actuals & Arrears", "Investor Base", "Economics", "SPV Cashflows",
                    "Fees & Settlement", "Export"])

    # ── Setup & terms ──────────────────────────────────────────
    prefix = f"deal_{deal_id}_"
    with tabs[1]:
        setup_tabs = st.tabs(["Tranche & Dates", "Returns & Allocation", "Fees & Covenants"])
        with setup_tabs[0]:
            st.subheader("Tranche terms and timeline")
            c1, c2, c3 = st.columns(3)
            init_widget_state(prefix + "facility_type", facility_type_default)
            if st.session_state[prefix + "facility_type"] not in FACILITY_TYPES:
                st.session_state[prefix + "facility_type"] = facility_type_default
            facility_type = c1.selectbox("Facility type", FACILITY_TYPES, key=prefix + "facility_type")
            init_widget_state(prefix + "facility_amount", facility_default)
            facility_amount = c2.number_input("Facility limit", min_value=0.0, max_value=10_000_000_000.0,
                step=100_000.0, key=prefix + "facility_amount", help="The facility limit constrains peak outstanding. Financed upfront costs and capitalised interest count against it; cash-serviced interest does not increase outstanding balance.")
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

            treatment_default = str(existing_params.get("interest_treatment", "Serviced"))
            if treatment_default.strip().lower() in ["capitalised", "capitalized"]:
                treatment_default = "Capitalised"
            else:
                treatment_default = "Serviced"
            init_widget_state(prefix + "interest_treatment", treatment_default)
            interest_treatment = st.selectbox("Interest treatment", ["Serviced", "Capitalised"],
                key=prefix + "interest_treatment",
                help="Serviced: borrower pays interest in cash and it does not increase peak outstanding, allowing a larger draw; lifetime cash paid may exceed the facility limit. Capitalised: interest is added to principal and must fit inside the cap.")

            stored_advance = existing_params.get("interest_advance_months", 1)
            all_advance_default = str(stored_advance).strip().lower() == "all"
            try:
                advance_months_default = int(stored_advance) if not all_advance_default else 1
            except (TypeError, ValueError):
                advance_months_default = 1
            advance_months_default = max(1, min(int(term_months), advance_months_default))
            init_widget_state(prefix + "interest_advance_all", all_advance_default)
            init_widget_state(prefix + "interest_advance_months", advance_months_default)
            adv1, adv2 = st.columns([1, 2])
            interest_advance_all = adv1.checkbox("All serviced interest for the full term in advance",
                key=prefix + "interest_advance_all", disabled=interest_treatment == "Capitalised")
            if st.session_state[prefix + "interest_advance_months"] > int(term_months):
                st.session_state[prefix + "interest_advance_months"] = int(term_months)
            months_label = ("Capitalisation interval (months)" if interest_treatment == "Capitalised"
                            else "Initial months’ interest in advance")
            months_help = ("No cash is collected up front. Accrued ACT/365 interest is added to principal every N settlement anniversaries; any remaining accrual is capitalised at maturity or payoff."
                if interest_treatment == "Capitalised" else
                "Collect this many months of interest in advance. After the initial coverage, interest is billed one month in advance each month.")
            interest_advance_months = adv2.number_input(months_label,
                min_value=1, max_value=int(term_months), step=1,
                key=prefix + "interest_advance_months",
                disabled=(interest_treatment == "Serviced" and interest_advance_all),
                help=months_help)
            interest_advance_setting = ("All" if interest_treatment == "Serviced" and interest_advance_all
                                        else int(interest_advance_months))
            capitalisation_interval_text = (f"{int(interest_advance_months)} month"
                if int(interest_advance_months) == 1 else f"{int(interest_advance_months)} months")
            net_advance_interest_from_proceeds = (interest_treatment == "Serviced"
                and str(interest_advance_setting).strip().lower() == "all"
                and normalize_facility_type(facility_type) in ["Bridging", "Residual Stock"])
            if interest_treatment == "Capitalised":
                st.caption(f"Interest is not collected in advance. It accrues ACT/365 on the outstanding balance for each period and is added to principal every {capitalisation_interval_text}, with the final accrual capitalised at maturity or payoff.")
            else:
                st.caption("The selected initial interest coverage is collected in advance; afterward, serviced interest is billed one month in advance.")

            construction_default = int(existing_params.get("construction_months", min(6, int(term_months))))
            construction_default = max(0, min(int(term_months), construction_default))
            init_widget_state(prefix + "construction_months", construction_default)
            if st.session_state[prefix + "construction_months"] > int(term_months):
                st.session_state[prefix + "construction_months"] = int(term_months)
            interest_only_default = int(existing_params.get("interest_only_months", construction_default))
            interest_only_default = max(0, min(int(term_months), interest_only_default))
            init_widget_state(prefix + "interest_only_months", interest_only_default)
            if st.session_state[prefix + "interest_only_months"] > int(term_months):
                st.session_state[prefix + "interest_only_months"] = int(term_months)

            c7, c8, c9 = st.columns(3)
            construction_months = c7.number_input("Construction period (months)", min_value=0,
                max_value=int(term_months), step=1, key=prefix + "construction_months",
                disabled=facility_type != "Construction",
                help="Construction drawdown and line-fee period. Only applies to Construction facilities.")
            interest_only_months = c8.number_input("Principal grace period (months)", min_value=0,
                max_value=int(term_months), step=1, key=prefix + "interest_only_months",
                help="Used by Standard amortising and CPR projections. Custom repayment dates take precedence; Bullet mode has no scheduled principal before maturity.")

            principal_mode_key = prefix + "principal_repayment_mode"
            principal_mode_type_key = prefix + "principal_repayment_default_type"
            previous_mode_type = st.session_state.get(principal_mode_type_key)
            if previous_mode_type is None:
                # A hot-reloaded session may retain the former hard-coded Standard default
                # for an unsaved legacy Bridging facility; correct it unless a custom mode
                # was already selected or a mode has been persisted to the deal.
                if ("principal_repayment_mode" not in existing_params
                        and st.session_state.get(principal_mode_key) == "Standard amortising"
                        and default_principal_repayment_mode(facility_type) == "Bullet at maturity"):
                    st.session_state[principal_mode_key] = "Bullet at maturity"
                st.session_state[principal_mode_type_key] = facility_type
            elif previous_mode_type != facility_type:
                current_mode = st.session_state.get(principal_mode_key,
                    existing_params.get("principal_repayment_mode",
                        default_principal_repayment_mode(previous_mode_type)))
                if current_mode == default_principal_repayment_mode(previous_mode_type):
                    st.session_state[principal_mode_key] = default_principal_repayment_mode(facility_type)
                st.session_state[principal_mode_type_key] = facility_type
            principal_mode_default = str(existing_params.get("principal_repayment_mode",
                default_principal_repayment_mode(facility_type)))
            init_widget_state(principal_mode_key, principal_mode_default)
            if st.session_state[principal_mode_key] not in PRINCIPAL_REPAYMENT_MODES:
                st.session_state[principal_mode_key] = default_principal_repayment_mode(facility_type)
            principal_repayment_mode = st.selectbox("Principal repayment pattern",
                PRINCIPAL_REPAYMENT_MODES, key=prefix + "principal_repayment_mode",
                help="Bridging facilities default to one maturity bullet. Residual Stock defaults to its standard equal-principal schedule. All facility types can use a custom schedule.")

            forecast_principal_dates = [add_months(settlement, month) for month in range(1, int(term_months) + 1)]
            principal_date_labels = {due.strftime("%d %b %Y"): due.isoformat()
                for due in forecast_principal_dates}
            stored_principal_rows = parse_principal_repayment_rows(
                existing_params.get("principal_repayment_schedule", []))
            principal_editor_rows = []
            for repayment in stored_principal_rows:
                due = parse_date(repayment["due_date"])
                label = due.strftime("%d %b %Y") if due else ""
                if label in principal_date_labels:
                    principal_editor_rows.append({"Due date": label,
                        "Amount basis": repayment["basis"], "Amount": repayment["amount"]})
            principal_editor_df = pd.DataFrame(principal_editor_rows,
                columns=["Due date", "Amount basis", "Amount"])
            principal_editor_fingerprint = hashlib.sha1(
                f"{settlement.isoformat()}|{int(term_months)}".encode("utf-8")).hexdigest()[:8]
            principal_editor_key = prefix + "principal_schedule_" + principal_editor_fingerprint
            with st.expander("Custom principal repayments", expanded=principal_repayment_mode == "Custom schedule"):
                st.caption("Forecast repayments are scheduled on the monthly settlement anniversaries. Choose either dollars or a percentage of the original tranche on each row. Any remaining principal is due at maturity. Actual receipts keep their actual value date; forecast-period interest continues through the scheduled anniversary.")
                edited_principal_plan = st.data_editor(principal_editor_df,
                    key=principal_editor_key, num_rows="dynamic", hide_index=True,
                    use_container_width=True, disabled=principal_repayment_mode != "Custom schedule",
                    column_config={
                        "Due date": st.column_config.SelectboxColumn("Forecast due date",
                            options=list(principal_date_labels.keys()), required=True),
                        "Amount basis": st.column_config.SelectboxColumn("Amount basis",
                            options=PRINCIPAL_REPAYMENT_BASES, default="Dollar amount", required=True),
                        "Amount": st.column_config.NumberColumn("Amount", min_value=0.0,
                            step=1000.0, format="%.2f", default=0.0, required=True),
                    })
            principal_repayment_schedule = []
            for _, repayment_row in edited_principal_plan.iterrows():
                due_value = repayment_row.get("Due date", "")
                due_label = "" if pd.isna(due_value) else str(due_value).strip()
                due_iso = principal_date_labels.get(due_label)
                try:
                    amount = float(repayment_row.get("Amount", 0.0))
                    if not np.isfinite(amount) or amount <= 0:
                        continue
                except (TypeError, ValueError):
                    continue
                basis_value = repayment_row.get("Amount basis", "Dollar amount")
                basis = "Dollar amount" if pd.isna(basis_value) else str(basis_value).strip()
                if due_iso and basis in PRINCIPAL_REPAYMENT_BASES:
                    principal_repayment_schedule.append({"due_date": due_iso,
                        "amount": amount, "basis": basis})
            principal_repayment_schedule = sorted(principal_repayment_schedule,
                key=lambda item: (item["due_date"], item["basis"], item["amount"]))
            if principal_repayment_mode == "Custom schedule" and not principal_repayment_schedule:
                st.info("Add one or more principal repayments above. Until then, the outstanding balance will be due at maturity.")

            init_widget_state(prefix + "borrower_irr", float(existing_params.get("borrower_irr", 12.0)))
            borrower_irr = c9.number_input("Borrower rate (% p.a.)", min_value=0.0, max_value=50.0,
                step=0.25, key=prefix + "borrower_irr")

            c10, c11, c12 = st.columns(3)
            init_widget_state(prefix + "cpr", float(existing_params.get("cpr", 0.0)))
            cpr = c10.number_input("Annual prepayment rate / CPR (%)", min_value=0.0, max_value=50.0,
                step=0.5, key=prefix + "cpr")
            init_widget_state(prefix + "drawdown_profile", str(existing_params.get("drawdown_profile", "Automatic")))
            if facility_type == "Construction":
                profile_options = ["Automatic", "Full at settlement", "S-curve", "Straight-line"]
                if st.session_state[prefix + "drawdown_profile"] not in profile_options:
                    st.session_state[prefix + "drawdown_profile"] = "Automatic"
            else:
                profile_options = ["Full at settlement"]
                st.session_state[prefix + "drawdown_profile"] = "Full at settlement"
            drawdown_profile = c11.selectbox("Default drawdown profile", profile_options,
                key=prefix + "drawdown_profile", disabled=facility_type != "Construction",
                help="Construction can use an S-curve or straight-line schedule. Bridging and Residual Stock draw in full at settlement.")
            init_widget_state(prefix + "s_curve_steepness", float(existing_params.get("s_curve_steepness", 5.0)))
            s_curve_active = (facility_type == "Construction"
                and drawdown_profile in ["Automatic", "S-curve"])
            s_curve_steepness = c12.slider("S-curve shape", min_value=1.0, max_value=9.0, step=0.5,
                key=prefix + "s_curve_steepness", disabled=not s_curve_active,
                help="Construction-only: higher values concentrate draws around the middle of the build period.")
            initial_security_type = str(deal_row.get("security_type", "First Mortgage") or "First Mortgage")
            st.caption(f"Security type: {initial_security_type}")

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
            st.caption("Each investor has separate Senior and Mezzanine deal-return defaults. Edit them per deal in Investor Portal → Deal Assignments; override either rate for this tranche in the table.")
            return_rows = []
            for _, investor_row in deal_investors_df.iterrows():
                investor_id = int(investor_row["id"])
                stored = stored_return_overrides.get(str(investor_id), stored_return_overrides.get(investor_id, {}))
                stored = dict(stored) if isinstance(stored, dict) else {}
                senior_default = safe_rate_pct(investor_row.get("deal_senior_target_irr",
                    investor_row.get("senior_target_irr", 8.0)), 8.0)
                mezz_default = safe_rate_pct(investor_row.get("deal_mezz_target_irr",
                    investor_row.get("mezz_target_irr", 8.0)), 8.0)
                if legacy_return_snapshot and stored:
                    global_senior = safe_rate_pct(investor_row.get("senior_target_irr", 8.0), 8.0)
                    global_mezz = safe_rate_pct(investor_row.get("mezz_target_irr", 8.0), 8.0)
                    try:
                        if abs(float(stored.get("senior", float("nan"))) - global_senior) < 1e-8:
                            stored.pop("senior", None)
                        if abs(float(stored.get("mezzanine", stored.get("mezz", float("nan")))) - global_mezz) < 1e-8:
                            stored.pop("mezzanine", None)
                            stored.pop("mezz", None)
                    except (TypeError, ValueError):
                        pass
                if stored:
                    senior_rate = stored.get("senior", senior_default)
                    mezz_rate = stored.get("mezzanine", stored.get("mezz", mezz_default))
                else:
                    senior_rate, mezz_rate = senior_default, mezz_default
                try:
                    senior_rate = float(senior_rate)
                    if not np.isfinite(senior_rate):
                        raise ValueError("non-finite Senior return")
                    senior_rate = max(0.0, min(50.0, senior_rate))
                except (TypeError, ValueError):
                    senior_rate = senior_default
                try:
                    mezz_rate = float(mezz_rate)
                    if not np.isfinite(mezz_rate):
                        raise ValueError("non-finite Mezzanine return")
                    mezz_rate = max(0.0, min(50.0, mezz_rate))
                except (TypeError, ValueError):
                    mezz_rate = mezz_default
                applied_rate = senior_rate if loan_ranking == "Senior" else mezz_rate
                return_rows.append({
                    "Investor ID": investor_id, "Investor": str(investor_row["name"]),
                    "Type": str(investor_row["investor_type"]),
                    "Senior allocation default (%)": float(investor_row.get("senior_default_pct", 0.0) or 0.0),
                    "Mezzanine allocation default (%)": float(investor_row.get("mezz_default_pct", 0.0) or 0.0),
                    "Senior return (% p.a.)": senior_rate,
                    "Mezzanine return (% p.a.)": mezz_rate,
                    "Applied return (% p.a.)": applied_rate,
                })
            return_columns = ["Investor ID", "Investor", "Type", "Senior allocation default (%)",
                "Mezzanine allocation default (%)", "Senior return (% p.a.)",
                "Mezzanine return (% p.a.)", "Applied return (% p.a.)"]
            return_matrix = pd.DataFrame(return_rows, columns=return_columns)
            return_fingerprint = hashlib.sha1(str(sorted(return_matrix["Investor ID"].astype(int).tolist())).encode()).hexdigest()[:8]
            return_rate_signature = return_matrix[["Investor ID", "Senior return (% p.a.)",
                "Mezzanine return (% p.a.)"]].to_json(orient="records")
            return_rate_fingerprint = hashlib.sha1(return_rate_signature.encode()).hexdigest()[:8]
            return_editor_key = prefix + "investor_returns_matrix_" + return_fingerprint + "_" + return_rate_fingerprint
            edited_returns = st.data_editor(return_matrix, key=return_editor_key, hide_index=True,
                use_container_width=True, num_rows="fixed", column_config={
                    "Investor ID": st.column_config.NumberColumn("Investor ID", disabled=True),
                    "Investor": st.column_config.TextColumn("Investor", disabled=True),
                    "Type": st.column_config.TextColumn("Type", disabled=True),
                    "Senior allocation default (%)": st.column_config.NumberColumn("Senior allocation default (%)", disabled=True),
                    "Mezzanine allocation default (%)": st.column_config.NumberColumn("Mezzanine allocation default (%)", disabled=True),
                    "Senior return (% p.a.)": st.column_config.NumberColumn("Senior return (% p.a.)",
                        min_value=0.0, max_value=50.0, step=0.25, format="%.2f", required=True),
                    "Mezzanine return (% p.a.)": st.column_config.NumberColumn("Mezzanine return (% p.a.)",
                        min_value=0.0, max_value=50.0, step=0.25, format="%.2f", required=True),
                    "Applied return (% p.a.)": st.column_config.NumberColumn("Applied return (% p.a.)", disabled=True),
                })
            investor_return_overrides = dict(stored_return_overrides)
            investor_return_rates = {}
            deal_defaults_by_id = {}
            for _, investor_row in deal_investors_df.iterrows():
                investor_id = int(investor_row["id"])
                deal_defaults_by_id[investor_id] = (
                    safe_rate_pct(investor_row.get("deal_senior_target_irr", investor_row.get("senior_target_irr", 8.0))),
                    safe_rate_pct(investor_row.get("deal_mezz_target_irr", investor_row.get("mezz_target_irr", 8.0))))
            return_defaults_by_id = {int(row["Investor ID"]): row for _, row in return_matrix.iterrows()}
            for _, return_row in edited_returns.iterrows():
                investor_id = int(return_row["Investor ID"])
                defaults = return_defaults_by_id.get(investor_id, {})
                try:
                    senior_rate = float(return_row["Senior return (% p.a.)"])
                    if not np.isfinite(senior_rate):
                        raise ValueError("non-finite Senior return")
                except (TypeError, ValueError):
                    senior_rate = float(defaults.get("Senior return (% p.a.)", 8.0))
                try:
                    mezz_rate = float(return_row["Mezzanine return (% p.a.)"])
                    if not np.isfinite(mezz_rate):
                        raise ValueError("non-finite Mezzanine return")
                except (TypeError, ValueError):
                    mezz_rate = float(defaults.get("Mezzanine return (% p.a.)", 8.0))
                senior_rate = max(0.0, min(50.0, senior_rate))
                mezz_rate = max(0.0, min(50.0, mezz_rate))
                selected_rate = senior_rate if loan_ranking == "Senior" else mezz_rate
                investor_return_overrides.pop(str(investor_id), None)
                investor_return_overrides.pop(investor_id, None)
                base_senior, base_mezz = deal_defaults_by_id.get(investor_id, (8.0, 8.0))
                tranche_override = {}
                if abs(senior_rate - base_senior) > 1e-8:
                    tranche_override["senior"] = senior_rate
                if abs(mezz_rate - base_mezz) > 1e-8:
                    tranche_override["mezzanine"] = mezz_rate
                if tranche_override:
                    investor_return_overrides[str(investor_id)] = tranche_override
                investor_return_rates[str(return_row["Investor"])] = selected_rate
            if not deal_investors_df.empty:
                deal_investors_df = deal_investors_df.copy()
                deal_investors_df["target_irr"] = deal_investors_df["name"].map(investor_return_rates).fillna(8.0)
            st.caption("The selected Loan ranking determines which investor-specific return is applied. Allocation defaults remain separate from return rates.")
            st.caption("Attach or remove optional investors (for example, CBFM or Syndicate) in the Investor Portal. Their allocation columns will then appear in the drawdown plan.")

        with setup_tabs[2]:
            st.subheader("Fees and security assumptions")
            f1, f2, f3, f4 = st.columns(4)
            init_widget_state(prefix + "estab_fee_pct", float(existing_params.get("estab_fee_pct", 1.5)))
            estab_fee_pct = f1.number_input("Establishment fee (%)", min_value=0.0, max_value=10.0,
                step=0.25, key=prefix + "estab_fee_pct")
            init_widget_state(prefix + "line_fee_pct", float(existing_params.get("line_fee_pct", 0.5)))
            line_fee_pct = f2.number_input("Undrawn line fee (% p.a.)", min_value=0.0, max_value=5.0,
                step=0.1, key=prefix + "line_fee_pct", disabled=facility_type != "Construction",
                help="Construction-only fee on undrawn facility balance. It is not charged for Bridging or Residual Stock.")
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
            penalty_rate = float(borrower_irr)
            c3.caption(f"Unpaid interest accrues at the borrower rate ({borrower_irr:.2f}% p.a.) and remains separate from principal; fees and principal shortfalls do not compound as arrears.")

    # Current inputs become the single source of truth for the live model.
    maturity_date = add_months(settlement, int(term_months))
    params = {
        "facility_amount": float(facility_amount), "facility_type": facility_type,
        "property_value": float(property_value), "settlement_date": settlement.isoformat(),
        "maturity_date": maturity_date.isoformat(), "term_months": int(term_months),
        "interest_treatment": interest_treatment,
        "interest_advance_months": interest_advance_setting,
        "default_investor_target_irr": 8.0,
        "investor_return_overrides_version": 2,
        "investor_return_overrides": investor_return_overrides,
        "investor_return_rates": investor_return_rates,
        "construction_months": int(construction_months),
        "interest_only_months": int(interest_only_months),
        "principal_repayment_mode": principal_repayment_mode,
        "principal_repayment_schedule": principal_repayment_schedule,
        "borrower_irr": float(borrower_irr),
        "cpr": float(cpr), "drawdown_profile": drawdown_profile,
        "s_curve_steepness": float(s_curve_steepness), "loan_ranking": loan_ranking,
        "estab_fee_pct": float(estab_fee_pct),
        "line_fee_pct": float(line_fee_pct), "admin_fee_monthly": float(admin_fee_monthly),
        "discharge_fee": float(discharge_fee), "risk_review_fee": float(risk_review_fee),
        "legal_fee": float(legal_fee), "other_upfront_fees": float(other_upfront_fees),
        "restructure_fee": float(restructure_fee), "pv_growth_annual": float(pv_growth_annual),
        "covenant_lvr": float(covenant_lvr), "penalty_rate": float(penalty_rate),
    }
    if existing_params.get("seed_key"):
        params["seed_key"] = existing_params["seed_key"]

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
    # The seed's suggested Capitalised draw deliberately retains 8% of the limit
    # for interest/charges; Serviced can draw the rest because cash interest does not
    # increase peak outstanding. Apply this to Reset-to-suggested as well as first seed.
    if existing_params.get("seed_key") == "demo_bridging_10m_bullet_v1":
        target_suggested_draw = (max(0.0, float(facility_amount) - upfront_total)
            if interest_treatment == "Serviced" else
            max(0.0, float(facility_amount) - upfront_total - 0.08 * float(facility_amount)))
        template_total = float(template["Drawdown Amount"].sum()) if not template.empty else 0.0
        if template_total > 0:
            template["Drawdown Amount"] = template["Drawdown Amount"] * (target_suggested_draw / template_total)
    drawdown_plan_source = existing_params.get("drawdown_schedule", {})
    # The seeded $10m illustration sizes its settlement draw according to treatment:
    # reserve 8% of the cap when interest capitalises; when serviced, cash interest
    # does not consume peak outstanding capacity, so the undrawn capacity can be drawn.
    previous_treatment = str(existing_params.get("interest_treatment", "Capitalised"))
    current_treatment = str(interest_treatment)
    if (existing_params.get("seed_key") == "demo_bridging_10m_bullet_v1"
            and previous_treatment != current_treatment):
        plan_to_resize = safe_json(drawdown_plan_source, {})
        if isinstance(plan_to_resize, dict):
            nonzero_rows = []
            for date_key, plan_item in plan_to_resize.items():
                item = plan_item if isinstance(plan_item, dict) else {"gross_draw": plan_item}
                amount = max(0.0, float(item.get("gross_draw", item.get("amount", 0.0)) or 0.0))
                if amount > 0.01:
                    nonzero_rows.append((date_key, item, amount))
            old_params = dict(existing_params)
            old_limit = float(old_params.get("facility_amount", facility_amount) or facility_amount)
            previous_upfront = compute_fees(old_params)["upfront_total"]
            previous_mode = ("Capitalised" if previous_treatment.strip().lower() in
                             ["capitalised", "capitalized"] else "Serviced")
            expected_previous_draw = (max(0.0, old_limit - previous_upfront)
                if previous_mode == "Serviced" else max(0.0, old_limit - previous_upfront - 0.08 * old_limit))
            if (len(nonzero_rows) == 1
                    and abs(nonzero_rows[0][2] - expected_previous_draw) <= 0.01
                    and str(nonzero_rows[0][1].get("allocation_source", "Standard")) == "Standard"
                    and str(nonzero_rows[0][1].get("status", "Forecast")) == "Forecast"):
                target_draw = (max(0.0, float(facility_amount) - upfront_total)
                    if current_treatment == "Serviced" else
                    max(0.0, float(facility_amount) - upfront_total - 0.08 * float(facility_amount)))
                nonzero_rows[0][1]["gross_draw"] = target_draw
                plan_to_resize[nonzero_rows[0][0]] = nonzero_rows[0][1]
                drawdown_plan_source = json.dumps(plan_to_resize)
    # Repair a legacy seeded forecast that drew almost the entire cap before the
    # capped-interest reserve was introduced. Do not alter actual or manual rows.
    if (existing_params.get("seed_key") == "demo_bridging_10m_bullet_v1"
            and current_treatment in ["Capitalised", "Capitalized", "Serviced"]):
        legacy_plan = safe_json(drawdown_plan_source, {})
        if isinstance(legacy_plan, dict):
            nonzero_rows = []
            for date_key, plan_item in legacy_plan.items():
                item = plan_item if isinstance(plan_item, dict) else {"gross_draw": plan_item}
                amount = max(0.0, float(item.get("gross_draw", item.get("amount", 0.0)) or 0.0))
                if amount > 0.01:
                    nonzero_rows.append((date_key, item, amount))
            cap_fees = compute_fees(params)
            old_full_limit_draw = float(facility_amount)
            old_fee_clamped_draw = max(0.0, float(facility_amount)
                - cap_fees["admin_total_estimate"] - cap_fees["line_fee_estimate"] - cap_fees["discharge"])
            legacy_amounts = [old_full_limit_draw, old_fee_clamped_draw]
            is_legacy_full_cap_draw = (len(nonzero_rows) == 1 and any(
                abs(nonzero_rows[0][2] - expected) <= 0.01 for expected in legacy_amounts))
            is_default_forecast = (is_legacy_full_cap_draw
                and str(nonzero_rows[0][1].get("allocation_source", "Standard")) == "Standard"
                and str(nonzero_rows[0][1].get("status", "Forecast")) == "Forecast")
            if is_default_forecast:
                corrected_draw = (max(0.0, float(facility_amount) - upfront_total)
                    if current_treatment == "Serviced" else
                    max(0.0, float(facility_amount) - upfront_total - 0.08 * float(facility_amount)))
                nonzero_rows[0][1]["gross_draw"] = corrected_draw
                legacy_plan[nonzero_rows[0][0]] = nonzero_rows[0][1]
                drawdown_plan_source = json.dumps(legacy_plan)
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
            if resolved_profile == "Straight-line" else "Single borrower draw at settlement by default; investors also fund actual upfront costs.")
        standard_text = ("Senior default: 50% BPCFMC / 15% BPCFLC / 35% CBCF."
            if loan_ranking == "Senior" else "Mezz default: 80% BPCFMC / 20% BPCFLC; CBCF is excluded by default.")
        st.markdown(f'<div class="section-note">{profile_description} {standard_text} Optional investors are included only when attached to this deal. Choose Standard, Investor Ledger or Manual for each row. Manual allocations are editable dollar amounts and should reconcile to borrower draw plus any upfront costs funded at settlement.</div>',
                    unsafe_allow_html=True)
        if interest_treatment == "Serviced":
            st.caption("Serviced interest is paid in cash and does not consume peak outstanding capacity. The borrower draw can use the facility limit less upfront costs funded at settlement; lifetime cash paid may exceed the limit.")
        else:
            st.caption("Investor Ledger rows use Capital Call movements recorded in the Investor Portal. Forecast rows default to the Senior/Mezz split. Required capital calls equal borrower draw plus upfront costs. Remaining capacity is reserved within the cap for capitalised interest and other charges.")
        reset_clicked = st.button("Reset to suggested schedule", key=prefix + "reset_drawdown_button")
        if reset_clicked:
            st.session_state[force_template_key] = True
            st.session_state[reset_count_key] = int(st.session_state[reset_count_key]) + 1
            st.rerun()

        use_template = bool(st.session_state.get(force_template_key, False)) or (
            stored_signature is not None and stored_signature != signature)
        initial_matrix = build_drawdown_matrix(template, drawdown_plan_source,
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
            "Gross Drawdown": st.column_config.NumberColumn("Borrower cash draw", min_value=0.0,
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
        net_total = float(plan_summary["Investor Funding Required"].sum()) if not plan_summary.empty else 0.0
        allocation_total = float(plan_summary[investor_names].sum().sum()) if investor_names and not plan_summary.empty else 0.0
        capacity_left = float(facility_amount) - net_total
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Borrower cash draw", money(gross_total))
        s2.metric("Investor funding required", money(net_total))
        s3.metric("Peak cap headroom" if interest_treatment == "Serviced" else "Capacity left for interest / fees", money(capacity_left))
        s4.metric("Allocation variance", money(net_total - allocation_total))
        if net_total > float(facility_amount) + 0.01:
            st.error("Borrower draws plus upfront costs exceed the facility's peak limit.")
        allocation_issues = plan_summary[plan_summary["Allocation Difference"].abs() > 0.01] if not plan_summary.empty else pd.DataFrame()
        if not allocation_issues.empty:
            issue_dates = ", ".join(value.strftime("%d %b %Y") for value in allocation_issues["Drawdown Date"])
            st.warning(f"Investor allocations do not reconcile to investor funding required (borrower draw plus upfront costs) on: {issue_dates}. Check the row’s Allocation Source or adjust its investor amounts.")

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

    # Persist current workspace inputs on every run, but only write when values changed.
    # This protects edits made on any tab from being lost on a session/app restart.
    params_changed = (json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
        != json.dumps(existing_params, sort_keys=True, separators=(",", ":"), default=str))
    if params_changed:
        update_deal_model(deal_id, params)
        workspace_saved_now = True
    else:
        workspace_saved_now = False

    # Build actuals and all model outputs from the current inputs and edited grid.
    actuals_df = load_actuals(deal_id)
    receipt_allocations_df = load_receipt_allocations(facility_deal_id=deal_id)
    actuals_map = actuals_to_map(actuals_df, settlement, term_months)
    actuals_map = merge_receipt_allocations_into_actuals(actuals_map, receipt_allocations_df)
    schedule_df, fees, borrower_xirr, borrower_cashflows = build_schedule(params, actuals_map, drawdown_map)
    investor_ledger_df, investor_summary_df, investor_cashflows = build_investor_ledger(
        schedule_df, params, deal_investors_df, investor_transactions_df, receipt_allocations_df)
    pnl = build_pnl(schedule_df, fees, investor_summary_df, params)
    cashflow_df = build_cashflows(schedule_df, investor_ledger_df, fees,
                                  investor_cashflows, receipt_allocations_df)

    if workspace_saved_now:
        st.success(f"Workspace changes saved to {DB.name}.")
    else:
        st.caption(f"All workspace assumptions and drawdown edits are saved in {DB.name}.")

    # ── Overview ────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("Live deal overview")
        st.caption("All figures below use the current on-screen assumptions and drawdown plan. Changes autosave; the button below can also be used to force a save.")
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Facility limit", money(facility_amount))
        k2.metric("Borrower cash drawn", money(schedule_df["Drawdown_Gross"].sum()))
        k3.metric("Outstanding at maturity", money(schedule_df["Ending_Balance"].iloc[-1]))
        k4.metric("Projected borrower XIRR", percent(borrower_xirr * 100.0))
        k5.metric("Investor target cost", percent(pnl["Weighted Investor Target (%)"]))
        k6.metric("Platform spread", f"{pnl['Platform Spread (bps)']:.0f} bps")
        st.markdown("#### Key dates")
        d1, d2, d3 = st.columns(3)
        d1.metric("Settlement", settlement.strftime("%d %b %Y"))
        d2.metric("Next anniversary", schedule_df["Payment_Date"].iloc[0].strftime("%d %b %Y"))
        d3.metric("Maturity anniversary", maturity_date.strftime("%d %b %Y"))
        st.markdown("#### Persistence status")
        st.caption("Terms, dates, fees, investor assumptions and the edited drawdown schedule autosave to SQLite. Use this button to force a save if needed.")
        if st.button("Save all changes now", type="primary", key=prefix + "save_all_changes"):
            update_deal_model(deal_id, params)
            st.success("Deal assumptions saved; actual drawdown allocations have been synced to the Investor Portal.")

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
            recognized_interest = float(schedule_df["Interest_Due"].sum()
                + schedule_df["Capitalised_Interest"].sum() + schedule_df["Penalty_Interest"].sum())
            income_data = pd.DataFrame({"Type": ["Interest", "Fees", "Investor cost", "Net revenue"],
                "Amount": [recognized_interest, float(pnl["Total Income"] - recognized_interest),
                    float(pnl["Investor Interest Expense"]), float(pnl["Net Revenue"])]})
            fig = px.bar(income_data, x="Type", y="Amount", color="Type",
                         color_discrete_sequence=["#4778b8", "#43a581", "#d96b68", "#6b83c4"])
            fig.update_layout(height=310, showlegend=False, margin=dict(t=20, b=20))
            fig.update_yaxes(tickprefix="$", tickformat=",.")
            st.plotly_chart(fig, use_container_width=True)

    # ── Borrower schedule and covenant monitoring ───────────────
    with tabs[3]:
        st.subheader("Borrower loan schedule")
        cap_interest_not_charged = float(schedule_df.get("Interest_Cap_Shortfall", pd.Series(dtype=float)).sum())
        if interest_treatment == "Capitalised" and cap_interest_not_charged > 0.01:
            st.warning(f"The facility limit binds: {money(cap_interest_not_charged)} of gross forecast interest is not charged because it would exceed the cap. The payoff and borrower yield reflect the cap, not the full stated rate.")
        breaches = schedule_df[schedule_df["Covenant_Breach"]]
        if not breaches.empty:
            breach_dates = ", ".join(value.strftime("%d %b %Y") for value in breaches["Payment_Date"])
            st.markdown(f'<div class="breach-alert">Covenant breach projected on: {breach_dates}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="covenant-ok">All projected LVRs are within the selected covenant.</div>', unsafe_allow_html=True)
        if interest_treatment == "Capitalised":
            st.caption(f"Capitalised interest accrues on ACT/365 against each period’s outstanding balance and is posted to principal every {capitalisation_interval_text}, with any final stub capitalised at maturity or payoff. It must fit inside the facility limit; ‘Interest Not Charged (cap)’ exposes any contractual accrual that did not fit.")
        else:
            st.caption("Interest in advance is collected on each Drawdown Date. ‘Due on Payment Date’ excludes that prepayment, so it is not counted twice at maturity.")
        if net_advance_interest_from_proceeds:
            st.caption("All-term serviced interest for this Bridging / Residual Stock facility is withheld from the initial proceeds. It is shown separately from cash interest due and is not charged again on the settlement date.")

        display_cols = ["Drawdown_Date", "Payment_Date", "Drawdown_Gross", "Upfront_Fees_Deducted",
            "Advance_Interest_Withheld", "Investor_Funding", "Drawdown_Net", "Opening_Balance",
            "Balance_After_Draw", "Gross_Interest_Accrued", "Interest_Accrued", "Interest_Cap_Shortfall",
            "Capitalised_Interest", "Balance_After_Capitalisation",
            "Interest_Due", "Cash_Interest_Advance", "Penalty_Interest", "Line_Fee", "Admin_Fee", "Discharge_Fee",
            "Scheduled_Principal", "Prepayment", "Maturity_Repayment", "Contract_Principal_Due",
            "Actual_Interest", "Actual_Principal", "Actual_Fees", "Total_Due",
            "Arrears", "Principal_Applied", "Ending_Balance", "Property_Value", "LVR_pct", "Covenant_Breach"]
        display = schedule_df[display_cols].rename(columns={
            "Drawdown_Date": "Drawdown Date", "Payment_Date": "Payment Date",
            "Drawdown_Gross": "Borrower Cash Draw", "Upfront_Fees_Deducted": "Upfront Costs Funded",
            "Advance_Interest_Withheld": "Advance Interest Withheld",
            "Investor_Funding": "Investor Funding", "Drawdown_Net": "Net Drawdown",
            "Opening_Balance": "Opening Balance", "Balance_After_Draw": "Balance After Draw",
            "Gross_Interest_Accrued": "Gross Interest (before cap)",
            "Interest_Accrued": "Interest Charged / Capitalised",
            "Interest_Cap_Shortfall": "Interest Not Charged (cap)",
            "Capitalised_Interest": "Capitalised Interest",
            "Balance_After_Capitalisation": "Balance After Capitalisation",
            "Interest_Due": "Interest Assessed", "Cash_Interest_Advance": "Cash Interest Advance",
            "Penalty_Interest": "Penalty Interest", "Line_Fee": "Line Fee", "Admin_Fee": "Admin Fee",
            "Discharge_Fee": "Discharge Fee", "Scheduled_Principal": "Scheduled Principal",
            "Prepayment": "Prepayment", "Maturity_Repayment": "Maturity Repayment",
            "Contract_Principal_Due": "Contract Principal Due",
            "Actual_Interest": "Actual Interest", "Actual_Principal": "Actual Principal",
            "Actual_Fees": "Actual Fees", "Total_Due": "Due on Payment Date", "Arrears": "Arrears",
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
        st.markdown('<div class="section-note">Select a forecast payment anniversary and log actual collections. For receipts on a different value date, record them in Borrower Servicing so the actual principal movement is dated correctly. Forecast-period interest still runs through the scheduled anniversary; the reduced balance carries into the next forecast period. Unpaid interest remains separate arrears, while fees and principal shortfalls are not added to principal.</div>',
                    unsafe_allow_html=True)
        payment_dates = [parse_date(value) for value in schedule_df["Payment_Date"].tolist()]
        selected_payment_date = st.selectbox("Payment / anniversary date", payment_dates,
            format_func=lambda value: value.strftime("%d %B %Y"), key=prefix + "actual_date_select")
        selected_row = schedule_df[schedule_df["Payment_Date"] == selected_payment_date].iloc[0]
        existing_actual = actuals_map.get(selected_payment_date.isoformat(), {})
        existing_override = bool(existing_actual.get("drawdown_override", False))
        existing_payment_log = bool(existing_actual.get("payment_actuals_logged", False))
        advance_note = (f" · interest advance on draw date **{money(selected_row['Cash_Interest_Advance'])}**"
            if float(selected_row.get("Cash_Interest_Advance", 0.0) or 0.0) > 0 else "")
        st.write(f"Draw period starts **{selected_row['Drawdown_Date'].strftime('%d %b %Y')}** · due on payment date **{money(selected_row['Total_Due'])}**{advance_note} · planned draw **{money(selected_row['Planned_Drawdown'])}**")
        if interest_treatment == "Capitalised":
            st.info(f"Capitalised interest accrues on ACT/365 and is added to principal every {capitalisation_interval_text} (and at payoff or maturity); no advance cash collection is made.")
        elif float(selected_row.get("Advance_Interest_Withheld", 0.0) or 0.0) > 0:
            st.info(f"{money(selected_row['Advance_Interest_Withheld'])} of advance interest was withheld from initial proceeds and is not included again in scheduled cash due.")

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
                max_value=1_000_000_000.0, step=100.0, key=interest_key,
                disabled=(not record_payments or interest_treatment == "Capitalised"))
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
        st.caption(f"Investor target returns are individual; the {loan_ranking} return is applied for this tranche. See Setup & Terms → Returns & Allocation for each investor’s rate and the alternate-ranking default.")
        st.info("Standard rows use the tranche ranking: Senior 50% / 15% / 35% (BPCFMC / BPCFLC / CBCF), or Mezzanine 80% / 20% (BPCFMC / BPCFLC). Investor Ledger and Manual rows use their recorded/entered splits. In Serviced mode, projected investor interest is paid only when borrower servicing is projected; actual-period payments are limited to borrower interest actually collected. In Capitalised mode, investor interest is added to balances at the selected interval. At repayment, contributed capital is shown under Principal Returned and any capitalised interest settled is shown under Interest Paid. Buyouts and actual repayments are applied from Investor Portal transactions.")
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
            "Interest_Paid": "${:,.0f}", "Capitalised_Interest": "${:,.0f}",
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
        economics_note = ("The limit applies to peak outstanding; cash-serviced interest is billed separately and may make lifetime receipts exceed the limit."
            if interest_treatment == "Serviced" else
            "Borrower draw, funded upfront costs, capitalised interest and scheduled charges are constrained within the facility limit.")
        st.caption("Income, investor interest expense and funded upfront-cost expense are modelled from the current contractual schedule. " + economics_note)
        st.dataframe(pnl_display, use_container_width=True, hide_index=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total income", money(pnl["Total Income"]))
        c2.metric("Investor interest expense", money(pnl["Investor Interest Expense"]))
        c3.metric("Net revenue after upfront costs", money(pnl["Net Revenue"]))
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
        cashflow_cap_note = ("The facility limit caps peak outstanding; cash-serviced interest and fees paid over the term are outside that peak-balance test."
            if interest_treatment == "Serviced" else
            "The facility limit includes borrower draws, funded upfront costs, capitalised interest and scheduled charges.")
        st.caption(cashflow_cap_note + " Investor calls, repayments, distributions and other movements are recorded on their transaction dates; forecast servicing and scheduled principal remain on settlement anniversaries.")
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
        upfront_funded_at_settlement = (min(fees["upfront_total"], first_draw)
            if first_draw > 0 else 0.0)
        waterfall = [
            ("Borrower cash draw", first_draw),
            ("Upfront costs funded at settlement", upfront_funded_at_settlement),
            ("Investor funding required", first_draw + upfront_funded_at_settlement),
            ("Capacity remaining inside facility cap", max(0.0,
                facility_amount - first_draw - upfront_funded_at_settlement)),
        ]
        left, right = st.columns([1, 1.25])
        with left:
            st.markdown("#### Settlement waterfall")
            st.dataframe(pd.DataFrame([{"Item": label, "Amount": money(amount)} for label, amount in waterfall]),
                         use_container_width=True, hide_index=True)
            st.caption("Upfront costs are funded by the investor capital call and included in peak outstanding at settlement. Cash-serviced interest is paid over the term and may make lifetime borrower receipts exceed the facility limit.")
        with right:
            st.markdown("#### Fee summary")
            fee_rows = [
                {"Fee": "Establishment", "Amount": money(fees["establishment"])},
                {"Fee": "Legal", "Amount": money(fees["legal_fee"])},
                {"Fee": "Risk review", "Amount": money(fees["risk_review"])},
                {"Fee": "Other upfront", "Amount": money(fees["other_upfront"])},
                {"Fee": "Total upfront costs (funded by investors)", "Amount": money(fees["upfront_total"])},
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
# PAGE: BORROWER SERVICING / RECEIPTS
# ═══════════════════════════════════════════════════════════════

def page_borrower_servicing():
    st.title("Borrower Servicing")
    st.caption("Record one borrower receipt at deal level, review the senior-first tranche waterfall, and override tranche allocations when needed.")
    deals = load_deals()
    roots = deals[deals["parent_deal_id"] == deals["id"]] if not deals.empty else deals
    if roots.empty:
        st.info("Create a deal and at least one facility before recording borrower receipts.")
        return

    root_labels = {f"{row['deal_name']} · {row['borrower_name']} · Deal ID {int(row['id'])}": int(row["id"])
                   for _, row in roots.iterrows()}
    root_label = st.selectbox("Deal", list(root_labels.keys()), key="borrower_receipt_deal")
    parent_id = root_labels[root_label]
    root_row = roots[roots["id"] == parent_id].iloc[0]
    facilities = deals[deals["parent_deal_id"] == parent_id].copy()
    st.caption("Record each receipt on its actual value date. Forecast repayment and interest periods remain anchored to settlement anniversaries: partial actual principal reduces the balance for the next forecast period, while current-period forecast interest continues through the anniversary. For a payoff, enter an interest-through date; the selected minimum prepaid period remains due, with excess advance interest credited.")

    c1, c2, c3 = st.columns(3)
    receipt_date = c1.date_input("Receipt / value date", value=date.today(), key=f"receipt_date_{parent_id}")
    receipt_amount = c2.number_input("Total borrower receipt", min_value=0.0,
        max_value=10_000_000_000.0, value=0.0, step=10_000.0, key=f"receipt_amount_{parent_id}")
    is_payoff = c3.checkbox("Final payoff", key=f"receipt_payoff_{parent_id}")
    interest_through = st.date_input("Interest calculated through", value=receipt_date,
        disabled=not is_payoff, key=f"receipt_interest_through_{parent_id}")
    notes = st.text_input("Receipt notes / bank reference", key=f"receipt_notes_{parent_id}")
    if receipt_amount > 0:
        proposed, unapplied = suggest_receipt_allocations(parent_id, receipt_date,
            receipt_amount, interest_through if is_payoff else None, is_payoff)
    else:
        proposed, unapplied = pd.DataFrame(), 0.0
    if proposed.empty:
        edited_allocations = proposed
        allocated_total = 0.0
    else:
        display = proposed.rename(columns={
            "facility_deal_id": "Tranche ID", "Facility": "Tranche", "Ranking": "Ranking",
            "period_date": "Service period", "fees": "Fees", "penalty_interest": "Arrears interest",
            "interest": "Interest", "principal": "Principal"})
        editor_key = f"receipt_allocations_{parent_id}_{receipt_date.isoformat()}_{int(round(receipt_amount))}_{int(is_payoff)}"
        edited_display = st.data_editor(display[["Tranche ID", "Tranche", "Ranking", "Service period",
            "fees_available", "penalty_available", "interest_available", "principal_available",
            "Fees", "Arrears interest", "Interest", "Principal"]],
            key=editor_key, hide_index=True, use_container_width=True, num_rows="fixed",
            column_config={
                "Tranche ID": st.column_config.NumberColumn("Tranche ID", disabled=True),
                "Tranche": st.column_config.TextColumn("Tranche", disabled=True),
                "Ranking": st.column_config.TextColumn("Rank", disabled=True),
                "Service period": st.column_config.DateColumn("Service period", disabled=True, format="DD MMM YYYY"),
                "fees_available": st.column_config.NumberColumn("Fees available", disabled=True, format="$%.0f"),
                "penalty_available": st.column_config.NumberColumn("Arrears interest available", disabled=True, format="$%.0f"),
                "interest_available": st.column_config.NumberColumn("Interest available", disabled=True, format="$%.0f"),
                "principal_available": st.column_config.NumberColumn("Principal available", disabled=True, format="$%.0f"),
                "Fees": st.column_config.NumberColumn("Fees", min_value=0.0, step=100.0, format="$%.0f"),
                "Arrears interest": st.column_config.NumberColumn("Arrears interest", min_value=0.0, step=100.0, format="$%.0f"),
                "Interest": st.column_config.NumberColumn("Interest", min_value=0.0, step=100.0, format="$%.0f"),
                "Principal": st.column_config.NumberColumn("Principal", min_value=0.0, step=100.0, format="$%.0f"),
            })
        edited_allocations = proposed.copy()
        for column, internal in [("Fees", "fees"), ("Arrears interest", "penalty_interest"),
                                 ("Interest", "interest"), ("Principal", "principal")]:
            edited_allocations[internal] = pd.to_numeric(edited_display[column], errors="coerce").fillna(0.0).values
        caps = {"fees": "fees_available", "penalty_interest": "penalty_available",
                "interest": "interest_available", "principal": "principal_available"}
        edited_allocations["override_flag"] = False
        for column, cap in caps.items():
            edited_allocations["override_flag"] |= (edited_allocations[column] - proposed[column]).abs() > 0.01
            edited_allocations["_over_cap"] = edited_allocations.get("_over_cap", False) | (
                edited_allocations[column] > edited_allocations[cap] + 0.01)
        allocated_total = float(edited_allocations[["fees", "penalty_interest", "interest", "principal"]].sum().sum())
        unapplied_amount = max(0.0, receipt_amount - allocated_total)
        st.metric("Allocated across facilities", money(allocated_total),
                  delta=f"Unapplied / refund {money(unapplied_amount)}")
    if st.button("Save borrower receipt and tranche split", type="primary", key=f"save_borrower_receipt_{parent_id}"):
        if receipt_amount <= 0:
            st.error("Enter a receipt amount greater than zero.")
        elif edited_allocations.empty:
            st.error("There are no facilities to allocate this receipt to.")
        elif allocated_total > receipt_amount + 0.01:
            st.error("Facility allocations cannot exceed the total receipt.")
        elif bool(edited_allocations.get("_over_cap", pd.Series([False])).any()):
            st.error("An allocation exceeds the amount currently due or principal available for that facility.")
        elif is_payoff and any(
            (edited_allocations[component] + 0.01 < proposed[cap]).any()
            for component, cap in [("fees", "fees_available"), ("penalty_interest", "penalty_available"),
                                  ("interest", "interest_available"), ("principal", "principal_available")]
        ):
            st.error("A final payoff must clear all currently available facility fees, arrears, interest and principal. Unapplied excess cash can remain as a refund/credit.")
        else:
            save_borrower_receipt(parent_id, receipt_date, receipt_amount,
                interest_through if is_payoff else None, is_payoff, notes, edited_allocations)
            st.success("Receipt saved. Facility schedules and investor cashflows will recalculate from this split.")
            st.rerun()

    receipts = load_borrower_receipts(parent_id)
    if not receipts.empty:
        st.markdown("#### Receipt history")
        allocation_history = load_receipt_allocations(parent_deal_id=parent_id)
        summary = receipts[["id", "receipt_date", "amount", "is_payoff", "notes"]].copy()
        if not allocation_history.empty:
            allocated_by_receipt = allocation_history.assign(
                _allocated=allocation_history[["fees", "penalty_interest", "interest", "principal"]].fillna(0).sum(axis=1)
            ).groupby("receipt_id")["_allocated"].sum()
            summary["Allocated"] = summary["id"].map(allocated_by_receipt).fillna(0.0)
        else:
            summary["Allocated"] = 0.0
        summary["Unapplied / refund"] = (summary["amount"] - summary["Allocated"]).clip(lower=0.0)
        summary = summary.rename(columns={"id": "Receipt ID", "receipt_date": "Receipt date",
            "amount": "Total received", "is_payoff": "Final payoff", "notes": "Notes"})
        st.dataframe(summary.style.format({column: "${:,.2f}" for column in
            ["Total received", "Allocated", "Unapplied / refund"]}),
            use_container_width=True, hide_index=True)
        if not allocation_history.empty:
            st.markdown("##### Facility allocation detail")
            history = allocation_history[["receipt_id", "receipt_date", "facility_name", "fees",
                "penalty_interest", "interest", "principal", "override_flag"]].rename(columns={
                "receipt_id": "Receipt ID", "receipt_date": "Receipt date", "facility_name": "Facility",
                "fees": "Fees", "penalty_interest": "Arrears interest", "interest": "Interest",
                "principal": "Principal", "override_flag": "Staff override"})
            st.dataframe(history.style.format({column: "${:,.2f}" for column in
                ["Fees", "Arrears interest", "Interest", "Principal"]}),
                use_container_width=True, hide_index=True)

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
    st.caption("Investor Register return rates are global starting defaults; set deal-specific Senior and Mezzanine rates under Deal Assignments. Optional investors are attached only to the tranches where they participate.")
    investor_tabs = st.tabs(["Investor Register", "Deal Assignments", "Record Movement", "Activity & Balances"])

    # Register investors
    with investor_tabs[0]:
        st.subheader("Investor register")
        with st.expander("＋ Add investor", expanded=False):
            with st.form("add_investor_form"):
                c1, c2, c3, c4 = st.columns(4)
                investor_name = c1.text_input("Investor name *")
                investor_type = c2.selectbox("Investor type", ["Fund", "CBFM", "Syndicate", "Institutional", "Individual", "Other"])
                senior_default_return = c3.number_input("Senior default return (% p.a.)",
                    min_value=0.0, max_value=50.0, value=8.0, step=0.25)
                mezz_default_return = c4.number_input("Mezzanine default return (% p.a.)",
                    min_value=0.0, max_value=50.0, value=8.0, step=0.25)
                investor_notes = st.text_input("Notes")
                if st.form_submit_button("Save investor", type="primary"):
                    if not investor_name.strip():
                        st.error("Investor name is required.")
                    else:
                        try:
                            save_investor(investor_name, investor_type, 8.0, investor_notes,
                                senior_target_irr=senior_default_return,
                                mezz_target_irr=mezz_default_return)
                            st.success(f"Added {investor_name.strip()}.")
                            st.rerun()
                        except sqlite3.IntegrityError:
                            st.error("An investor with that name already exists.")
        investors = load_investors(active_only=False)
        if investors.empty:
            st.info("No investors found.")
        else:
            st.dataframe(investors[["id", "name", "investor_type", "senior_target_irr",
                "mezz_target_irr", "status", "notes"]].rename(columns={
                "id": "ID", "name": "Investor", "investor_type": "Type",
                "senior_target_irr": "Senior default return (% p.a.)",
                "mezz_target_irr": "Mezzanine default return (% p.a.)",
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
                ec1, ec2, ec3, ec4 = st.columns(4)
                edited_name = ec1.text_input("Investor name", value=str(manage_row["name"]),
                    disabled=is_core, key=f"edit_investor_name_{int(manage_id)}")
                edited_type = ec2.selectbox("Investor type", type_options,
                    index=type_options.index(str(manage_row["investor_type"])),
                    key=f"edit_investor_type_{int(manage_id)}")
                edited_senior_return = ec3.number_input("Senior default return (% p.a.)",
                    min_value=0.0, max_value=50.0,
                    value=max(0.0, min(50.0, float(manage_row.get("senior_target_irr")
                        if pd.notna(manage_row.get("senior_target_irr")) else 8.0))),
                    step=0.25, key=f"edit_investor_senior_return_{int(manage_id)}")
                edited_mezz_return = ec4.number_input("Mezzanine default return (% p.a.)",
                    min_value=0.0, max_value=50.0,
                    value=max(0.0, min(50.0, float(manage_row.get("mezz_target_irr")
                        if pd.notna(manage_row.get("mezz_target_irr")) else 8.0))),
                    step=0.25, key=f"edit_investor_mezz_return_{int(manage_id)}")
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
                            update_investor(manage_id, edited_name, edited_type,
                                            float(manage_row["target_irr"] or 8.0),
                                            edited_status, edited_notes,
                                            senior_target_irr=edited_senior_return,
                                            mezz_target_irr=edited_mezz_return)
                            st.success("Investor details updated.")
                            st.rerun()
                        except sqlite3.IntegrityError:
                            st.error("An investor with that name already exists.")
            if is_core:
                st.caption("Core pool names and active status are fixed; Senior and Mezzanine return defaults remain editable.")

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

    deal_labels = {f"{row['deal_name']} · {row['facility_name']} · {row['borrower_name']} · Tranche ID {int(row['id'])}": int(row["id"])
                   for _, row in deals.iterrows()}
    label_by_deal = {value: key for key, value in deal_labels.items()}
    investor_labels = {int(row["id"]): str(row["name"]) for _, row in active_investors.iterrows()}

    # Attach optional investor entities to a deal
    with investor_tabs[1]:
        st.subheader("Tranche investor assignments")
        assignment_label = st.selectbox("Select tranche", list(deal_labels.keys()), key="investor_assign_deal")
        assignment_deal_id = deal_labels[assignment_label]
        current_assignment = load_deal_investors(assignment_deal_id)
        current_ids = [int(value) for value in current_assignment["id"].tolist()]
        assignment_key = f"investor_assignment_{assignment_deal_id}"
        if assignment_key not in st.session_state:
            st.session_state[assignment_key] = current_ids
        selected_ids = st.multiselect("Investors attached to this tranche", list(investor_labels.keys()),
            key=assignment_key, format_func=lambda value: investor_labels.get(int(value), str(value)))
        st.caption("Set investor return defaults for this selected tranche/deal here. They can differ on another deal; Investor Register rates seed new assignments. Newly added tranches inherit the parent deal’s current defaults, with independent overrides available in Deal Workspace.")
        edited_assignment = pd.DataFrame(columns=["Investor ID", "Investor", "Type",
            "Senior allocation default (%)", "Mezzanine allocation default (%)",
            "Senior deal return (% p.a.)", "Mezzanine deal return (% p.a.)"])
        if not current_assignment.empty:
            assignment_view = pd.DataFrame({
                "Investor ID": current_assignment["id"].astype(int),
                "Investor": current_assignment["name"].astype(str),
                "Type": current_assignment["investor_type"].astype(str),
                "Senior allocation default (%)": pd.to_numeric(current_assignment["senior_default_pct"], errors="coerce").fillna(0.0),
                "Mezzanine allocation default (%)": pd.to_numeric(current_assignment["mezz_default_pct"], errors="coerce").fillna(0.0),
                "Senior deal return (% p.a.)": current_assignment["deal_senior_target_irr"].map(safe_rate_pct),
                "Mezzanine deal return (% p.a.)": current_assignment["deal_mezz_target_irr"].map(safe_rate_pct),
            })
            assignment_fingerprint = hashlib.sha1(str(sorted(assignment_view["Investor ID"].tolist())).encode()).hexdigest()[:8]
            edited_assignment = st.data_editor(assignment_view,
                key=f"deal_return_matrix_{assignment_deal_id}_{assignment_fingerprint}",
                hide_index=True, use_container_width=True, num_rows="fixed", column_config={
                    "Investor ID": st.column_config.NumberColumn("Investor ID", disabled=True),
                    "Investor": st.column_config.TextColumn("Investor", disabled=True),
                    "Type": st.column_config.TextColumn("Type", disabled=True),
                    "Senior allocation default (%)": st.column_config.NumberColumn("Senior allocation default (%)", disabled=True),
                    "Mezzanine allocation default (%)": st.column_config.NumberColumn("Mezzanine allocation default (%)", disabled=True),
                    "Senior deal return (% p.a.)": st.column_config.NumberColumn("Senior deal return (% p.a.)",
                        min_value=0.0, max_value=50.0, step=0.25, format="%.2f", required=True),
                    "Mezzanine deal return (% p.a.)": st.column_config.NumberColumn("Mezzanine deal return (% p.a.)",
                        min_value=0.0, max_value=50.0, step=0.25, format="%.2f", required=True),
                })
        if st.button("Save tranche assignments and deal returns", type="primary",
                     key=f"save_assignments_{assignment_deal_id}"):
            senior_rates = {int(row["Investor ID"]): safe_rate_pct(row["Senior deal return (% p.a.)"])
                for _, row in edited_assignment.iterrows()}
            mezz_rates = {int(row["Investor ID"]): safe_rate_pct(row["Mezzanine deal return (% p.a.)"])
                for _, row in edited_assignment.iterrows()}
            save_deal_investors(assignment_deal_id, selected_ids,
                senior_target_returns=senior_rates, mezz_target_returns=mezz_rates)
            st.success("Tranche assignments and deal-specific returns were saved.")
            st.rerun()

    # Record actual movements; buyouts are entered as a single total with a user-defined seller split.
    with investor_tabs[2]:
        st.subheader("Record an investor movement")
        movement_label = st.selectbox("Deal / tranche", list(deal_labels.keys()), key="movement_deal")
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
                "Tranche": transactions["facility_name"],
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
    "💰 Borrower Servicing": page_borrower_servicing,
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
