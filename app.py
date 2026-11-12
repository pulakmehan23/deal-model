import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import brentq
import json
import io
from datetime import datetime

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(page_title="Structured Lending Deal Model",
                   layout="wide", page_icon="🏦")
st.title("🏦 Structured Lending Deal Model v4")
st.caption("IRR Pricing · BBSY · Tranching · Drawdowns · Prepayment · "
           "Fee Engine · LVR · TLS · Fund / Syndicate / CBFM")

# ═══════════════════════════════════════════════════════════════
#  FINANCIAL ENGINE
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


# ── FEE ENGINE ───────────────────────────────────────────────
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


# ── SCHEDULE BUILDER ─────────────────────────────────────────
def build_schedule(facility, target_irr, n_months,
                   interest_timing="arrears", advance_n=1,
                   capitalise_months=0,
                   bbsy_curve=None, margin=0.0, reset_freq=12,
                   drawdown_events=None, cpr=0.0,
                   fee_config=None,
                   default_month=None, restructure_month=None,
                   restructure_amount=0, construction_end=0,
                   property_value=0, property_growth=0):
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
        dd_amt = 0.0
        while dd_idx < len(drawdown_events) and drawdown_events[dd_idx][0] == m:
            dd_amt += drawdown_events[dd_idx][1]
            dd_idx += 1
        balance += dd_amt

        is_def = default_month is not None and m == default_month
        is_res = restructure_month is not None and m == restructure_month
        is_exit = m == n_months or (balance <= 0 and m > 0)

        month_fees = compute_fees(
            facility, balance, m, fee_config,
            is_def, is_res, restructure_amount, is_exit, construction_end)
        total_fees = sum(month_fees.values())

        # Property value (monthly compounding growth)
        prop_val = property_value * (1 + property_growth/12) ** m if property_value > 0 else 0
        lvr = balance / prop_val * 100 if prop_val > 0 else 0
        current_limit = max(facility - cum_principal, 0)

        if m == 0:
            net_disb = dd_amt - total_fees
            rows.append({
                "Month": m, "Drawdown": dd_amt,
                "Opening": balance, "Rate": get_rate(m),
                "Interest": 0.0, "Interest Paid": 0.0,
                "Interest Capitalised": 0.0,
                "Principal": 0.0, "Prepayment": 0.0,
                "Fees": total_fees, "Fee Detail": month_fees,
                "Total Payment": -net_disb,
                "Closing": balance,
                "Original Limit": facility,
                "Current Limit": current_limit,
                "Property Value": prop_val, "LVR": lvr,
                "Borrower CF": net_disb,
                "Anniversary": False
            })
            continue

        rate_m = get_rate(m - 1)
        interest = balance * rate_m
        is_cap = m <= capitalise_months
        cap_amt = interest if is_cap else 0.0
        cash_int = 0.0 if is_cap else interest

        if is_cap:
            balance += cap_amt

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

        total_payment = cash_int + total_princ + total_fees
        current_limit = max(facility - cum_principal, 0)

        rows.append({
            "Month": m, "Drawdown": dd_amt,
            "Opening": balance + total_princ,
            "Rate": rate_m,
            "Interest": interest,
            "Interest Paid": cash_int,
            "Interest Capitalised": cap_amt,
            "Principal": sched_princ,
            "Prepayment": prepay,
            "Fees": total_fees, "Fee Detail": month_fees,
            "Total Payment": total_payment,
            "Closing": max(balance, 0),
            "Original Limit": facility,
            "Current Limit": current_limit,
            "Property Value": prop_val,
            "LVR": max(balance, 0) / prop_val * 100 if prop_val > 0 else 0,
            "Borrower CF": -total_payment,
            "Anniversary": m % 12 == 0
        })

    df = pd.DataFrame(rows)

    # Interest timing adjustments
    if interest_timing == "all_advance":
        total_int = df["Interest"].sum()
        df.loc[0, "Borrower CF"] -= total_int
        df.loc[0, "Total Payment"] = -df.loc[0, "Borrower CF"]
        df.loc[0, "Interest Paid"] = total_int
        for i in range(1, len(df)):
            df.loc[i, "Interest Paid"] = 0.0
            df.loc[i, "Borrower CF"] = -(df.loc[i, "Principal"]
                                          + df.loc[i, "Prepayment"]
                                          + df.loc[i, "Fees"])
            df.loc[i, "Total Payment"] = -df.loc[i, "Borrower CF"]

    elif interest_timing == "advance":
        for i in range(1, len(df)):
            if (i - 1) % advance_n == 0:
                block = sum(df.loc[j, "Interest"]
                            for j in range(i, min(i + advance_n, len(df))))
                df.loc[i, "Interest Paid"] = block
                df.loc[i, "Total Payment"] += block - df.loc[i, "Interest"]
                df.loc[i, "Borrower CF"] = -df.loc[i, "Total Payment"]
                for j in range(i + 1, min(i + advance_n, len(df))):
                    df.loc[j, "Interest Paid"] = 0.0

    return df


# ═══════════════════════════════════════════════════════════════
#  DEFAULT PARAMS (for save/load)
# ═══════════════════════════════════════════════════════════════
DEFAULT_PARAMS = {
    "facility": 10_000_000, "target_irr": 12.0, "tenure": 36,
    "construction_end": 0, "interest_timing": "arrears",
    "advance_n": 3, "capitalise_months": 0,
    "use_floating": False, "margin": 5.0, "reset_freq": 12,
    "n_dd": 1, "cpr": 0.0, "pd_annual": 2.0, "lgd": 40.0,
    "est_pct": 1.5, "line_pct": 1.0, "admin_type": "fixed",
    "admin_amount": 1000, "admin_pct": 0.5, "discharge_amt": 2000,
    "risk_review_amt": 5000, "restructure_pct": 1.0,
    "property_value": 15_000_000, "property_growth": 3.0,
    "n_tranches": 1
}


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════
st.sidebar.header("⚙️ Deal Parameters")

with st.sidebar.expander("📋 Loan Structure", expanded=True):
    facility = st.number_input("Facility / Original Limit ($)",
                                100_000, 500_000_000, 10_000_000, step=1_000_000)
    target_irr = st.slider("Target IRR (%)", 1.0, 35.0, 12.0, 0.25) / 100
    tenure = st.slider("Tenure (months)", 3, 120, 36)
    construction_end = st.slider("Construction End (month)", 0, tenure, 0)

with st.sidebar.expander("🏠 Collateral / LVR", expanded=True):
    property_value = st.number_input("Initial Property Value ($)",
                                      0, 1_000_000_000, 15_000_000,
                                      step=500_000)
    property_growth = st.slider("Property Growth (% p.a.)", -10.0, 20.0, 3.0, 0.5) / 100
    initial_lvr = facility / property_value * 100 if property_value > 0 else 0
    st.metric("Initial LVR", f"{initial_lvr:.1f}%")

with st.sidebar.expander("🕐 Interest Timing", expanded=True):
    interest_timing = st.selectbox("Timing",
        ["arrears", "advance", "all_advance"],
        format_func=lambda x: {
            "arrears": "In Arrears",
            "advance": "In Advance (N-month)",
            "all_advance": "All In Advance"
        }[x])
    advance_n = 1
    if interest_timing == "advance":
        advance_n = st.selectbox("Advance Period",
                                  [1, 2, 3, 4, 5, 6, 12], index=2)
    capitalise_months = st.slider("Capitalisation (months)", 0, tenure, 0)

with st.sidebar.expander("📈 BBSY Floating Rate", expanded=False):
    use_floating = st.checkbox("Use BBSY", value=False)
    bbsy_curve = None
    margin = 0.0
    reset_freq = 12
    if use_floating:
        n_resets = max(1, tenure // 12 + 1)
        bbsy_inputs = []
        for i in range(n_resets):
            r = st.number_input(f"Period {i+1} BBSY (%)", 0.0, 15.0,
                                 4.35, 0.05, key=f"bbsy_{i}")
            bbsy_inputs.append(r / 100)
        bbsy_curve = bbsy_inputs
        margin = st.slider("Margin over BBSY (%)", 0.0, 15.0, 5.0, 0.25) / 100
        reset_freq = st.selectbox("Reset Frequency", [3, 6, 12], index=2,
                                   format_func=lambda x: {
                                       3: "Quarterly", 6: "Semi-Annual",
                                       12: "Anniversary"}[x])

with st.sidebar.expander("💰 Drawdown Schedule", expanded=False):
    n_dd = st.number_input("Number of Drawdowns", 1, 10, 1)
    dd_events = []
    if n_dd == 1:
        dd_events = [(0, facility)]
    else:
        remaining = facility
        for i in range(n_dd):
            c1, c2 = st.columns(2)
            dd_m = c1.number_input(f"DD {i+1} Month", 0, tenure-1,
                                    i * (tenure // n_dd), key=f"dd_m_{i}")
            if i < n_dd - 1:
                dd_a = c2.number_input(f"DD {i+1} ($)", 0, int(remaining),
                                        int(remaining // (n_dd - i)),
                                        step=100_000, key=f"dd_a_{i}")
            else:
                dd_a = remaining
                c2.write(f"**${dd_a:,.0f}**")
            dd_events.append((dd_m, dd_a))
            remaining -= dd_a

with st.sidebar.expander("💵 Fee Structure", expanded=False):
    est_pct = st.slider("Establishment (% of facility)", 0.0, 5.0, 1.5, 0.25) / 100
    line_pct = st.slider("Line Fee (% p.a., construction)", 0.0, 5.0, 1.0, 0.25) / 100
    admin_type = st.radio("Admin Fee Type", ["fixed", "% of outstanding"],
                           horizontal=True)
    admin_amount = 0.0
    admin_pct = 0.0
    if admin_type == "fixed":
        admin_amount = st.number_input("Admin ($/month)", 0, 50_000, 1_000, step=100)
    else:
        admin_pct = st.slider("Admin (% p.a.)", 0.0, 3.0, 0.5, 0.1) / 100
    discharge_amt = st.number_input("Discharge ($)", 0, 100_000, 2_000, step=500)
    risk_review_amt = st.number_input("Risk Review ($)", 0, 100_000, 5_000, step=500)
    restructure_pct = st.slider("Restructure (% of facility)", 0.0, 5.0, 1.0, 0.25) / 100

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
    default_month = None
    if trigger_default:
        default_month = st.number_input("Default Month", 1, tenure, 12)
    trigger_restructure = st.checkbox("Simulate Restructure", False)
    restructure_month = None
    restructure_amount = 0
    if trigger_restructure:
        restructure_month = st.number_input("Restructure Month", 1, tenure, 18)
        restructure_amount = st.number_input("Restructured ($)", 0, facility,
                                              facility // 2, step=100_000)

with st.sidebar.expander("🏗️ Tranche / Investor", expanded=False):
    n_tranches = st.number_input("Number of Tranches", 1, 5, 1)
    tranches = []
    names = ["Senior A", "Senior B", "Mezzanine", "Junior", "Equity"]
    inv_types = ["Fund", "Syndicate", "CBFM"]
    rem = facility
    for i in range(n_tranches):
        st.markdown(f"**{names[i]}**")
        c1, c2, c3 = st.columns(3)
        if i < n_tranches - 1:
            ts = c1.number_input("Size ($)", 0, int(rem),
                                  int(rem // (n_tranches - i)),
                                  step=100_000, key=f"ts_{i}")
        else:
            ts = rem
            c1.write(f"**${ts:,.0f}**")
        tr = c2.slider("Rate (%)", 0.0, 30.0, 8.0 + i*3, 0.25, key=f"tr_{i}") / 100
        ti = c3.selectbox("Investor", inv_types, key=f"ti_{i}")
        tranches.append({"name": names[i], "size": ts, "rate": tr,
                          "rate_m": to_monthly(tr), "priority": i+1,
                          "investor": ti})
        rem -= ts

# ═══════════════════════════════════════════════════════════════
#  COMPUTE
# ═══════════════════════════════════════════════════════════════
sched = build_schedule(
    facility, target_irr, tenure,
    interest_timing, advance_n, capitalise_months,
    bbsy_curve, margin, reset_freq,
    dd_events, cpr, fee_config,
    default_month, restructure_month,
    restructure_amount, construction_end,
    property_value, property_growth
)

borrower_cfs = list(sched["Borrower CF"])
borrower_irr = calc_irr(borrower_cfs)
investor_cfs = [-facility] + list(sched["Total Payment"].iloc[1:])
investor_irr = calc_irr(investor_cfs)

total_fees = sched["Fees"].sum()
total_interest = sched["Interest"].sum()
total_principal = sched["Principal"].sum() + sched["Prepayment"].sum()

# Fee breakdown
all_fee_details = []
for _, row in sched.iterrows():
    if isinstance(row["Fee Detail"], dict):
        for k, v in row["Fee Detail"].items():
            if v > 0:
                all_fee_details.append({"Type": k, "Amount": v, "Month": row["Month"]})
fee_df = pd.DataFrame(all_fee_details) if all_fee_details else pd.DataFrame(
    columns=["Type", "Amount", "Month"])

# ═══════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════
t1, t2, t3, t4, t5, t6, t7, t8, t9 = st.tabs([
    "📊 Overview", "📅 Cash Flows", "💵 Fees",
    "🏗️ Tranches", "💼 Investors", "⚠️ Risk",
    "🏦 Fund", "📋 TLS", "💾 Scenarios"
])

# ── TAB 1: OVERVIEW ──────────────────────────────────────────
with t1:
    st.subheader("Deal Economics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Facility", f"${facility:,.0f}")
    c2.metric("Target IRR", f"{target_irr*100:.2f}%")
    c3.metric("Borrower IRR",
              f"{borrower_irr*100:.2f}%" if not np.isnan(borrower_irr) else "—")
    c4.metric("Investor IRR",
              f"{investor_irr*100:.2f}%" if not np.isnan(investor_irr) else "—")
    c5.metric("Initial LVR", f"{initial_lvr:.1f}%")

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 🙋 Borrower")
        st.metric("Total Drawn", f"${sched['Drawdown'].sum():,.0f}")
        st.metric("Total Interest", f"${total_interest:,.0f}")
        st.metric("Total Fees", f"${total_fees:,.0f}")
        st.metric("Total Repaid", f"${sched['Total Payment'].iloc[1:].sum():,.0f}")
        st.metric("Timing", {"arrears": "Arrears", "advance": f"{advance_n}M Adv",
                              "all_advance": "All Adv"}[interest_timing])
        if capitalise_months:
            st.metric("PIK", f"{capitalise_months}m")

    with c2:
        st.markdown("### 🏢 Platform")
        st.metric("Interest Revenue", f"${total_interest:,.0f}")
        st.metric("Fee Revenue", f"${total_fees:,.0f}")
        st.metric("Total Revenue", f"${total_interest + total_fees:,.0f}")
        st.metric("Rev / Facility",
                  f"{(total_interest + total_fees)/facility*100:.2f}%")

    with c3:
        st.markdown("### 💼 Investor")
        st.metric("Deployed", f"${facility:,.0f}")
        st.metric("Total Return",
                  f"${sched['Total Payment'].iloc[1:].sum() - facility:,.0f}")
        st.metric("Gross IRR",
                  f"{investor_irr*100:.2f}%" if not np.isnan(investor_irr) else "—")
        el = pd_annual * lgd
        st.metric("EL", f"{el*100:.2f}%")
        if not np.isnan(investor_irr):
            st.metric("Net IRR", f"{(investor_irr - el)*100:.2f}%")

    # Charts row
    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sched["Month"], y=sched["Rate"]*100,
                                  name="Rate (%)", line=dict(color="orange")))
        fig.add_trace(go.Scatter(x=sched["Month"], y=sched["Closing"]/1e6,
                                  name="Balance ($M)", yaxis="y2",
                                  line=dict(color="blue")))
        if capitalise_months > 0:
            fig.add_vrect(x0=0, x1=capitalise_months,
                          fillcolor="purple", opacity=0.1, annotation_text="PIK")
        if construction_end > 0:
            fig.add_vrect(x0=0, x1=construction_end,
                          fillcolor="green", opacity=0.08,
                          annotation_text="Construction")
        for m in sched[sched["Anniversary"]]["Month"]:
            if m > 0:
                fig.add_vline(x=m, line_dash="dash", line_color="gray",
                              annotation_text=f"Yr{m//12}")
        fig.update_layout(height=350, yaxis=dict(title="Rate (%)"),
                          yaxis2=dict(title="Bal ($M)", overlaying="y", side="right"))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig_lvr = go.Figure()
        if property_value > 0:
            fig_lvr.add_trace(go.Scatter(
                x=sched["Month"], y=sched["LVR"],
                name="Dynamic LVR (%)", line=dict(color="red", width=2)))
            fig_lvr.add_trace(go.Scatter(
                x=sched["Month"], y=sched["Property Value"]/1e6,
                name="Property Value ($M)", yaxis="y2",
                line=dict(color="green", dash="dot")))
            fig_lvr.add_hline(y=80, line_dash="dash", line_color="orange",
                              annotation_text="80% LVR")
            fig_lvr.add_hline(y=70, line_dash="dash", line_color="green",
                              annotation_text="70% LVR")
        fig_lvr.update_layout(height=350, yaxis=dict(title="LVR (%)"),
                               yaxis2=dict(title="Prop Val ($M)",
                                           overlaying="y", side="right"))
        st.plotly_chart(fig_lvr, use_container_width=True)

    # Limit chart
    st.markdown("#### Limit Utilisation")
    fig_lim = go.Figure()
    fig_lim.add_trace(go.Scatter(x=sched["Month"], y=sched["Original Limit"]/1e6,
                                  name="Original Limit", line=dict(dash="dash", color="gray")))
    fig_lim.add_trace(go.Scatter(x=sched["Month"], y=sched["Current Limit"]/1e6,
                                  name="Current Limit", line=dict(color="blue")))
    fig_lim.add_trace(go.Scatter(x=sched["Month"], y=sched["Closing"]/1e6,
                                  name="Outstanding", fill="tozeroy",
                                  line=dict(color="red")))
    fig_lim.update_layout(height=300, yaxis_title="$M", xaxis_title="Month")
    st.plotly_chart(fig_lim, use_container_width=True)

# ── TAB 2: CASH FLOWS ────────────────────────────────────────
with t2:
    st.subheader("Cash Flow Schedule")
    fig_cf = go.Figure()
    fig_cf.add_trace(go.Bar(x=sched["Month"].iloc[1:], y=sched["Interest Paid"].iloc[1:],
                             name="Interest Paid", marker_color="#FF9800"))
    fig_cf.add_trace(go.Bar(x=sched["Month"].iloc[1:], y=sched["Principal"].iloc[1:],
                             name="Principal", marker_color="#2196F3"))
    fig_cf.add_trace(go.Bar(x=sched["Month"].iloc[1:], y=sched["Prepayment"].iloc[1:],
                             name="Prepayment", marker_color="#4CAF50"))
    fig_cf.add_trace(go.Bar(x=sched["Month"].iloc[1:], y=sched["Fees"].iloc[1:],
                             name="Fees", marker_color="#9C27B0"))
    if capitalise_months > 0:
        fig_cf.add_trace(go.Bar(x=sched["Month"].iloc[1:],
                                 y=sched["Interest Capitalised"].iloc[1:],
                                 name="PIK", marker_color="#607D8B"))
    fig_cf.update_layout(barmode="stack", height=400)
    st.plotly_chart(fig_cf, use_container_width=True)

    with st.expander("📋 Full Schedule"):
        cols = ["Month", "Drawdown", "Opening", "Rate", "Interest",
                "Interest Paid", "Interest Capitalised",
                "Principal", "Prepayment", "Fees",
                "Total Payment", "Closing", "Current Limit", "LVR"]
        d = sched[cols].copy()
        d["Rate"] = d["Rate"].apply(lambda x: f"{x*100:.3f}%")
        d["LVR"] = d["LVR"].apply(lambda x: f"{x:.1f}%")
        for c in cols:
            if c not in ["Month", "Rate", "LVR"]:
                d[c] = d[c].apply(lambda x: f"${x:,.2f}")
        st.dataframe(d, use_container_width=True, height=500)

# ── TAB 3: FEES ──────────────────────────────────────────────
with t3:
    st.subheader("💵 Fee Breakdown")
    if fee_df.empty:
        st.info("No fees generated.")
    else:
        fee_summary = fee_df.groupby("Type")["Amount"].sum().reset_index()
        fee_summary = fee_summary.sort_values("Amount", ascending=False)
        fee_summary["% of Total"] = fee_summary["Amount"] / fee_summary["Amount"].sum() * 100

        c1, c2 = st.columns(2)
        with c1:
            d = fee_summary.copy()
            d["Amount"] = d["Amount"].apply(lambda x: f"${x:,.2f}")
            d["% of Total"] = d["% of Total"].apply(lambda x: f"{x:.1f}%")
            st.dataframe(d, use_container_width=True, hide_index=True)
            st.metric("Total Fees", f"${total_fees:,.2f}")
            st.metric("Fees / Facility", f"{total_fees/facility*100:.2f}%")
        with c2:
            st.plotly_chart(px.pie(fee_summary, values="Amount", names="Type",
                                    hole=0.4), use_container_width=True)

        fig_t = px.bar(fee_df, x="Month", y="Amount", color="Type",
                       barmode="stack", title="Fees by Month")
        fig_t.update_layout(height=300)
        st.plotly_chart(fig_t, use_container_width=True)

        # IRR impact
        sched_nf = build_schedule(
            facility, target_irr, tenure, interest_timing, advance_n,
            capitalise_months, bbsy_curve, margin, reset_freq,
            dd_events, cpr,
            {k: 0 for k in fee_config},
            default_month, restructure_month, restructure_amount,
            construction_end, property_value, property_growth)
        irr_nf = calc_irr(list(sched_nf["Borrower CF"]))
        c1, c2, c3 = st.columns(3)
        c1.metric("IRR (no fees)", f"{irr_nf*100:.2f}%" if not np.isnan(irr_nf) else "—")
        c2.metric("IRR (with fees)", f"{borrower_irr*100:.2f}%" if not np.isnan(borrower_irr) else "—")
        if not np.isnan(borrower_irr) and not np.isnan(irr_nf):
            c3.metric("Fee Drag", f"{(borrower_irr - irr_nf)*100:.2f}%")

# ── TAB 4: TRANCHES ──────────────────────────────────────────
with t4:
    st.subheader("🏗️ Tranche Structure")
    if len(tranches) == 1:
        t = tranches[0]
        st.info(f"Single tranche: {t['name']} ({t['investor']}) "
                f"${t['size']:,.0f} @ {t['rate']*100:.2f}%")
    else:
        t_df = pd.DataFrame([{
            "Tranche": t["name"], "Size": t["size"],
            "Rate": f"{t['rate']*100:.2f}%",
            "Priority": t["priority"], "Investor": t["investor"]
        } for t in tranches])
        t_df["Size"] = t_df["Size"].apply(lambda x: f"${x:,.0f}")
        st.dataframe(t_df, use_container_width=True, hide_index=True)

        sample_m = st.slider("Sample Month", 1, tenure, min(tenure//2, 12))
        avail = sched.loc[sample_m, "Total Payment"]
        rem_cf = avail
        alloc = []
        for t in sorted(tranches, key=lambda x: x["priority"]):
            owed = t["size"] * t["rate_m"]
            princ = min(max(rem_cf - owed, 0), t["size"])
            pay = min(owed + princ, rem_cf)
            alloc.append({"Tranche": t["name"], "Paid": pay})
            rem_cf -= pay
        fig_wf = px.bar(pd.DataFrame(alloc), x="Tranche", y="Paid",
                        color="Tranche", text_auto="$,.0f",
                        title=f"Month {sample_m} Waterfall (${avail:,.0f})")
        fig_wf.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_wf, use_container_width=True)

# ── TAB 5: INVESTORS ─────────────────────────────────────────
with t5:
    st.subheader("💼 Investor Returns")
    for inv_type in ["Fund", "Syndicate", "CBFM"]:
        inv_t = [t for t in tranches if t["investor"] == inv_type]
        if not inv_t:
            continue
        icon = {"Fund": "🏦", "Syndicate": "🤝", "CBFM": "📄"}[inv_type]
        st.markdown(f"### {icon} {inv_type}")
        sz = sum(t["size"] for t in inv_t)
        wr = sum(t["size"]*t["rate"] for t in inv_t) / sz
        c1, c2, c3 = st.columns(3)
        c1.metric("Allocation", f"${sz:,.0f}")
        c2.metric("Wtd Rate", f"{wr*100:.2f}%")
        c3.metric("IRR", f"{investor_irr*100:.2f}%" if not np.isnan(investor_irr) else "—")
        st.caption({"Fund": "Pooled across deals. See Fund tab.",
                     "Syndicate": "Direct deal. IRR pass-through.",
                     "CBFM": "Deal SPV asset. IRR pass-through."}[inv_type])
        st.markdown("---")

# ── TAB 6: RISK ──────────────────────────────────────────────
with t6:
    st.subheader("⚠️ Risk & Sensitivity")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PD", f"{pd_annual*100:.1f}%")
    c2.metric("LGD", f"{lgd*100:.0f}%")
    c3.metric("EL", f"{pd_annual*lgd*100:.2f}%")
    c4.metric("CPR", f"{cpr*100:.1f}%")

    cpr_r = np.arange(0, 25, 5) / 100
    pd_r = np.arange(0, 10, 2) / 100
    mat = np.zeros((len(pd_r), len(cpr_r)))
    for i, p in enumerate(pd_r):
        for j, c in enumerate(cpr_r):
            s = build_schedule(facility, target_irr, tenure,
                               interest_timing, advance_n, capitalise_months,
                               bbsy_curve, margin, reset_freq,
                               dd_events, c, fee_config,
                               construction_end=construction_end,
                               property_value=property_value,
                               property_growth=property_growth)
            cfs = [-facility] + list(s["Total Payment"].iloc[1:])
            g = calc_irr(cfs)
            mat[i, j] = (g - p*lgd)*100 if not np.isnan(g) else 0
    fig_s = px.imshow(mat,
                       x=[f"{c*100:.0f}%" for c in cpr_r],
                       y=[f"{p*100:.0f}%" for p in pd_r],
                       labels=dict(x="CPR", y="PD", color="Net IRR %"),
                       color_continuous_scale="RdYlGn", text_auto=".1f")
    fig_s.update_layout(height=420)
    st.plotly_chart(fig_s, use_container_width=True)

# ── TAB 7: FUND ──────────────────────────────────────────────
with t7:
    st.subheader("🏦 Fund Portfolio")
    if "fund_deals" not in st.session_state:
        st.session_state.fund_deals = []

    with st.expander("➕ Add Deal", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        dn = c1.text_input("Deal Name", "Deal 1")
        ds = c2.number_input("Invested ($)", 0, 100_000_000, 1_000_000, step=100_000)
        di = c3.number_input("IRR (%)", 0.0, 30.0, 10.0, 0.5)
        dst = c4.selectbox("Status", ["Active", "Matured", "Defaulted"])
        if st.button("Add Deal"):
            st.session_state.fund_deals.append(
                {"Name": dn, "Invested": ds, "IRR": di, "Status": dst})

    if st.session_state.fund_deals:
        fdf = pd.DataFrame(st.session_state.fund_deals)
        ti = fdf["Invested"].sum()
        gav = st.number_input("Fund GAV ($)", ti, 1_000_000_000,
                               int(ti * 1.1), step=100_000)
        cash = gav - ti
        wirr = (fdf["Invested"] * fdf["IRR"]).sum() / max(ti, 1)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("GAV", f"${gav:,.0f}")
        c2.metric("Invested", f"${ti:,.0f}")
        c3.metric("Cash", f"${cash:,.0f}")
        c4.metric("Wtd IRR", f"{wirr:.2f}%")
        st.plotly_chart(px.pie(fdf, values="Invested", names="Name",
                                hole=0.4), use_container_width=True)
        st.dataframe(fdf.style.format({"Invested": "${:,.0f}", "IRR": "{:.2f}%"}),
                     use_container_width=True, hide_index=True)
        if st.button("Clear Deals"):
            st.session_state.fund_deals = []
            st.rerun()
    else:
        st.info("Add deals to build your fund portfolio view.")

# ═══════════════════════════════════════════════════════════════
#  TAB 8: TLS (BORROWER POSITION) — INTERNAL USE
# ═══════════════════════════════════════════════════════════════
with t8:
    st.subheader("📋 Transaction Ledger Statement (TLS)")
    st.caption("🔒 Internal use only — Borrower position with all cash movements")

    # Build TLS
    tls_rows = []
    for _, row in sched.iterrows():
        m = int(row["Month"])
        fee_detail = row["Fee Detail"] if isinstance(row["Fee Detail"], dict) else {}

        tls_rows.append({
            "Month": m,
            "Date Marker": f"M{m}" + (" 🎂" if row["Anniversary"] else "")
                           + (" ⚠️DEFAULT" if m == default_month else "")
                           + (" 🔧RESTRUCTURE" if m == restructure_month else ""),
            "Opening Balance": row["Opening"],
            "Drawdown": row["Drawdown"],
            "Interest Accrued": row["Interest"],
            "Interest Paid": row["Interest Paid"],
            "Interest Capitalised": row["Interest Capitalised"],
            "Est. Fee": fee_detail.get("Establishment", 0),
            "Line Fee": fee_detail.get("Line Fee", 0),
            "Admin Fee": fee_detail.get("Admin", 0),
            "Risk Review": fee_detail.get("Risk Review", 0),
            "Restructure Fee": fee_detail.get("Restructure", 0),
            "Discharge Fee": fee_detail.get("Discharge", 0),
            "Total Fees": row["Fees"],
            "Principal Repaid": row["Principal"],
            "Prepayment": row["Prepayment"],
            "Total Outflow": row["Total Payment"] if m > 0 else 0,
            "Closing Balance": row["Closing"],
            "Original Limit": row["Original Limit"],
            "Current Limit": row["Current Limit"],
            "Limit Utilisation %": (row["Closing"] / row["Original Limit"] * 100
                                     if row["Original Limit"] > 0 else 0),
            "Property Value": row["Property Value"],
            "LVR %": row["LVR"],
            "Period Rate %": row["Rate"] * 100
        })

    tls_df = pd.DataFrame(tls_rows)

    # Summary metrics
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Peak Balance",
              f"${tls_df['Closing Balance'].max():,.0f}")
    c2.metric("Final Balance",
              f"${tls_df['Closing Balance'].iloc[-1]:,.0f}")
    c3.metric("Total Interest Paid",
              f"${tls_df['Interest Paid'].sum():,.0f}")
    c4.metric("Total Fees Paid",
              f"${tls_df['Total Fees'].sum():,.0f}")
    c5.metric("Min LVR",
              f"{tls_df['LVR %'].min():.1f}%")
    c6.metric("Max LVR",
              f"{tls_df['LVR %'].max():.1f}%")

    st.markdown("---")

    # TLS Table
    st.markdown("#### Full Ledger")
    display_tls = tls_df.copy()
    pct_cols = ["Limit Utilisation %", "LVR %", "Period Rate %"]
    dollar_cols = [c for c in display_tls.columns
                   if c not in ["Month", "Date Marker"] + pct_cols]
    for c in dollar_cols:
        display_tls[c] = display_tls[c].apply(lambda x: f"${x:,.2f}")
    for c in pct_cols:
        display_tls[c] = display_tls[c].apply(lambda x: f"{x:.2f}%")

    st.dataframe(display_tls, use_container_width=True, height=600)

    # TLS Charts
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        fig_tls1 = go.Figure()
        fig_tls1.add_trace(go.Scatter(
            x=tls_df["Month"], y=tls_df["Closing Balance"]/1e6,
            name="Outstanding ($M)", fill="tozeroy", line=dict(color="red")))
        fig_tls1.add_trace(go.Scatter(
            x=tls_df["Month"], y=tls_df["Current Limit"]/1e6,
            name="Current Limit ($M)", line=dict(dash="dash", color="blue")))
        fig_tls1.add_trace(go.Scatter(
            x=tls_df["Month"], y=tls_df["Original Limit"]/1e6,
            name="Original Limit ($M)", line=dict(dash="dot", color="gray")))
        fig_tls1.update_layout(height=350, yaxis_title="$M",
                               title="Balance vs Limits")
        st.plotly_chart(fig_tls1, use_container_width=True)

    with c2:
        fig_tls2 = go.Figure()
        fig_tls2.add_trace(go.Scatter(
            x=tls_df["Month"], y=tls_df["LVR %"],
            name="LVR (%)", line=dict(color="red", width=2)))
        fig_tls2.add_trace(go.Scatter(
            x=tls_df["Month"], y=tls_df["Limit Utilisation %"],
            name="Limit Utilisation (%)", line=dict(color="blue", width=2)))
        fig_tls2.add_hline(y=80, line_dash="dash", line_color="orange",
                           annotation_text="80%")
        fig_tls2.update_layout(height=350, yaxis_title="%",
                               title="LVR & Utilisation Over Time")
        st.plotly_chart(fig_tls2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
#  TAB 9: SCENARIOS + EXPORT
# ═══════════════════════════════════════════════════════════════
with t9:
    st.subheader("💾 Save / Load Scenarios & Export")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Save Scenario")
        scenario_name = st.text_input("Scenario Name", "Base Case")

        current_params = {
            "name": scenario_name,
            "facility": facility, "target_irr": target_irr * 100,
            "tenure": tenure, "construction_end": construction_end,
            "interest_timing": interest_timing, "advance_n": advance_n,
            "capitalise_months": capitalise_months,
            "use_floating": use_floating, "margin": margin * 100,
            "reset_freq": reset_freq, "n_dd": n_dd,
            "cpr": cpr * 100, "pd_annual": pd_annual * 100, "lgd": lgd * 100,
            "est_pct": est_pct * 100, "line_pct": line_pct * 100,
            "admin_type": admin_type, "admin_amount": admin_amount,
            "admin_pct": admin_pct * 100, "discharge_amt": discharge_amt,
            "risk_review_amt": risk_review_amt,
            "restructure_pct": restructure_pct * 100,
            "property_value": property_value,
            "property_growth": property_growth * 100,
            "n_tranches": n_tranches,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
        }

        scenario_json = json.dumps(current_params, indent=2)
        st.download_button(
            "⬇️ Download Scenario JSON",
            data=scenario_json,
            file_name=f"scenario_{scenario_name.replace(' ', '_')}.json",
            mime="application/json"
        )

    with c2:
        st.markdown("### Load Scenario")
        uploaded = st.file_uploader("Upload Scenario JSON", type=["json"])
        if uploaded is not None:
            loaded = json.loads(uploaded.read())
            st.json(loaded)
            st.info("⬆️ Parameters loaded. Adjust sidebar inputs to match "
                     "the loaded values, then re-run. "
                     "(Streamlit doesn't allow programmatic sidebar updates.)")

    st.markdown("---")
    st.markdown("### 📥 Export to Excel")

    # Build Excel workbook
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Sheet 1: Summary
        summary_data = {
            "Metric": ["Facility", "Target IRR", "Borrower IRR", "Investor IRR",
                        "Total Interest", "Total Fees", "Total Repaid",
                        "Initial LVR", "Tenure", "Interest Timing"],
            "Value": [
                f"${facility:,.0f}", f"{target_irr*100:.2f}%",
                f"{borrower_irr*100:.2f}%" if not np.isnan(borrower_irr) else "N/A",
                f"{investor_irr*100:.2f}%" if not np.isnan(investor_irr) else "N/A",
                f"${total_interest:,.2f}", f"${total_fees:,.2f}",
                f"${sched['Total Payment'].iloc[1:].sum():,.2f}",
                f"{initial_lvr:.1f}%", f"{tenure} months",
                interest_timing
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)

        # Sheet 2: Cash Flow Schedule
        sched_export = sched.drop(columns=["Fee Detail"]).copy()
        sched_export["Rate"] = sched_export["Rate"] * 100
        sched_export.to_excel(writer, sheet_name="Cash Flows", index=False)

        # Sheet 3: TLS
        tls_export = tls_df.copy()
        tls_export.to_excel(writer, sheet_name="TLS", index=False)

        # Sheet 4: Fee Breakdown
        if not fee_df.empty:
            fee_df.to_excel(writer, sheet_name="Fees", index=False)

        # Sheet 5: Tranches
        if tranches:
            t_export = pd.DataFrame([{
                "Tranche": t["name"], "Size": t["size"],
                "Rate": t["rate"], "Investor": t["investor"]
            } for t in tranches])
            t_export.to_excel(writer, sheet_name="Tranches", index=False)

    output.seek(0)
    st.download_button(
        "⬇️ Download Full Excel Report",
        data=output,
        file_name=f"deal_model_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.caption("Excel includes: Summary, Cash Flows, TLS, Fees, Tranches")
