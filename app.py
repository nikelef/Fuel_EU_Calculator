# app.py — FuelEU Cost Calculator (Single Vessel)
# ------------------------------------------------
# A transparent Streamlit tool to estimate a ship’s FuelEU Maritime
# compliance status and cost exposure by year.
#
# What it covers (high level):
# • GHG-intensity (WtW) vs annual target (baseline 91.16 gCO2e/MJ)
# • Penalty for GHG-intensity deficit (Annex IV) using 2,400 €/t VLSFOe (= ~58.54 €/GJ)
# • OPS non-compliance penalty for container/passenger ships (from 2030)
# • Optional RFNBO sub-target penalty (2% from 2034 if sector uptake <1% in 2031)
# • Correct geographical scope treatment: 100% intra-EU, 50% extra-EU
#
# IMPORTANT:
# • Use certified/default WtW factors (Annex II / verifier guidance).
# • This tool is a planning aid; the verifier computes the official balances.

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Constants & configuration
# ──────────────────────────────────────────────────────────────────────────────

DEFAULTS_PATH = ".fueleu_defaults.json"

# Baseline (2020 fleet avg, gCO2e/MJ) — used for targets
GHG_REF_2020 = 91.16  # gCO2e/MJ

# Anchor reduction targets vs 2020 (Reg. (EU) 2023/1805):
# 2025: −2%, 2030: −6%, 2035: −14.5%, 2040: −31%, 2045: −62%, 2050: −80%
REDUCTION_ANCHORS = {
    2025: 2.0,
    2030: 6.0,
    2035: 14.5,
    2040: 31.0,
    2045: 62.0,
    2050: 80.0,
}

# GHG-intensity penalty rate (Annex IV, expressed equivalently):
EUR_PER_TON_VLSFOE = 2400.0
MJ_PER_TON_VLSFOE = 41000.0
EUR_PER_GJ_NONCOMPL = EUR_PER_TON_VLSFOE / (MJ_PER_TON_VLSFOE / 1000.0)  # ≈ 58.54 €/GJ

# OPS penalty (Annex IV): €1.5 × (kW at berth) × (non-compliant hours, rounded up)
OPS_EUR_PER_KWH = 1.5

YEARS = list(range(2025, 2051))
FUEL_COLUMNS = [
    "Fuel",
    "Mass_t",
    "LCV_MJ_per_t",
    "WtW_g_per_MJ",
    "Is_RFNBO",
]

ELECTRICITY_COLUMNS = [
    "OPS_Electricity_MWh",
    "Elec_WtW_g_per_MJ",
]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_defaults() -> Dict:
    try:
        with open(DEFAULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_defaults(dct: Dict) -> None:
    try:
        with open(DEFAULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(dct, f, indent=2)
    except Exception:
        pass

def piecewise_linear_target(year: int) -> float:
    """
    Return the GHG target (gCO2e/MJ) for the given year using linear interpolation
    between the regulation’s anchor reductions. This is suitable for planning;
    the official path is defined by the Regulation.
    """
    if year <= min(REDUCTION_ANCHORS):
        r = REDUCTION_ANCHORS[min(REDUCTION_ANCHORS)]
    elif year >= max(REDUCTION_ANCHORS):
        r = REDUCTION_ANCHORS[max(REDUCTION_ANCHORS)]
    else:
        # interpolate between anchor points
        anchors = sorted(REDUCTION_ANCHORS.items())
        for (y0, r0), (y1, r1) in zip(anchors[:-1], anchors[1:]):
            if y0 <= year <= y1:
                frac = (year - y0) / (y1 - y0)
                r = r0 + frac * (r1 - r0)
                break
    return (1.0 - r / 100.0) * GHG_REF_2020

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def _fmt_money(x: float) -> str:
    return f"€{x:,.0f}"

# ──────────────────────────────────────────────────────────────────────────────
# Core calculations
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ScopeShares:
    intra_eu_share: float  # fraction of yearly energy used on intra-EU/EEA voyages/ports
    extra_eu_share: float  # fraction on voyages to/from third countries
    # remainder is non-scope (third-country to third-country), implicitly 0%

    @property
    def scope_weight(self) -> float:
        """
        Effective scope multiplier: 1.0 for intra-EU share; 0.5 for extra-EU share.
        Used to scale total energy and emissions into the Regulation’s scope, assuming
        uniform fuel mix across legs (planning approximation).
        """
        f_intra = clamp01(self.intra_eu_share)
        f_extra = clamp01(self.extra_eu_share)
        f_out = max(0.0, 1.0 - f_intra - f_extra)
        return f_intra * 1.0 + f_extra * 0.5  # out-of-scope ignored (0%)

@dataclass
class Inputs:
    year: int
    fuels: pd.DataFrame  # columns: FUEL_COLUMNS
    electricity: pd.Series  # index: ELECTRICITY_COLUMNS
    scope: ScopeShares
    apply_rfnbo_reward: bool  # 2x energy credit through 2033
    rfnbo_subtarget_on: bool  # (from 2034 if sector-wide RFNBO<1% in 2031)
    rfnbo_price_gap_eur_per_t_vlsfoe: float  # Pd for RFNBO subtarget penalty
    vessel_type: str        # "Other" | "Container" | "Passenger"
    ops_berth_kw: float     # total electric power demand at berth (kW)
    ops_noncompliant_hours: float  # non-compliant hours in covered ports (rounded up)
    consecutive_prev_penalty_years: int  # n_prev for escalation (0,1,2,...)

@dataclass
class Results:
    target_g_per_MJ: float
    achieved_g_per_MJ: float
    scope_energy_MJ_incl_ops: float
    scope_energy_MJ_excl_ops: float
    noncompliance_pct: float
    ghg_penalty_eur: float
    ops_penalty_eur: float
    rfnbo_sub_penalty_eur: float
    total_penalty_eur: float

def compute_ghg_intensity_and_penalties(inp: Inputs) -> Results:
    df = inp.fuels.copy()
    # Clean & types
    for col in ["Mass_t", "LCV_MJ_per_t", "WtW_g_per_MJ"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["Is_RFNBO"] = df["Is_RFNBO"].astype(bool)

    # Base energy (MJ) per fuel
    df["Energy_MJ"] = df["Mass_t"] * df["LCV_MJ_per_t"]

    # Electricity at berth (OPS used) — MJ
    ops_mwh = float(inp.electricity.get("OPS_Electricity_MWh", 0.0) or 0.0)
    elec_wtw = float(inp.electricity.get("Elec_WtW_g_per_MJ", 0.0) or 0.0)
    elec_MJ = ops_mwh * 3600.0  # 1 MWh = 3,600 MJ

    # Scope weighting (planning approximation: same fuel mix on all legs)
    w_scope = inp.scope.scope_weight

    # RFNBO reward factor (denominator only) through 2033 (incl.)
    rwd_active = inp.apply_rfnbo_reward and (inp.year <= 2033)

    df["Denom_MJ_eff"] = df["Energy_MJ"] * (np.where(df["Is_RFNBO"] & rwd_active, 2.0, 1.0))
    # Scope-adjusted totals
    total_energy_MJ = df["Energy_MJ"].sum()
    total_denom_MJ_eff = df["Denom_MJ_eff"].sum()
    num_grams = (df["Energy_MJ"] * df["WtW_g_per_MJ"]).sum()

    # Add electricity (GHG intensity calc includes all energy used onboard)
    total_energy_MJ += elec_MJ
    total_denom_MJ_eff += elec_MJ  # no reward for grid electricity
    num_grams += elec_MJ * elec_wtw

    # Apply geographical scope to numerator & denominators
    scope_energy_MJ_incl_ops = total_energy_MJ * w_scope
    scope_denom_MJ_eff = total_denom_MJ_eff * w_scope
    scope_num_grams = num_grams * w_scope

    # Achieved GHG intensity (g/MJ); guard zero
    if scope_denom_MJ_eff <= 0:
        achieved_g_per_MJ = 0.0
    else:
        achieved_g_per_MJ = scope_num_grams / scope_denom_MJ_eff

    # Year target
    target_g_per_MJ = piecewise_linear_target(inp.year)

    # Non-compliance percentage = percentage reduction needed to reach the target
    # As per EC Q&A: “percentage by which the actual GHG intensity should have been reduced”:
    # max(0, (Achieved - Target) / Achieved)
    noncompliance_pct = 0.0
    if achieved_g_per_MJ > 0.0:
        noncompliance_pct = max(0.0, (achieved_g_per_MJ - target_g_per_MJ) / achieved_g_per_MJ)

    # Penalty energy excludes OPS electricity (per Art. 16(4)(d))
    scope_energy_MJ_excl_ops = (df["Energy_MJ"].sum()) * w_scope

    # GHG-intensity penalty (Annex IV):
    noncompliant_MJ = scope_energy_MJ_excl_ops * noncompliance_pct
    ghg_penalty_base = (noncompliant_MJ / 1000.0) * EUR_PER_GJ_NONCOMPL  # MJ→GJ
    # Escalation if consecutive deficits in previous years: multiply by (1 + n_prev/10)
    if noncompliance_pct > 0.0 and inp.consecutive_prev_penalty_years > 0:
        factor = 1.0 + (inp.consecutive_prev_penalty_years / 10.0)
    else:
        factor = 1.0
    ghg_penalty_eur = ghg_penalty_base * factor

    # OPS penalty (for container/passenger from 2030 at covered ports):
    ops_penalty_eur = 0.0
    if inp.year >= 2030 and inp.vessel_type in {"Container", "Passenger"}:
        hours = float(np.ceil(max(0.0, inp.ops_noncompliant_hours)))
        ops_penalty_eur = OPS_EUR_PER_KWH * max(0.0, inp.ops_berth_kw) * hours

    # RFNBO sub-target penalty (optional; 2% from 2034 if triggered by sector uptake result)
    rfnbo_sub_penalty_eur = 0.0
    if inp.rfnbo_subtarget_on and inp.year >= 2034:
        rfnbo_energy_MJ = df.loc[df["Is_RFNBO"], "Energy_MJ"].sum() * w_scope  # actual energy (no reward)
        required_MJ = 0.02 * scope_energy_MJ_excl_ops
        shortfall_MJ = max(0.0, required_MJ - rfnbo_energy_MJ)
        if shortfall_MJ > 0.0 and inp.rfnbo_price_gap_eur_per_t_vlsfoe > 0:
            # Annex IV concept: shortfall (VLSFOe) × Pd
            shortfall_t_vlsfoe = shortfall_MJ / MJ_PER_TON_VLSFOE
            rfnbo_sub_penalty_eur = shortfall_t_vlsfoe * inp.rfnbo_price_gap_eur_per_t_vlsfoe

    total_penalty_eur = ghg_penalty_eur + ops_penalty_eur + rfnbo_sub_penalty_eur

    return Results(
        target_g_per_MJ=target_g_per_MJ,
        achieved_g_per_MJ=achieved_g_per_MJ,
        scope_energy_MJ_incl_ops=scope_energy_MJ_incl_ops,
        scope_energy_MJ_excl_ops=scope_energy_MJ_excl_ops,
        noncompliance_pct=noncompliance_pct,
        ghg_penalty_eur=ghg_penalty_eur,
        ops_penalty_eur=ops_penalty_eur,
        rfnbo_sub_penalty_eur=rfnbo_sub_penalty_eur,
        total_penalty_eur=total_penalty_eur,
    )

# ──────────────────────────────────────────────────────────────────────────────
# UI — Streamlit
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="FuelEU Cost Calculator", layout="wide")
st.title("FuelEU Cost Calculator — Single Vessel")

with st.expander("Methodology (concise) & Units", expanded=False):
    st.markdown(
        """
**GHG intensity (achieved)**  \\
\\( I = \\frac{\\sum_i E_i\\,\\cdot\\,WtW_i}{\\sum_i E_i^{\\mathrm{eff}}} \\) in gCO₂e/MJ, where  
• \\(E_i = m_i \\cdot LCV_i\\) (MJ), \\(WtW_i\\) is in g/MJ.  
• For RFNBOs (2025–2033), \\(E_i^{\\mathrm{eff}} = 2\\,E_i\\) (reward factor); else \\(E_i^{\\mathrm{eff}}=E_i\\).  
• Electricity at berth is included in intensity (with its grid WtW).  
• Geographical scope (planning approx.): numerator & denominators scaled by \\(1.0\\times\\text{intra} + 0.5\\times\\text{extra}\\).

**Year target**  \\
Baseline 91.16 g/MJ reduced by: 2025 **2%**, 2030 **6%**, 2035 **14.5%**, 2040 **31%**, 2045 **62%**, 2050 **80%**.  
We linearly interpolate between anchor years for planning.

**Non-compliance percentage**  \\
\\( p = \\max(0,\\, (I - I_{\\text{target}})/I) \\).

**GHG penalty (Annex IV)**  \\
Non-compliant energy = \\(p \\times E_{\\text{scope, excl OPS}}\\). Penalty = **≈ €58.54/GJ** × non-compliant energy (MJ→GJ). Consecutive deficits multiply by \\(1+n_{prev}/10\\).

**OPS penalty** (from 2030 for container/passenger at covered ports)  \\
€1.5 × **kW at berth** × **non-compliant hours** (rounded up).

**RFNBO sub-target penalty** (if applicable from 2034)  \\
If enforced: shortfall vs **2%** of \\(E_{\\text{scope, excl OPS}}\\) × **Pd** (€/t VLSFOe).
        """
    )

states = load_defaults()

# ---- Sidebar inputs
st.sidebar.header("Scenario Inputs")

year = st.sidebar.selectbox("Year", YEARS, index=YEARS.index(2025))
vessel_type = st.sidebar.selectbox("Vessel type (OPS obligation from 2030)", ["Other", "Container", "Passenger"])

st.sidebar.markdown("**Geographical scope (energy shares)**")
col_g1, col_g2 = st.sidebar.columns(2)
intra_share = col_g1.number_input("Intra-EU share (0–1)", min_value=0.0, max_value=1.0,
                                  value=float(states.get("intra_share", 1.0)), step=0.05, format="%.2f")
extra_share = col_g2.number_input("Extra-EU share (0–1)", min_value=0.0, max_value=1.0,
                                  value=float(states.get("extra_share", 0.0)), step=0.05, format="%.2f")

# RFNBO settings
st.sidebar.markdown("---")
apply_rfnbo_reward = st.sidebar.checkbox("Apply RFNBO reward factor (2× energy) (2025–2033)", value=True)
rfnbo_subtarget_on = st.sidebar.checkbox("RFNBO sub-target enforced? (2% from 2034)", value=False)
rfnbo_price_gap = st.sidebar.number_input("Pd for RFNBO sub-target penalty [€/t VLSFOe]", min_value=0.0,
                                          value=float(states.get("rfnbo_price_gap", 0.0)), step=50.0)

# OPS inputs
st.sidebar.markdown("---")
ops_kw = st.sidebar.number_input("OPS berth power, kW (for penalty calc)", min_value=0.0,
                                 value=float(states.get("ops_kw", 1500.0)), step=100.0)
ops_hours = st.sidebar.number_input("Non-compliant hours at covered ports (year)", min_value=0.0,
                                    value=float(states.get("ops_hours", 0.0)), step=1.0)

# Penalty escalation
st.sidebar.markdown("---")
n_prev = st.sidebar.number_input("Consecutive previous penalty years (0,1,2…)", min_value=0, step=1,
                                 value=int(states.get("n_prev", 0)))

# Defaults save button
if st.sidebar.button("Save inputs as defaults", use_container_width=True):
    save_defaults({
        "intra_share": intra_share,
        "extra_share": extra_share,
        "ops_kw": ops_kw,
        "ops_hours": ops_hours,
        "n_prev": n_prev,
        "rfnbo_price_gap": rfnbo_price_gap
    })
    st.sidebar.success("Defaults saved.")

# ---- Fuels table
st.subheader("Fuel Use & Factors (annual)")
st.caption("Provide **Mass [t]**, **LCV [MJ/t]**, **WtW [g/MJ]**, and flag if a row is RFNBO. "
           "Use Annex II defaults or certified values per verifier guidance.")

if "fuels_df" not in st.session_state:
    # Conservative, editable defaults — adjust to your fleet & verifier data.
    st.session_state["fuels_df"] = pd.DataFrame([
        ["HFO",   10000.0, 40200.0, 92.8,  False],
        ["LFO",     500.0, 41000.0, 91.3,  False],
        ["MDO/MGO", 800.0, 42700.0, 93.9,  False],
        ["Biofuel",   0.0, 37000.0, 70.0,  False],
        ["RFNBO",     0.0, 36000.0, 10.0,  True],   # example placeholder
    ], columns=FUEL_COLUMNS)

fuels_df = st.data_editor(
    st.session_state["fuels_df"],
    num_rows="dynamic",
    use_container_width=True
)
# persist edited table
st.session_state["fuels_df"] = fuels_df

# ---- Electricity at berth (OPS) row
st.subheader("Electricity at Berth (OPS) — if used")
elec_defaults = {
    "OPS_Electricity_MWh": 0.0,
    "Elec_WtW_g_per_MJ": 25.0,  # example grid WtW; replace with appropriate national/regional factor
}
elec_state = {k: states.get(k, v) for k, v in elec_defaults.items()}
col_e1, col_e2 = st.columns(2)
ops_mwh = col_e1.number_input("OPS electricity used [MWh/year]", min_value=0.0, value=float(elec_state["OPS_Electricity_MWh"]), step=10.0)
elec_wtw = col_e2.number_input("Electricity WtW [g/MJ]", min_value=0.0, value=float(elec_state["Elec_WtW_g_per_MJ"]), step=0.5)

electricity = pd.Series({"OPS_Electricity_MWh": ops_mwh, "Elec_WtW_g_per_MJ": elec_wtw})

# ---- Run calculation
scope = ScopeShares(intra_eu_share=intra_share, extra_eu_share=extra_share)
inp = Inputs(
    year=year,
    fuels=fuels_df,
    electricity=electricity,
    scope=scope,
    apply_rfnbo_reward=apply_rfnbo_reward,
    rfnbo_subtarget_on=rfnbo_subtarget_on,
    rfnbo_price_gap_eur_per_t_vlsfoe=rfnbo_price_gap,
    vessel_type=vessel_type,
    ops_berth_kw=ops_kw,
    ops_noncompliant_hours=ops_hours,
    consecutive_prev_penalty_years=int(n_prev),
)
res = compute_ghg_intensity_and_penalties(inp)

# ──────────────────────────────────────────────────────────────────────────────
# KPIs & charts
# ──────────────────────────────────────────────────────────────────────────────

k1, k2, k3, k4 = st.columns(4)
k1.metric("GHG Target (g/MJ)", f"{res.target_g_per_MJ:.2f}")
k2.metric("GHG Achieved (g/MJ)", f"{res.achieved_g_per_MJ:.2f}")
k3.metric("Scope energy incl. OPS (MJ)", f"{res.scope_energy_MJ_incl_ops:,.0f}")
k4.metric("Scope energy excl. OPS (MJ)", f"{res.scope_energy_MJ_excl_ops:,.0f}")

k5, k6, k7 = st.columns(3)
k5.metric("Non-compliance (%)", f"{100.0*res.noncompliance_pct:.2f}%")
k6.metric("GHG penalty", _fmt_money(res.ghg_penalty_eur))
k7.metric("OPS penalty", _fmt_money(res.ops_penalty_eur))

k8, k9 = st.columns(2)
k8.metric("RFNBO sub-target penalty", _fmt_money(res.rfnbo_sub_penalty_eur))
k9.metric("TOTAL penalty", _fmt_money(res.total_penalty_eur))

# Step chart: target vs achieved for the next decade for context
years_plot = list(range(year, min(2051, year + 11)))
targets_plot = [piecewise_linear_target(y) for y in years_plot]
ach = [res.achieved_g_per_MJ for _ in years_plot]

fig = go.Figure()
fig.add_trace(go.Scatter(x=years_plot, y=targets_plot, mode="lines+markers",
                         name="Year Target", line=dict(width=2)))
fig.add_trace(go.Scatter(x=years_plot, y=ach, mode="lines",
                         name="Achieved (this scenario)", line=dict(dash="dash", width=2)))
fig.update_layout(
    height=280, margin=dict(l=6, r=6, t=26, b=4),
    yaxis_title="gCO₂e/MJ", xaxis_title="Year",
    legend=dict(orientation="h", y=-0.25)
)
st.plotly_chart(fig, use_container_width=True)

# Detailed table
st.subheader("Detailed Results")
details = pd.DataFrame({
    "Metric": [
        "Year",
        "Scope factor (1×intra + 0.5×extra)",
        "Target (g/MJ)",
        "Achieved (g/MJ)",
        "Non-compliance (%)",
        "Scope energy incl. OPS (MJ)",
        "Scope energy excl. OPS (MJ)",
        "GHG penalty (€)",
        "OPS penalty (€)",
        "RFNBO sub-target penalty (€)",
        "TOTAL penalty (€)",
    ],
    "Value": [
        year,
        f"{scope.scope_weight:.3f}",
        f"{res.target_g_per_MJ:.3f}",
        f"{res.achieved_g_per_MJ:.3f}",
        f"{100.0*res.noncompliance_pct:.3f}%",
        f"{res.scope_energy_MJ_incl_ops:,.0f}",
        f"{res.scope_energy_MJ_excl_ops:,.0f}",
        _fmt_money(res.ghg_penalty_eur),
        _fmt_money(res.ops_penalty_eur),
        _fmt_money(res.rfnbo_sub_penalty_eur),
        _fmt_money(res.total_penalty_eur),
    ]
})
st.dataframe(details, use_container_width=True, height=360)

# Download results
buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as xw:
    fuels_df.to_excel(xw, sheet_name="Fuels_Input", index=False)
    pd.DataFrame(electricity).to_excel(xw, sheet_name="Electricity_OPS", header=False)
    details.to_excel(xw, sheet_name="Results", index=False)
st.download_button(
    "Download Excel",
    data=buf.getvalue(),
    file_name=f"FuelEU_Cost_{year}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.caption("© 2025 — Planning tool for FuelEU Maritime. Always align inputs with your verifier’s guidance and THETIS-MRV records.")
