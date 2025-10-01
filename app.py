# app.py — FuelEU Maritime Cost Calculator (Vessel, Voyages, Pooling)
# -------------------------------------------------------------------
# What this app does
# 1) Single Vessel (Annual): Achieved GHG intensity (WtW) vs year target; scope handling (100%/50%);
#    penalties: GHG-intensity (Annex IV), OPS (from 2030 for container/passenger), optional RFNBO sub-target.
# 2) Per-Voyage (Single Vessel): Voyage-level scope (1.0 intra / 0.5 extra) and OPS by voyage; aggregates to annual result.
# 3) Company Pooling & Borrowing: Combine multiple ships' balances; offset surpluses/deficits; optional borrowing (cap %).
#
# Notes (assumptions kept transparent):
# • RFNBO reward (2025–2033) implemented as denominator multiplier ×2 for RFNBO energy (planning approximation).
# • Scope approximation in "Single Vessel (Annual)" uses intra/extra shares; "Per-Voyage" applies scope per voyage.
# • OPS electricity is included in intensity, excluded from GHG-intensity penalty energy (per Art. 16(4)(d)).
# • Borrowing is implemented as a user-set cap (%) applied to total scoped energy excl. OPS (company level). Confirm your verifier’s interpretation.

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

# Baseline (2020) used for targets
GHG_REF_2020 = 91.16  # gCO2e/MJ

# Reduction anchors vs 2020 (interpolate between for planning)
REDUCTION_ANCHORS = {
    2025: 2.0,
    2030: 6.0,
    2035: 14.5,
    2040: 31.0,
    2045: 62.0,
    2050: 80.0,
}

# GHG-intensity penalty rate (Annex IV equivalence)
EUR_PER_TON_VLSFOE = 2400.0
MJ_PER_TON_VLSFOE = 41000.0
EUR_PER_GJ_NONCOMPL = EUR_PER_TON_VLSFOE / (MJ_PER_TON_VLSFOE / 1000.0)  # ≈ 58.54 €/GJ

# OPS penalty: €1.5 × (kW at berth) × (non-compliant hours, rounded up)
OPS_EUR_PER_KWH = 1.5

YEARS = list(range(2025, 2051))

FUEL_COLUMNS = [
    "Fuel",              # label
    "Mass_t",            # tons
    "LCV_MJ_per_t",      # MJ/t
    "WtW_g_per_MJ",      # g/MJ (Well-to-Wake)
    "Is_RFNBO",          # bool
]

VOYAGE_COLUMNS = [
    "Voyage_ID",
    "Leg_Type",          # "IntraEU" or "ExtraEU"
    "OPS_Electricity_MWh",
    "Elec_WtW_g_per_MJ",
    "OPS_Berth_kW",
    "OPS_NonCompliant_Hours",
]

VOYAGE_FUELS_COLUMNS = [
    "Voyage_ID",
    "Fuel",
    "Mass_t",
    "LCV_MJ_per_t",
    "WtW_g_per_MJ",
    "Is_RFNBO",
]

POOL_COLUMNS = [
    "Vessel",
    "Achieved_g_per_MJ",
    "Scope_Energy_MJ_ExclOPS",
    "Vessel_Type",               # "Other" | "Container" | "Passenger"
    "OPS_Berth_kW",
    "OPS_NonCompliant_Hours",
    "Prev_Consec_Penalty_Years", # int
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
    """Return planning target (g/MJ) via interpolation between anchor reductions."""
    if year <= min(REDUCTION_ANCHORS):
        r = REDUCTION_ANCHORS[min(REDUCTION_ANCHORS)]
    elif year >= max(REDUCTION_ANCHORS):
        r = REDUCTION_ANCHORS[max(REDUCTION_ANCHORS)]
    else:
        anchors = sorted(REDUCTION_ANCHORS.items())
        for (y0, r0), (y1, r1) in zip(anchors[:-1], anchors[1:]):
            if y0 <= year <= y1:
                frac = (year - y0) / (y1 - y0)
                r = r0 + frac * (r1 - r0)
                break
    return (1.0 - r / 100.0) * GHG_REF_2020

def scope_factor(leg_type: str) -> float:
    """Per-voyage scope factor: 1.0 for intra-EU, 0.5 for extra-EU."""
    return 1.0 if str(leg_type).strip().lower().startswith("intra") else 0.5

def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def _fmt_money(x: float) -> str:
    return f"€{x:,.0f}"

# ──────────────────────────────────────────────────────────────────────────────
# Core calculation primitives
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AnnualInputs:
    year: int
    fuels: pd.DataFrame                # FUEL_COLUMNS
    electricity_MWh: float             # OPS electricity used (if any) — included in intensity
    elec_wtw_g_per_MJ: float
    intra_share: float                 # 0..1 energy share
    extra_share: float                 # 0..1 energy share
    rfnbo_reward_on: bool              # reward ×2 on denominator, 2025–2033
    rfnbo_subtarget_on: bool           # from 2034 if sector uptake <1% in 2031
    rfnbo_price_gap_eur_per_t_vlsfoe: float
    vessel_type: str                   # "Other" | "Container" | "Passenger"
    ops_berth_kW: float                # for penalty if non-compliant
    ops_noncompliant_hours: float
    prev_consec_penalty_years: int

@dataclass
class AnnualResults:
    target_g_per_MJ: float
    achieved_g_per_MJ: float
    scope_energy_MJ_incl_ops: float
    scope_energy_MJ_excl_ops: float
    noncompliance_pct: float
    ghg_penalty_eur: float
    ops_penalty_eur: float
    rfnbo_sub_penalty_eur: float
    total_penalty_eur: float

def compute_annual(inp: AnnualInputs) -> AnnualResults:
    df = inp.fuels.copy()
    for c in ["Mass_t", "LCV_MJ_per_t", "WtW_g_per_MJ"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["Is_RFNBO"] = df["Is_RFNBO"].astype(bool)

    df["Energy_MJ"] = df["Mass_t"] * df["LCV_MJ_per_t"]
    total_fuels_MJ = df["Energy_MJ"].sum()

    # Electricity (OPS) into intensity (MJ), with grid WtW
    elec_MJ = float(inp.electricity_MWh or 0.0) * 3600.0
    elec_g = elec_MJ * float(inp.elec_wtw_g_per_MJ or 0.0)

    # RFNBO reward (2025–2033)
    rwd_active = inp.rfnbo_reward_on and (inp.year <= 2033)
    df["Denom_MJ_eff"] = df["Energy_MJ"] * (np.where(df["Is_RFNBO"] & rwd_active, 2.0, 1.0))

    denom_eff_MJ = df["Denom_MJ_eff"].sum() + elec_MJ
    numer_g = (df["Energy_MJ"] * df["WtW_g_per_MJ"]).sum() + elec_g
    total_MJ_incl_ops = total_fuels_MJ + elec_MJ

    # Scope (annual shares)
    w_scope = clamp01(inp.intra_share) * 1.0 + clamp01(inp.extra_share) * 0.5
    scope_denom_eff_MJ = denom_eff_MJ * w_scope
    scope_numer_g = numer_g * w_scope
    scope_energy_MJ_incl_ops = total_MJ_incl_ops * w_scope
    scope_energy_MJ_excl_ops = total_fuels_MJ * w_scope

    achieved = 0.0 if scope_denom_eff_MJ <= 0 else scope_numer_g / scope_denom_eff_MJ
    target = piecewise_linear_target(inp.year)

    noncomp_pct = 0.0 if achieved <= 0 else max(0.0, (achieved - target) / achieved)

    # GHG-intensity penalty (exclude OPS energy)
    noncomp_MJ = scope_energy_MJ_excl_ops * noncomp_pct
    ghg_penalty_base = (noncomp_MJ / 1000.0) * EUR_PER_GJ_NONCOMPL
    factor = 1.0 + max(0, int(inp.prev_consec_penalty_years)) / 10.0 if noncomp_pct > 0 else 1.0
    ghg_penalty_eur = ghg_penalty_base * factor

    # OPS penalty (containers/passengers, from 2030)
    ops_penalty = 0.0
    if inp.year >= 2030 and inp.vessel_type in {"Container", "Passenger"}:
        hours = float(np.ceil(max(0.0, inp.ops_noncompliant_hours)))
        ops_penalty = OPS_EUR_PER_KWH * max(0.0, inp.ops_berth_kW) * hours

    # Optional RFNBO sub-target (2% from 2034) — shortfall × Pd
    rfnbo_sub_penalty = 0.0
    if inp.rfnbo_subtarget_on and inp.year >= 2034:
        rfnbo_MJ = df.loc[df["Is_RFNBO"], "Energy_MJ"].sum() * w_scope
        required_MJ = 0.02 * scope_energy_MJ_excl_ops
        shortfall_MJ = max(0.0, required_MJ - rfnbo_MJ)
        if shortfall_MJ > 0.0 and inp.rfnbo_price_gap_eur_per_t_vlsfoe > 0:
            shortfall_t_vlsfoe = shortfall_MJ / MJ_PER_TON_VLSFOE
            rfnbo_sub_penalty = shortfall_t_vlsfoe * inp.rfnbo_price_gap_eur_per_t_vlsfoe

    total_pen = ghg_penalty_eur + ops_penalty + rfnbo_sub_penalty
    return AnnualResults(
        target_g_per_MJ=target,
        achieved_g_per_MJ=achieved,
        scope_energy_MJ_incl_ops=scope_energy_MJ_incl_ops,
        scope_energy_MJ_excl_ops=scope_energy_MJ_excl_ops,
        noncompliance_pct=noncomp_pct,
        ghg_penalty_eur=ghg_penalty_eur,
        ops_penalty_eur=ops_penalty,
        rfnbo_sub_penalty_eur=rfnbo_sub_penalty,
        total_penalty_eur=total_pen,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Per-Voyage calculations (single vessel)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class VoyageResults:
    achieved_g_per_MJ: float
    target_g_per_MJ: float
    scope_energy_MJ_incl_ops: float
    scope_energy_MJ_excl_ops: float
    noncompliance_pct: float
    ghg_penalty_eur: float
    ops_penalty_eur: float
    total_penalty_eur: float

def compute_per_voyage(
    year: int,
    voyages_df: pd.DataFrame,       # VOYAGE_COLUMNS
    voyage_fuels_df: pd.DataFrame,  # VOYAGE_FUELS_COLUMNS
    rfnbo_reward_on: bool,
    vessel_type_for_ops: str,
    prev_consec_penalty_years: int,
) -> VoyageResults:
    # Normalize / types
    v = voyages_df.copy()
    for c in ["OPS_Electricity_MWh", "Elec_WtW_g_per_MJ", "OPS_Berth_kW", "OPS_NonCompliant_Hours"]:
        v[c] = pd.to_numeric(v[c], errors="coerce").fillna(0.0)
    vf = voyage_fuels_df.copy()
    for c in ["Mass_t", "LCV_MJ_per_t", "WtW_g_per_MJ"]:
        vf[c] = pd.to_numeric(vf[c], errors="coerce").fillna(0.0)
    vf["Is_RFNBO"] = vf["Is_RFNBO"].astype(bool)

    # Merge scope factor per voyage
    v["Scope_Factor"] = v["Leg_Type"].apply(scope_factor).astype(float)

    # OPS electricity (intensity includes it; penalty handled separately)
    v["OPS_MJ"] = v["OPS_Electricity_MWh"] * 3600.0
    v["OPS_g"]  = v["OPS_MJ"] * v["Elec_WtW_g_per_MJ"]

    # Energy & grams from fuels per voyage
    vf["Energy_MJ"] = vf["Mass_t"] * vf["LCV_MJ_per_t"]
    rwd_active = rfnbo_reward_on and (year <= 2033)
    vf["Denom_MJ_eff"] = vf["Energy_MJ"] * (np.where(vf["Is_RFNBO"] & rwd_active, 2.0, 1.0))

    # Aggregate per voyage
    fuel_agg = vf.groupby("Voyage_ID", as_index=False).agg(
        Fuel_Energy_MJ=("Energy_MJ", "sum"),
        Fuel_Numer_g=("Energy_MJ", lambda s: float(np.dot(s, vf.loc[s.index, "WtW_g_per_MJ"]))),
        Fuel_Denom_MJ_eff=("Denom_MJ_eff", "sum"),
        RFNBO_MJ=("Energy_MJ", lambda s: vf.loc[s.index, "Energy_MJ"][vf.loc[s.index, "Is_RFNBO"]].sum()),
    )

    # Join with OPS
    merged = v.merge(fuel_agg, on="Voyage_ID", how="left").fillna(0.0)

    # Totals (pre-scope)
    numer_g_total = (merged["Fuel_Numer_g"] + merged["OPS_g"]).sum()
    denom_eff_MJ_total = (merged["Fuel_Denom_MJ_eff"] + merged["OPS_MJ"]).sum()
    total_energy_MJ_incl_ops = (merged["Fuel_Energy_MJ"] + merged["OPS_MJ"]).sum()
    total_energy_MJ_excl_ops = merged["Fuel_Energy_MJ"].sum()

    # Apply scope per voyage to numerators and denominators
    numer_g_scoped = (merged["Scope_Factor"] * (merged["Fuel_Numer_g"] + merged["OPS_g"])).sum()
    denom_eff_MJ_scoped = (merged["Scope_Factor"] * (merged["Fuel_Denom_MJ_eff"] + merged["OPS_MJ"])).sum()
    energy_scoped_incl_ops_MJ = (merged["Scope_Factor"] * (merged["Fuel_Energy_MJ"] + merged["OPS_MJ"])).sum()
    energy_scoped_excl_ops_MJ = (merged["Scope_Factor"] * merged["Fuel_Energy_MJ"]).sum()

    achieved = 0.0 if denom_eff_MJ_scoped <= 0 else numer_g_scoped / denom_eff_MJ_scoped
    target = piecewise_linear_target(year)
    noncomp_pct = 0.0 if achieved <= 0 else max(0.0, (achieved - target) / achieved)

    # Penalty on scoped energy excl. OPS
    noncomp_MJ = energy_scoped_excl_ops_MJ * noncomp_pct
    ghg_penalty_base = (noncomp_MJ / 1000.0) * EUR_PER_GJ_NONCOMPL
    factor = 1.0 + max(0, int(prev_consec_penalty_years)) / 10.0 if noncomp_pct > 0 else 1.0
    ghg_penalty = ghg_penalty_base * factor

    # OPS penalty (sum non-compliant hours across voyages at covered ports if container/passenger and year>=2030)
    ops_penalty = 0.0
    if year >= 2030 and vessel_type_for_ops in {"Container", "Passenger"}:
        hours_sum = float(np.ceil(max(0.0, merged["OPS_NonCompliant_Hours"].sum())))
        # For berth kW, you may keep per-voyage values; here we conservatively sum kW if multiple parallel calls are unrealistic.
        # Instead, we use the **max** kW across voyages as a simple proxy (adjust per your port ops).
        kW_for_penalty = float(max(0.0, merged["OPS_Berth_kW"].max()))
        ops_penalty = OPS_EUR_PER_KWH * kW_for_penalty * hours_sum

    total_penalty = ghg_penalty + ops_penalty
    return VoyageResults(
        achieved_g_per_MJ=achieved,
        target_g_per_MJ=target,
        scope_energy_MJ_incl_ops=energy_scoped_incl_ops_MJ,
        scope_energy_MJ_excl_ops=energy_scoped_excl_ops_MJ,
        noncompliance_pct=noncomp_pct,
        ghg_penalty_eur=ghg_penalty,
        ops_penalty_eur=ops_penalty,
        total_penalty_eur=total_penalty,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Company pooling / banking / borrowing (multi-ship)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PoolingInputs:
    year: int
    table: pd.DataFrame              # POOL_COLUMNS
    borrow_enabled: bool
    borrow_cap_pct: float            # e.g., 2.0 (% of total scoped energy excl OPS)
    used_borrow_previous_year: bool  # borrowing not allowed in consecutive periods

@dataclass
class PoolingResults:
    target_g_per_MJ: float
    fleet_noncomp_balance_MJ: float      # positive = deficit, negative = surplus
    fleet_noncomp_balance_after_borrow_MJ: float
    ghg_penalty_eur: float
    ops_penalty_eur: float
    total_penalty_eur: float

def compute_pooling(inp: PoolingInputs) -> PoolingResults:
    df = inp.table.copy()
    for c in ["Achieved_g_per_MJ", "Scope_Energy_MJ_ExclOPS", "OPS_Berth_kW", "OPS_NonCompliant_Hours", "Prev_Consec_Penalty_Years"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["Vessel_Type"] = df["Vessel_Type"].astype(str)

    target = piecewise_linear_target(inp.year)

    # Per ship balance in MJ (positive deficit; negative surplus). OPS penalty handled separately.
    def ship_balance(row) -> float:
        I = float(row["Achieved_g_per_MJ"])
        E = float(row["Scope_Energy_MJ_ExclOPS"])
        if I <= 0 or E <= 0:
            return 0.0
        p = max(0.0, (I - target) / I)  # if I <= target → 0, surplus handled by negative? To allow offset we use:
        # For pooling, allow negative contributions by defining:
        p_signed = (I - target) / I
        return p_signed * E  # MJ (can be <0)

    df["Balance_MJ"] = df.apply(ship_balance, axis=1)

    fleet_balance_MJ = float(df["Balance_MJ"].sum())   # can be negative if fleet has surplus
    total_scope_E_MJ = float(df["Scope_Energy_MJ_ExclOPS"].sum())

    # Borrowing (optional, not allowed in consecutive periods)
    borrow_MJ = 0.0
    if inp.borrow_enabled and (not inp.used_borrow_previous_year) and total_scope_E_MJ > 0:
        cap = max(0.0, float(inp.borrow_cap_pct)) / 100.0
        borrow_MJ = cap * total_scope_E_MJ
    balance_after_borrow_MJ = fleet_balance_MJ - borrow_MJ

    # GHG penalty only for positive remaining deficit
    positive_deficit_MJ = max(0.0, balance_after_borrow_MJ)
    # For escalation, apply per-ship escalation on positive ships proportionally (approximation).
    # Simpler (transparent): use no escalator at fleet level; or add a scalar input if needed.
    ghg_penalty = (positive_deficit_MJ / 1000.0) * EUR_PER_GJ_NONCOMPL

    # OPS penalties do NOT pool: sum per ship if applicable (containers/passengers, year>=2030)
    def ship_ops_penalty(row) -> float:
        vt = row["Vessel_Type"]
        if inp.year >= 2030 and vt in {"Container", "Passenger"}:
            hours = float(np.ceil(max(0.0, row["OPS_NonCompliant_Hours"])))
            return OPS_EUR_PER_KWH * max(0.0, row["OPS_Berth_kW"]) * hours
        return 0.0

    ops_penalty = float(df.apply(ship_ops_penalty, axis=1).sum())
    total_penalty = ghg_penalty + ops_penalty

    return PoolingResults(
        target_g_per_MJ=target,
        fleet_noncomp_balance_MJ=fleet_balance_MJ,
        fleet_noncomp_balance_after_borrow_MJ=balance_after_borrow_MJ,
        ghg_penalty_eur=ghg_penalty,
        ops_penalty_eur=ops_penalty,
        total_penalty_eur=total_penalty
    )

# ──────────────────────────────────────────────────────────────────────────────
# UI — Streamlit
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="FuelEU Maritime Cost Calculator", layout="wide")
st.title("FuelEU Maritime Cost Calculator — Vessel • Voyages • Pooling")

states = load_defaults()

tabs = st.tabs(["Single Vessel (Annual)", "Per-Voyage (Single Vessel)", "Company Pooling / Borrowing"])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — Single Vessel (Annual)
# ──────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Annual Inputs — tailored defaults for a dry-bulk profile")
    # Tailored defaults (TMS Dry-style bulker; adjust to your vessels)
    if "fuels_df" not in st.session_state:
        st.session_state["fuels_df"] = pd.DataFrame([
            # Fuel      Mass_t   LCV_MJ/t  WtW_g/MJ  RFNBO?
            ["HFO",     25000.0, 40200.0,  92.784,   False],
            ["LFO",       500.0, 41000.0,  91.251,   False],
            ["MDO/MGO",  1500.0, 42700.0,  93.932,   False],
            ["Biofuel",     0.0, 37000.0,  70.366,   False],
            ["RFNBO",       0.0, 36000.0,  10.000,   True],  # placeholder
        ], columns=FUEL_COLUMNS)

    col1, col2, col3 = st.columns([1,1,1])
    year = col1.selectbox("Year", YEARS, index=YEARS.index(2028))
    vessel_type = col2.selectbox("Vessel type (OPS applies to Container/Passenger from 2030)", ["Other", "Container", "Passenger"], index=0)
    prev_n = col3.number_input("Prev. consecutive penalty years (for escalator)", min_value=0, step=1, value=int(states.get("n_prev", 0)))

    st.caption("Geographical scope (annual energy shares). Intra-EU counted at 100%, Extra-EU at 50%.")
    g1, g2 = st.columns(2)
    intra_share = g1.number_input("Intra-EU share (0..1)", min_value=0.0, max_value=1.0, value=float(states.get("intra_share", 0.6)), step=0.05, format="%.2f")
    extra_share = g2.number_input("Extra-EU share (0..1)", min_value=0.0, max_value=1.0, value=float(states.get("extra_share", 0.4)), step=0.05, format="%.2f")

    st.markdown("**RFNBO & Sub-Target settings**")
    r1, r2, r3 = st.columns([1,1,1])
    rfnbo_reward_on = r1.checkbox("Apply RFNBO reward ×2 (2025–2033)", value=True)
    rfnbo_sub_on = r2.checkbox("RFNBO sub-target enforced? (2% from 2034)", value=False)
    Pd = r3.number_input("Pd for RFNBO sub-target penalty [€/t VLSFOe]", min_value=0.0, value=float(states.get("Pd", 0.0)), step=50.0)

    st.markdown("**OPS (electricity & penalty inputs)**")
    e1, e2, e3 = st.columns(3)
    ops_MWh = e1.number_input("OPS electricity used [MWh/year]", min_value=0.0, value=float(states.get("ops_mwh", 0.0)), step=10.0)
    elec_wtw = e2.number_input("Electricity WtW [g/MJ]", min_value=0.0, value=float(states.get("elec_wtw", 25.0)), step=0.5)
    # For dry bulkers, OPS penalty typically N/A; keep fields for container/passenger cases
    ops_kW = e3.number_input("OPS berth power for penalty [kW]", min_value=0.0, value=float(states.get("ops_kW", 2000.0)), step=100.0)
    e4 = st.number_input("Non-compliant OPS hours at covered ports (year)", min_value=0.0, value=float(states.get("ops_hours", 0.0)), step=1.0)

    st.markdown("**Annual Fuel Table**")
    fuels_df = st.data_editor(st.session_state["fuels_df"], num_rows="dynamic", use_container_width=True)
    st.session_state["fuels_df"] = fuels_df

    if st.button("Save annual defaults"):
        save_defaults({
            "intra_share": intra_share, "extra_share": extra_share,
            "ops_mwh": ops_MWh, "elec_wtw": elec_wtw,
            "ops_kW": ops_kW, "ops_hours": e4,
            "Pd": Pd, "n_prev": prev_n
        })
        st.success("Defaults saved.")

    # Compute
    ainp = AnnualInputs(
        year=year,
        fuels=fuels_df,
        electricity_MWh=ops_MWh,
        elec_wtw_g_per_MJ=elec_wtw,
        intra_share=intra_share,
        extra_share=extra_share,
        rfnbo_reward_on=rfnbo_reward_on,
        rfnbo_subtarget_on=rfnbo_sub_on,
        rfnbo_price_gap_eur_per_t_vlsfoe=Pd,
        vessel_type=vessel_type,
        ops_berth_kW=ops_kW,
        ops_noncompliant_hours=e4,
        prev_consec_penalty_years=prev_n,
    )
    ares = compute_annual(ainp)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Target (g/MJ)", f"{ares.target_g_per_MJ:.2f}")
    k2.metric("Achieved (g/MJ)", f"{ares.achieved_g_per_MJ:.2f}")
    k3.metric("Scoped energy incl. OPS (MJ)", f"{ares.scope_energy_MJ_incl_ops:,.0f}")
    k4.metric("Scoped energy excl. OPS (MJ)", f"{ares.scope_energy_MJ_excl_ops:,.0f}")

    k5, k6, k7 = st.columns(3)
    k5.metric("Non-compliance (%)", f"{100*ares.noncompliance_pct:.2f}%")
    k6.metric("GHG penalty", _fmt_money(ares.ghg_penalty_eur))
    k7.metric("OPS penalty", _fmt_money(ares.ops_penalty_eur))
    st.metric("TOTAL penalty", _fmt_money(ares.total_penalty_eur))

    # Chart: targets for context
    years_plot = list(range(year, min(2051, year + 11)))
    targets_plot = [piecewise_linear_target(y) for y in years_plot]
    achieved_plot = [ares.achieved_g_per_MJ for _ in years_plot]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years_plot, y=targets_plot, name="Year Target", mode="lines+markers", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=years_plot, y=achieved_plot, name="Achieved (this scenario)", mode="lines", line=dict(width=2, dash="dash")))
    fig.update_layout(height=280, margin=dict(l=6, r=6, t=26, b=4), yaxis_title="gCO₂e/MJ", xaxis_title="Year", legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, use_container_width=True)

    # Download
    details = pd.DataFrame({
        "Metric": [
            "Year","Target g/MJ","Achieved g/MJ","Non-compliance %","Scoped MJ incl. OPS","Scoped MJ excl. OPS",
            "GHG penalty €","OPS penalty €","RFNBO sub-target penalty €","TOTAL penalty €"
        ],
        "Value": [
            year, f"{ares.target_g_per_MJ:.3f}", f"{ares.achieved_g_per_MJ:.3f}", f"{100*ares.noncompliance_pct:.3f}%",
            f"{ares.scope_energy_MJ_incl_ops:,.0f}", f"{ares.scope_energy_MJ_excl_ops:,.0f}",
            _fmt_money(ares.ghg_penalty_eur), _fmt_money(ares.ops_penalty_eur), _fmt_money(ares.rfnbo_sub_penalty_eur), _fmt_money(ares.total_penalty_eur)
        ]
    })
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        fuels_df.to_excel(xw, sheet_name="Fuels", index=False)
        details.to_excel(xw, sheet_name="Annual_Results", index=False)
    st.download_button("Download Annual Results (Excel)", data=buf.getvalue(),
                       file_name=f"FuelEU_Annual_{year}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — Per-Voyage (Single Vessel)
# ──────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Per-Voyage Inputs — exact scope per leg (1.0 intra / 0.5 extra)")
    c1, c2, c3 = st.columns([1,1,1])
    year_v = c1.selectbox("Year (Per-Voyage)", YEARS, index=YEARS.index(2028))
    vessel_type_v = c2.selectbox("Vessel type for OPS", ["Other", "Container", "Passenger"], index=0)
    prev_n_v = c3.number_input("Prev. consecutive penalty years", min_value=0, step=1, value=0)

    rwd_on_v = st.checkbox("Apply RFNBO reward ×2 (2025–2033) — per-voyage", value=True)

    st.markdown("**Voyages table** (Leg_Type: IntraEU / ExtraEU)")
    if "voy_df" not in st.session_state:
        st.session_state["voy_df"] = pd.DataFrame([
            ["V001", "IntraEU", 0.0, 25.0, 2000.0, 0.0],
            ["V002", "ExtraEU", 0.0, 25.0, 2000.0, 0.0],
        ], columns=VOYAGE_COLUMNS)
    voy_df = st.data_editor(st.session_state["voy_df"], num_rows="dynamic", use_container_width=True)
    st.session_state["voy_df"] = voy_df

    st.markdown("**Voyage fuel lines** (add rows per fuel per voyage)")
    if "voy_fuels_df" not in st.session_state:
        st.session_state["voy_fuels_df"] = pd.DataFrame([
            ["V001", "HFO",     8000.0, 40200.0, 92.784,  False],
            ["V001", "MDO/MGO",  200.0, 42700.0, 93.932,  False],
            ["V002", "HFO",    12000.0, 40200.0, 92.784,  False],
        ], columns=VOYAGE_FUELS_COLUMNS)
    voy_fuels_df = st.data_editor(st.session_state["voy_fuels_df"], num_rows="dynamic", use_container_width=True)
    st.session_state["voy_fuels_df"] = voy_fuels_df

    vres = compute_per_voyage(
        year=year_v,
        voyages_df=voy_df,
        voyage_fuels_df=voy_fuels_df,
        rfnbo_reward_on=rwd_on_v,
        vessel_type_for_ops=vessel_type_v,
        prev_consec_penalty_years=prev_n_v,
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Target (g/MJ)", f"{vres.target_g_per_MJ:.2f}")
    k2.metric("Achieved (g/MJ)", f"{vres.achieved_g_per_MJ:.2f}")
    k3.metric("Scoped MJ incl. OPS", f"{vres.scope_energy_MJ_incl_ops:,.0f}")
    k4.metric("Scoped MJ excl. OPS", f"{vres.scope_energy_MJ_excl_ops:,.0f}")
    k5, k6 = st.columns(2)
    k5.metric("Non-compliance (%)", f"{100*vres.noncompliance_pct:.2f}%")
    k6.metric("TOTAL penalty", _fmt_money(vres.total_penalty_eur))
    st.caption(f"GHG penalty: {_fmt_money(vres.ghg_penalty_eur)} · OPS penalty: {_fmt_money(vres.ops_penalty_eur)}")

    # Download
    buf2 = io.BytesIO()
    with pd.ExcelWriter(buf2, engine="openpyxl") as xw:
        voy_df.to_excel(xw, sheet_name="Voyages", index=False)
        voy_fuels_df.to_excel(xw, sheet_name="Voyage_Fuels", index=False)
        pd.DataFrame({
            "Metric":["Target g/MJ","Achieved g/MJ","Scoped MJ incl OPS","Scoped MJ excl OPS","Non-compliance %","GHG penalty €","OPS penalty €","TOTAL penalty €"],
            "Value":[f"{vres.target_g_per_MJ:.3f}", f"{vres.achieved_g_per_MJ:.3f}", f"{vres.scope_energy_MJ_incl_ops:,.0f}", f"{vres.scope_energy_MJ_excl_ops:,.0f}", f"{100*vres.noncompliance_pct:.3f}%", _fmt_money(vres.ghg_penalty_eur), _fmt_money(vres.ops_penalty_eur), _fmt_money(vres.total_penalty_eur)]
        }).to_excel(xw, sheet_name="Results", index=False)
    st.download_button("Download Per-Voyage Results (Excel)", data=buf2.getvalue(),
                       file_name=f"FuelEU_PerVoyage_{year_v}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — Company Pooling / Borrowing
# ──────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Company Pooling & Borrowing — combine ships, offset deficits/surpluses")
    c1, c2, c3 = st.columns([1,1,1])
    year_p = c1.selectbox("Year (Pooling)", YEARS, index=YEARS.index(2028))
    borrow_enabled = c2.checkbox("Enable borrowing this year?", value=False)
    borrow_cap_pct = c3.number_input("Borrowing cap (% of total scoped energy excl. OPS)", min_value=0.0, max_value=10.0, value=2.0, step=0.5)

    used_borrow_prev = st.checkbox("Borrowing was used in previous year (then not allowed this year)", value=False)

    st.markdown("**Per-ship table** (enter achieved intensity and scoped energy excl. OPS; OPS penalty is per ship)")
    if "pool_df" not in st.session_state:
        st.session_state["pool_df"] = pd.DataFrame([
            ["MV Alpha", 90.5, 1.2e10, "Other",     0.0,   0.0, 0],
            ["MV Beta",  95.0, 0.9e10, "Container", 2500.0, 12.0, 0],
            ["MV Gamma", 88.0, 0.7e10, "Other",     0.0,   0.0, 0],
        ], columns=POOL_COLUMNS)
    pool_df = st.data_editor(st.session_state["pool_df"], num_rows="dynamic", use_container_width=True)
    st.session_state["pool_df"] = pool_df

    pres = compute_pooling(PoolingInputs(
        year=year_p,
        table=pool_df,
        borrow_enabled=borrow_enabled,
        borrow_cap_pct=borrow_cap_pct,
        used_borrow_previous_year=used_borrow_prev,
    ))

    k1, k2, k3 = st.columns(3)
    k1.metric("Year target (g/MJ)", f"{pres.target_g_per_MJ:.2f}")
    k2.metric("Fleet balance (MJ, +deficit)", f"{pres.fleet_noncomp_balance_MJ:,.0f}")
    k3.metric("After borrowing (MJ)", f"{pres.fleet_noncomp_balance_after_borrow_MJ:,.0f}")
    k4, k5 = st.columns(2)
    k4.metric("GHG penalty (fleet)", _fmt_money(pres.ghg_penalty_eur))
    k5.metric("OPS penalty (sum)", _fmt_money(pres.ops_penalty_eur))
    st.metric("TOTAL company penalty", _fmt_money(pres.total_penalty_eur))

    # Download
    buf3 = io.BytesIO()
    with pd.ExcelWriter(buf3, engine="openpyxl") as xw:
        pool_df.to_excel(xw, sheet_name="Company_Table", index=False)
        pd.DataFrame({
            "Metric":["Year target g/MJ","Fleet balance MJ (+deficit)","After borrowing MJ","GHG penalty €","OPS penalty €","TOTAL penalty €"],
            "Value":[f"{pres.target_g_per_MJ:.3f}", f"{pres.fleet_noncomp_balance_MJ:,.0f}", f"{pres.fleet_noncomp_balance_after_borrow_MJ:,.0f}", _fmt_money(pres.ghg_penalty_eur), _fmt_money(pres.ops_penalty_eur), _fmt_money(pres.total_penalty_eur)]
        }).to_excel(xw, sheet_name="Pooling_Results", index=False)
    st.download_button("Download Pooling Results (Excel)", data=buf3.getvalue(),
                       file_name=f"FuelEU_Pooling_{year_p}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.caption("© 2025 — FuelEU Maritime planning tool. Align with your verifier’s Monitoring Plan and THETIS-MRV data.")
