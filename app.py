# app.py — FuelEU Maritime Calculator (First Cut, 2025–2050, EUR-only + defaults)
# ------------------------------------------------------------------------------
from __future__ import annotations

import json
import os
from typing import Dict, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Constants & Assumptions
# ──────────────────────────────────────────────────────────────────────────────
# Units:
#   Mass: tons (t)
#   LCV : MJ/ton
#   WtW : gCO2eq/MJ
#
# Voyage scope:
#   • Intra‑EU: 100% of energy counts
#   • Extra‑EU: 50% of energy counts
#
# Cost model (EUR‑only):
#   • Premium [€/ton] = price(BIO) − price(selected fuel)
#   • Base price of the selected (replaced) fuel [€/ton] is required
#     to compute energy‑equivalent cost deltas when LCVs differ.
#   • FuelEU penalty per Annex IV simplified formula:
#       Penalty(€) = max(0, −CB) / (GHG_actual * 41,000) * 2,400
#     where CB (gCO2e) = (Target − Actual) * Energy_MJ_in_scope.
#     (41,000 MJ ~ 1 tonne VLSFO equivalent; 2,400 €/t VLSFO‑eq)
#
# References:
#   • GHG intensity baseline (2020): 91.16 gCO2e/MJ (EU FuelEU Maritime)
#   • Reduction steps: 2% (2025‑2029), 6% (2030‑2034), 14.5% (2035‑2039),
#     31% (2040‑2044), 62% (2045‑2049), 80% (2050).

BASELINE_2020_GFI = 91.16  # gCO2e/MJ
DEFAULTS_PATH = ".fueleu_defaults.json"

REDUCTION_STEPS = [
    # (start_year, end_year_inclusive, percent_reduction)
    (2025, 2029, 2.0),
    (2030, 2034, 6.0),
    (2035, 2039, 14.5),
    (2040, 2044, 31.0),
    (2045, 2049, 62.0),
    (2050, 2050, 80.0),
]

YEARS = list(range(2025, 2051))

def limits_by_year() -> pd.DataFrame:
    rows = []
    for y in YEARS:
        perc = next(p for s,e,p in REDUCTION_STEPS if s <= y <= e)
        limit = BASELINE_2020_GFI * (1 - perc/100.0)
        rows.append({"Year": y, "Reduction_%": perc, "Limit_gCO2e_per_MJ": round(limit, 2)})
    return pd.DataFrame(rows)

LIMITS_DF = limits_by_year()

# ──────────────────────────────────────────────────────────────────────────────
# Persistence helpers
# ──────────────────────────────────────────────────────────────────────────────
def _load_defaults() -> Dict[str, Any]:
    if os.path.exists(DEFAULTS_PATH):
        try:
            with open(DEFAULTS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_defaults(d: Dict[str, Any]) -> None:
    try:
        with open(DEFAULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)
    except Exception as e:
        st.error(f"Could not save defaults: {e}")

def _get(d: Dict[str, Any], key: str, fallback):
    return d.get(key, fallback)

DEFAULTS = _load_defaults()

# ──────────────────────────────────────────────────────────────────────────────
# Calculation helpers
# ──────────────────────────────────────────────────────────────────────────────
def compute_energy_MJ(mass_t: float, lcv_MJ_per_t: float) -> float:
    mass_t = max(float(mass_t), 0.0)
    lcv = max(float(lcv_MJ_per_t), 0.0)
    return mass_t * lcv

def compute_mix_intensity_g_per_MJ(energies_MJ: dict, wtw_g_per_MJ: dict) -> float:
    E_total = sum(energies_MJ.values())
    if E_total <= 0:
        return 0.0
    num = 0.0
    for k, E in energies_MJ.items():
        num += E * max(float(wtw_g_per_MJ.get(k, 0.0)), 0.0)
    return num / E_total

def penalty_eur_per_year(g_actual: float, g_target: float, E_scope_MJ: float) -> float:
    """
    Simplified Annex IV formulation:
      Compliance balance CB (g) = (Target - Actual) * E_scope
      Penalty(€) = max(0, -CB) / (g_actual * 41000) * 2400
    """
    if E_scope_MJ <= 0 or g_actual <= 0:
        return 0.0
    CB_g = (g_target - g_actual) * E_scope_MJ
    if CB_g >= 0:
        return 0.0
    return (-CB_g) / (g_actual * 41000.0) * 2400.0

def credit_eur_per_year(g_actual: float, g_target: float, E_scope_MJ: float, credit_price_eur_per_vlsfo_t: float) -> float:
    """
    If Actual < Target, estimate tradable 'benefit' using user‑assumed price:
      Credit(€) = CB_pos / (g_actual * 41000) * credit_price
    """
    if E_scope_MJ <= 0 or g_actual <= 0 or credit_price_eur_per_vlsfo_t <= 0:
        return 0.0
    CB_g = (g_target - g_actual) * E_scope_MJ
    if CB_g <= 0:
        return 0.0
    return (CB_g) / (g_actual * 41000.0) * credit_price_eur_per_vlsfo_t

def premium_cost_delta_eur(mass_bio_t: float,
                           lcv_bio_MJ_per_t: float,
                           lcv_repl_MJ_per_t: float,
                           base_price_repl_eur_per_t: float,
                           premium_eur_per_t: float) -> tuple[float, float]:
    """
    Energy‑equivalent delta vs. using the replaced fuel for the energy that BIO supplies.
      price_bio = base_price_repl + premium
      E_bio = mass_bio * LCV_bio
      tons_repl_equiv = E_bio / LCV_repl   (ensures *energy constant*)
      Δcost = price_bio * mass_bio  −  base_price_repl * tons_repl_equiv   [EUR]
    Returns (delta_eur, tons_repl_equiv).
    """
    mass_bio_t = max(float(mass_bio_t), 0.0)
    E_bio = mass_bio_t * max(float(lcv_bio_MJ_per_t), 0.0)
    if E_bio <= 0 or lcv_repl_MJ_per_t <= 0:
        return 0.0, 0.0
    t_repl_eq = E_bio / float(lcv_repl_MJ_per_t)
    price_bio = max(float(base_price_repl_eur_per_t), 0.0) + max(float(premium_eur_per_t), 0.0)
    delta = price_bio * mass_bio_t - max(float(base_price_repl_eur_per_t), 0.0) * t_repl_eq
    return float(delta), float(t_repl_eq)

# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="FuelEU Maritime Calculator", layout="wide")

st.title("FuelEU Maritime — GHG Intensity & Cost (First Cut)")
st.caption("Period: 2025–2050 • Limits derived from 2020 baseline 91.16 gCO₂e/MJ • WtW basis • Prices in EUR")

# Sidebar — voyage & compliance parameters
st.sidebar.header("Voyage & Compliance")
voyage_type = st.sidebar.radio(
    "Voyage scope",
    ["Intra‑EU (100%)", "Extra‑EU (50%)"],
    index=0 if _get(DEFAULTS, "voyage_type", "Intra‑EU (100%)") == "Intra‑EU (100%)" else 1
)
scope_factor = 1.0 if "Intra" in voyage_type else 0.5

st.sidebar.subheader("Compliance Market Assumptions")
credit_price_eur_per_vlsfo_t = st.sidebar.number_input(
    "Assumed **credit** price (€/VLSFO‑eq t)",
    min_value=0.0,
    value=float(_get(DEFAULTS, "credit_price_eur_per_vlsfo_t", 0.0)),
    step=50.0
)
consecutive_deficit_years = int(st.sidebar.number_input(
    "Consecutive deficit years (n)",
    min_value=1,
    value=int(_get(DEFAULTS, "consecutive_deficit_years", 1)),
    step=1
))

# Sidebar — replacement settings (BIO vs selected fuel)
st.sidebar.header("Replacement Settings (Energy‑Neutral)")
replaced_fuel = st.sidebar.selectbox(
    "Fuel to be **replaced** by BIO",
    ["HSFO", "LFO", "MGO"],
    index=["HSFO","LFO","MGO"].index(_get(DEFAULTS, "replaced_fuel", "HSFO"))
)
premium_eur_per_t = st.sidebar.number_input(
    "Premium (€/ton) = price(BIO) − price(selected fuel)",
    min_value=0.0,
    value=float(_get(DEFAULTS, "premium_eur_per_t", 280.0)),
    step=10.0
)
base_price_repl_eur_per_t = st.sidebar.number_input(
    f"Base price of selected fuel ({replaced_fuel}) [€/ton]",
    min_value=0.0,
    value=float(_get(DEFAULTS, "base_price_repl_eur_per_t", 550.0)),
    step=10.0
)

# Main inputs — fuels
st.header("Fuel Inputs")
c1, c2, c3, c4 = st.columns(4)
with c1:
    HSFO_t = st.number_input("HSFO mass [t]", min_value=0.0, value=float(_get(DEFAULTS, "HSFO_t", 5000.0)), step=100.0, format="%.2f")
    WtW_HSFO = st.number_input("HSFO WtW [gCO₂e/MJ]", min_value=0.0, value=float(_get(DEFAULTS, "WtW_HSFO", 92.78)), step=0.10, format="%.2f")
    LCV_HSFO = st.number_input("HSFO LCV [MJ/ton]", min_value=0.0, value=float(_get(DEFAULTS, "LCV_HSFO", 40200.0)), step=100.0, format="%.2f")
with c2:
    LFO_t = st.number_input("LFO mass [t]", min_value=0.0, value=float(_get(DEFAULTS, "LFO_t", 0.0)), step=50.0, format="%.2f")
    WtW_LFO = st.number_input("LFO WtW [gCO₂e/MJ]", min_value=0.0, value=float(_get(DEFAULTS, "WtW_LFO", 92.00)), step=0.10, format="%.2f")
    LCV_LFO = st.number_input("LFO LCV [MJ/ton]", min_value=0.0, value=float(_get(DEFAULTS, "LCV_LFO", 42700.0)), step=100.0, format="%.2f")
with c3:
    MGO_t = st.number_input("MGO mass [t]", min_value=0.0, value=float(_get(DEFAULTS, "MGO_t", 0.0)), step=50.0, format="%.2f")
    WtW_MGO = st.number_input("MGO WtW [gCO₂e/MJ]", min_value=0.0, value=float(_get(DEFAULTS, "WtW_MGO", 93.93)), step=0.10, format="%.2f")
    LCV_MGO = st.number_input("MGO LCV [MJ/ton]", min_value=0.0, value=float(_get(DEFAULTS, "LCV_MGO", 42700.0)), step=100.0, format="%.2f")
with c4:
    BIO_t = st.number_input("BIO mass [t]", min_value=0.0, value=float(_get(DEFAULTS, "BIO_t", 0.0)), step=50.0, format="%.2f")
    WtW_BIO = st.number_input("BIO WtW [gCO₂e/MJ]", min_value=0.0, value=float(_get(DEFAULTS, "WtW_BIO", 70.0)), step=0.10, format="%.2f")
    LCV_BIO = st.number_input("BIO LCV [MJ/ton]", min_value=0.0, value=float(_get(DEFAULTS, "LCV_BIO", 38000.0)), step=100.0, format="%.2f")

# Save defaults button (bottom of sidebar)
st.sidebar.markdown("---")
if st.sidebar.button("💾 Save current inputs as defaults"):
    defaults_to_save = {
        "voyage_type": voyage_type,
        "credit_price_eur_per_vlsfo_t": credit_price_eur_per_vlsfo_t,
        "consecutive_deficit_years": consecutive_deficit_years,
        "replaced_fuel": replaced_fuel,
        "premium_eur_per_t": premium_eur_per_t,
        "base_price_repl_eur_per_t": base_price_repl_eur_per_t,
        "HSFO_t": HSFO_t, "WtW_HSFO": WtW_HSFO, "LCV_HSFO": LCV_HSFO,
        "LFO_t": LFO_t, "WtW_LFO": WtW_LFO, "LCV_LFO": LCV_LFO,
        "MGO_t": MGO_t, "WtW_MGO": WtW_MGO, "LCV_MGO": LCV_MGO,
        "BIO_t": BIO_t, "WtW_BIO": WtW_BIO, "LCV_BIO": LCV_BIO,
    }
    try:
        with open(DEFAULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(defaults_to_save, f, indent=2)
        st.sidebar.success("Defaults saved. They will be used next time the app starts.")
    except Exception as e:
        st.sidebar.error(f"Could not save defaults: {e}")

# Derived energies and intensity
energies = {
    "HSFO": compute_energy_MJ(HSFO_t, LCV_HSFO),
    "LFO":  compute_energy_MJ(LFO_t,  LCV_LFO),
    "MGO":  compute_energy_MJ(MGO_t,  LCV_MGO),
    "BIO":  compute_energy_MJ(BIO_t,  LCV_BIO),
}
E_total_MJ = sum(energies.values())
E_scope_MJ = E_total_MJ * scope_factor

wtw = {"HSFO": WtW_HSFO, "LFO": WtW_LFO, "MGO": WtW_MGO, "BIO": WtW_BIO}
g_actual = compute_mix_intensity_g_per_MJ(energies, wtw)
tco2eq_total = (g_actual * E_scope_MJ) / 1e6  # tCO2e (considered scope)

# Premium economics (EUR) using energy‑equivalent baseline vs replaced fuel for BIO energy
lcv_map = {"HSFO": LCV_HSFO, "LFO": LCV_LFO, "MGO": LCV_MGO}
prem_delta_eur, t_repl_eq = premium_cost_delta_eur(
    mass_bio_t=BIO_t,
    lcv_bio_MJ_per_t=LCV_BIO,
    lcv_repl_MJ_per_t=max(lcv_map.get(replaced_fuel, 0.0), 0.0),
    base_price_repl_eur_per_t=base_price_repl_eur_per_t,
    premium_eur_per_t=premium_eur_per_t,
)

# ──────────────────────────────────────────────────────────────────────────────
# Premium Economics — TOP SECTION
# ──────────────────────────────────────────────────────────────────────────────
st.header("Premium Economics (energy‑equivalent comparison)")
colA, colB, colC, colD = st.columns(4)
colA.metric("BIO mass [t]", f"{BIO_t:,.2f}")
colB.metric(f"Energy from BIO [MJ]", f"{energies['BIO']:,.0f}")
colC.metric(f"Replaced {replaced_fuel} [t] (energy‑neutral)", f"{t_repl_eq:,.2f}")
colD.metric("Δ Cost vs replaced fuel [€]", f"{prem_delta_eur:,.0f}")

st.info(
    f"Energy neutrality enforced: **t_replaced = (BIO_t × LCV_BIO) / LCV_{replaced_fuel}**. "
    f"Premium = Price(BIO) − Price({replaced_fuel}) [€/t]. "
    f"Base price of {replaced_fuel} is needed to value the **energy‑equivalent tons** the replaced fuel would have supplied "
    f"when LCVs differ."
)

# ──────────────────────────────────────────────────────────────────────────────
# Plot — GHG Intensity vs Limit
# ──────────────────────────────────────────────────────────────────────────────
st.header("GHG Intensity vs. FuelEU Limit (2025–2050)")

# Prepare data
limit_series = LIMITS_DF["Limit_gCO2e_per_MJ"].tolist()
years = LIMITS_DF["Year"].tolist()
actual_series = [g_actual for _ in years]

fig = go.Figure()
# Step limit
fig.add_trace(go.Scatter(
    x=years, y=limit_series, name="FuelEU Limit (step)",
    mode="lines",
    line=dict(shape="hv", width=3),
))
# Actual (mix)
fig.add_trace(go.Scatter(
    x=years, y=actual_series, name="Your Mix (WtW)",
    mode="lines",
))

fig.update_layout(
    xaxis_title="Year",
    yaxis_title="GHG Intensity [gCO₂e/MJ]",
    hovermode="x unified",
    title=f"Total Energy Considered: {E_scope_MJ:,.0f} MJ  •  Voyage: {voyage_type}",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    margin=dict(l=40, r=20, t=80, b=40),
)
st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Tables
# ──────────────────────────────────────────────────────────────────────────────
st.header("Results")

# 1) GHG intensity table (per year)
df_intensity = LIMITS_DF.copy()
df_intensity["Actual_gCO2e_per_MJ"] = round(g_actual, 2)
df_intensity["Delta_vs_Limit"] = round(df_intensity["Actual_gCO2e_per_MJ"] - df_intensity["Limit_gCO2e_per_MJ"], 2)
st.subheader("GHG Intensity of Mix (per year)")
st.dataframe(df_intensity, use_container_width=True)

# 2) Emissions table (tCO2e) — scope‑considered (same each year for constant mix)
df_emis = pd.DataFrame({
    "Year": years,
    "Voyage_Scope": [voyage_type]*len(years),
    "Energy_Considered_MJ": [E_scope_MJ]*len(years),
    "Actual_gCO2e_per_MJ": [round(g_actual, 2)]*len(years),
    "Emissions_tCO2e": [round(tco2eq_total, 3)]*len(years),
})
st.subheader("Emissions (tCO₂e) — Considered Scope")
st.dataframe(df_emis, use_container_width=True)

# 3) FuelEU cost/benefit table
penalties_eur = []
credits_eur = []
net_eur = []
cb_t = []  # compliance balance in tCO2e
multiplier = 1.0 + (max(consecutive_deficit_years, 1) - 1)/10.0

for _, row in LIMITS_DF.iterrows():
    g_target = float(row["Limit_gCO2e_per_MJ"])
    # Compliance balance (g)
    CB_g = (g_target - g_actual) * E_scope_MJ
    CB_t = CB_g / 1e6
    cb_t.append(round(CB_t, 3))
    pen = penalty_eur_per_year(g_actual, g_target, E_scope_MJ) * multiplier
    penalties_eur.append(pen)
    cred = credit_eur_per_year(g_actual, g_target, E_scope_MJ, credit_price_eur_per_vlsfo_t)
    credits_eur.append(cred)
    net_eur.append(cred - pen)

df_cost = pd.DataFrame({
    "Year": years,
    "Compliance_Balance_tCO2e": cb_t,
    "Penalty_EUR": [round(x, 2) for x in penalties_eur],
    "Credit_EUR": [round(x, 2) for x in credits_eur],
    "Net_EUR": [round(x, 2) for x in net_eur],
})
st.subheader("FuelEU: Cost (Penalty) or Benefit (Credit) — per year")
st.dataframe(df_cost, use_container_width=True)

# Optional CSV download
st.download_button(
    "Download per‑year results (CSV)",
    data=pd.concat([df_intensity, df_emis.drop(columns=["Voyage_Scope"]), df_cost.drop(columns=["Compliance_Balance_tCO2e"])], axis=1).to_csv(index=False),
    file_name="fueleu_results_2025_2050.csv",
    mime="text/csv",
)
