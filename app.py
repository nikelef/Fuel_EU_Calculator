# app.py — FuelEU Maritime Calculator (Simplified, 2025–2050, EUR-only + defaults)
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
BASELINE_2020_GFI = 91.16  # gCO2e/MJ
DEFAULTS_PATH = ".fueleu_defaults.json"

REDUCTION_STEPS = [
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

# Persistence helpers
def _load_defaults() -> Dict[str, Any]:
    if os.path.exists(DEFAULTS_PATH):
        try:
            with open(DEFAULTS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _get(d: Dict[str, Any], key: str, fallback):
    return d.get(key, fallback)

DEFAULTS = _load_defaults()

# Calculations
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
    if E_scope_MJ <= 0 or g_actual <= 0:
        return 0.0
    CB_g = (g_target - g_actual) * E_scope_MJ
    if CB_g >= 0:
        return 0.0
    return (-CB_g) / (g_actual * 41000.0) * 2400.0

def credit_eur_per_year(g_actual: float, g_target: float, E_scope_MJ: float, credit_price_eur_per_vlsfo_t: float) -> float:
    if E_scope_MJ <= 0 or g_actual <= 0 or credit_price_eur_per_vlsfo_t <= 0:
        return 0.0
    CB_g = (g_target - g_actual) * E_scope_MJ
    if CB_g <= 0:
        return 0.0
    return (CB_g) / (g_actual * 41000.0) * credit_price_eur_per_vlsfo_t

# UI
st.set_page_config(page_title="FuelEU Maritime Calculator", layout="wide")

st.title("FuelEU Maritime — GHG Intensity & Cost (Simplified)")
st.caption("Period: 2025–2050 • Limits derived from 2020 baseline 91.16 gCO₂e/MJ • WtW basis • Prices in EUR")

# Sidebar — all inputs
with st.sidebar:
    st.header("Inputs")
    voyage_type = st.radio(
        "Voyage scope",
        ["Intra‑EU (100%)", "Extra‑EU (50%)"],
        index=0 if _get(DEFAULTS, "voyage_type", "Intra‑EU (100%)") == "Intra‑EU (100%)" else 1
    )
    scope_factor = 1.0 if "Intra" in voyage_type else 0.5

    st.markdown("**Fuel Quantities & Factors**")
    HSFO_t = st.number_input("HSFO mass [t]", min_value=0.0, value=float(_get(DEFAULTS, "HSFO_t", 5000.0)), step=100.0, format="%.2f")
    WtW_HSFO = st.number_input("HSFO WtW [gCO₂e/MJ]", min_value=0.0, value=float(_get(DEFAULTS, "WtW_HSFO", 92.78)), step=0.10, format="%.2f")
    LCV_HSFO = st.number_input("HSFO LCV [MJ/ton]", min_value=0.0, value=float(_get(DEFAULTS, "LCV_HSFO", 40200.0)), step=100.0, format="%.2f")

    LFO_t = st.number_input("LFO mass [t]", min_value=0.0, value=float(_get(DEFAULTS, "LFO_t", 0.0)), step=50.0, format="%.2f")
    WtW_LFO = st.number_input("LFO WtW [gCO₂e/MJ]", min_value=0.0, value=float(_get(DEFAULTS, "WtW_LFO", 92.00)), step=0.10, format="%.2f")
    LCV_LFO = st.number_input("LFO LCV [MJ/ton]", min_value=0.0, value=float(_get(DEFAULTS, "LCV_LFO", 42700.0)), step=100.0, format="%.2f")

    MGO_t = st.number_input("MGO mass [t]", min_value=0.0, value=float(_get(DEFAULTS, "MGO_t", 0.0)), step=50.0, format="%.2f")
    WtW_MGO = st.number_input("MGO WtW [gCO₂e/MJ]", min_value=0.0, value=float(_get(DEFAULTS, "WtW_MGO", 93.93)), step=0.10, format="%.2f")
    LCV_MGO = st.number_input("MGO LCV [MJ/ton]", min_value=0.0, value=float(_get(DEFAULTS, "LCV_MGO", 42700.0)), step=100.0, format="%.2f")

    BIO_t = st.number_input("BIO mass [t]", min_value=0.0, value=float(_get(DEFAULTS, "BIO_t", 0.0)), step=50.0, format="%.2f")
    WtW_BIO = st.number_input("BIO WtW [gCO₂e/MJ]", min_value=0.0, value=float(_get(DEFAULTS, "WtW_BIO", 70.0)), step=0.10, format="%.2f")
    LCV_BIO = st.number_input("BIO LCV [MJ/ton]", min_value=0.0, value=float(_get(DEFAULTS, "LCV_BIO", 38000.0)), step=100.0, format="%.2f")

    st.markdown("**Compliance Market**")
    credit_price_eur_per_vlsfo_t = st.number_input(
        "Credit price (€/VLSFO‑eq t)",
        min_value=0.0,
        value=float(_get(DEFAULTS, "credit_price_eur_per_vlsfo_t", 0.0)),
        step=50.0
    )
    consecutive_deficit_years = int(st.number_input(
        "Consecutive deficit years (n)",
        min_value=1,
        value=int(_get(DEFAULTS, "consecutive_deficit_years", 1)),
        step=1
    ))

    st.markdown("---")
    if st.button("💾 Save current inputs as defaults"):
        defaults_to_save = {
            "voyage_type": voyage_type,
            "credit_price_eur_per_vlsfo_t": credit_price_eur_per_vlsfo_t,
            "consecutive_deficit_years": consecutive_deficit_years,
            "HSFO_t": HSFO_t, "WtW_HSFO": WtW_HSFO, "LCV_HSFO": LCV_HSFO,
            "LFO_t": LFO_t, "WtW_LFO": WtW_LFO, "LCV_LFO": LCV_LFO,
            "MGO_t": MGO_t, "WtW_MGO": WtW_MGO, "LCV_MGO": LCV_MGO,
            "BIO_t": BIO_t, "WtW_BIO": WtW_BIO, "LCV_BIO": LCV_BIO,
        }
        try:
            with open(DEFAULTS_PATH, "w", encoding="utf-8") as f:
                json.dump(defaults_to_save, f, indent=2)
            st.success("Defaults saved. They will be used next time the app starts.")
        except Exception as e:
            st.error(f"Could not save defaults: {e}")

# Derived energies and intensity
energies = {
    "HSFO": compute_energy_MJ(HSFO_t, LCV_HSFO),
    "LFO":  compute_energy_MJ(LFO_t,  LCV_LFO),
    "MGO":  compute_energy_MJ(MGO_t,  LCV_MGO),
    "BIO":  compute_energy_MJ(BIO_t,  LCV_BIO),
}
E_total_MJ = sum(energies.values())
scope_factor = 1.0 if "Intra" in voyage_type else 0.5
E_scope_MJ = E_total_MJ * scope_factor

wtw = {"HSFO": WtW_HSFO, "LFO": WtW_LFO, "MGO": WtW_MGO, "BIO": WtW_BIO}
g_actual = compute_mix_intensity_g_per_MJ(energies, wtw)  # gCO2e/MJ

# Plot — GHG Intensity vs Limit
st.header("GHG Intensity vs. FuelEU Limit (2025–2050)")

limit_series = LIMITS_DF["Limit_gCO2e_per_MJ"].tolist()
years = LIMITS_DF["Year"].tolist()
actual_series = [g_actual for _ in years]

fig = go.Figure()
fig.add_trace(go.Scatter(x=years, y=limit_series, name="FuelEU Limit (step)", mode="lines", line=dict(shape="hv", width=3)))
fig.add_trace(go.Scatter(x=years, y=actual_series, name="Your Mix (WtW)", mode="lines"))

fig.update_layout(
    xaxis_title="Year",
    yaxis_title="GHG Intensity [gCO₂e/MJ]",
    hovermode="x unified",
    title=f"Total Energy (mix): {E_total_MJ:,.0f} MJ  •  Voyage: {voyage_type}",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    margin=dict(l=40, r=20, t=80, b=40),
)
st.plotly_chart(fig, use_container_width=True)
st.caption(f"Energy considered for compliance: {E_scope_MJ:,.0f} MJ (scope applied).")

# Results — intensity, emissions, and cost/credit per year
st.header("Results")

df_intensity = LIMITS_DF.copy()
df_intensity["Actual_gCO2e_per_MJ"] = round(g_actual, 2)
df_intensity["Delta_vs_Limit"] = round(df_intensity["Actual_gCO2e_per_MJ"] - df_intensity["Limit_gCO2e_per_MJ"], 2)
st.subheader("GHG Intensity of Mix (per year)")
st.dataframe(df_intensity, use_container_width=True)

emissions_tco2e = (g_actual * E_scope_MJ) / 1e6  # tCO2e
df_emis = pd.DataFrame({"Year": years, "Emissions_tCO2e": [round(emissions_tco2e, 3)] * len(years)})
st.subheader("Emissions (tCO₂e) — per year (considered scope)")
st.dataframe(df_emis, use_container_width=True)

penalties_eur, credits_eur, net_eur, cb_t = [], [], [], []
multiplier = 1.0 + (max(int(DEFAULTS.get("consecutive_deficit_years", 1)), 1) - 1)/10.0
# Use current sidebar value for multiplier:
multiplier = 1.0 + (max(int(consecutive_deficit_years), 1) - 1)/10.0

for _, row in LIMITS_DF.iterrows():
    g_target = float(row["Limit_gCO2e_per_MJ"])
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
st.subheader("FuelEU: Penalty or Credit (€) — per year")
st.dataframe(df_cost, use_container_width=True)

st.download_button(
    "Download per‑year results (CSV)",
    data=pd.concat([df_intensity, df_emis, df_cost.drop(columns=["Compliance_Balance_tCO2e"])], axis=1).to_csv(index=False),
    file_name="fueleu_results_2025_2050.csv",
    mime="text/csv",
)
