# app.py — FuelEU Maritime Calculator (Simplified, 2025–2050, EUR-only + defaults)
# -------------------------------------------------------------------------------
from __future__ import annotations

import json
import os
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Constants & Assumptions
# ──────────────────────────────────────────────────────────────────────────────
BASELINE_2020_GFI = 91.16  # gCO2e/MJ
DEFAULTS_PATH = ".fueleu_defaults.json"

# Reduction steps (stepwise limits)
REDUCTION_STEPS = [
    (2025, 2029, 2.0),
    (2030, 2034, 6.0),
    (2035, 2039, 14.5),
    (2040, 2044, 31.0),
    (2045, 2049, 62.0),
    (2050, 2050, 80.0),
]
YEARS: List[int] = list(range(2025, 2051))

def limits_by_year() -> pd.DataFrame:
    rows = []
    for y in YEARS:
        perc = next(p for s, e, p in REDUCTION_STEPS if s <= y <= e)
        limit = BASELINE_2020_GFI * (1 - perc / 100.0)
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

def _get(d: Dict[str, Any], key: str, fallback):
    return d.get(key, fallback)

DEFAULTS = _load_defaults()

# ──────────────────────────────────────────────────────────────────────────────
# Calculations
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
    Penalty = deficit converted to VLSFO-equivalent tons × 2,400 €/t.
    Deficit in grams: CB_g = (g_target - g_actual) * E_scope_MJ  (negative if above limit)
    VLSFO-eq tons = (-CB_g) / (g_actual [g/MJ] * 41,000 [MJ/t])
    """
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

# ──────────────────────────────────────────────────────────────────────────────
# Formatting helpers (US format with 2 decimals)
# ──────────────────────────────────────────────────────────────────────────────
def us2(x: float) -> str:
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return x

def parse_us(s: str, default: float = 0.0, min_value: float = 0.0) -> float:
    try:
        val = float(str(s).replace(",", ""))
    except Exception:
        val = float(default)
    return max(val, min_value)

def float_text_input(label: str, default_val: float, key: str, min_value: float = 0.0) -> float:
    """Text input that displays and preserves US formatting with 2 decimals (no +/- steppers)."""
    if key not in st.session_state:
        st.session_state[key] = us2(default_val)

    def _normalize():
        val = parse_us(st.session_state[key], default=default_val, min_value=min_value)
        st.session_state[key] = us2(val)

    st.text_input(label, value=st.session_state[key], key=key, on_change=_normalize)
    return parse_us(st.session_state[key], default=default_val, min_value=min_value)

# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="FuelEU Maritime Calculator", layout="wide")

st.title("FuelEU Maritime — GHG Intensity & Cost (Simplified)")
st.caption("Period: 2025–2050 • Limits derived from 2020 baseline 91.16 gCO₂e/MJ • WtW basis • Prices in EUR")

# Sidebar — inputs (two per row, compact). No +/- steppers except for 'Consecutive deficit years'.
with st.sidebar:
    st.header("Inputs")

    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] input[type="text"],
        section[data-testid="stSidebar"] input[type="number"]{
            padding: 0.25rem 0.5rem;
            min-height: 2rem;
            height: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    voyage_type = st.radio(
        "Voyage scope",
        ["Intra-EU (100%)", "Extra-EU (50%)"],
        index=0 if _get(DEFAULTS, "voyage_type", "Intra-EU (100%)") == "Intra-EU (100%)" else 1,
    )
    scope_factor = 1.0 if "Intra" in voyage_type else 0.5

    # Masses
    st.markdown("**Masses [t]**")
    m1, m2 = st.columns(2)
    with m1:
        HSFO_t = float_text_input("HSFO [t]", _get(DEFAULTS, "HSFO_t", 5_000.0), key="HSFO_t", min_value=0.0)
    with m2:
        LFO_t  = float_text_input("LFO [t]" , _get(DEFAULTS, "LFO_t" , 0.0), key="LFO_t", min_value=0.0)
    m3, m4 = st.columns(2)
    with m3:
        MGO_t  = float_text_input("MGO [t]" , _get(DEFAULTS, "MGO_t" , 0.0), key="MGO_t", min_value=0.0)
    with m4:
        BIO_t  = float_text_input("BIO [t]" , _get(DEFAULTS, "BIO_t" , 0.0), key="BIO_t", min_value=0.0)

    # LCVs
    st.markdown("**LCVs [MJ/ton]**")
    l1, l2 = st.columns(2)
    with l1:
        LCV_HSFO = float_text_input("HSFO LCV", _get(DEFAULTS, "LCV_HSFO", 40_200.0), key="LCV_HSFO", min_value=0.0)
    with l2:
        LCV_LFO  = float_text_input("LFO LCV" , _get(DEFAULTS, "LCV_LFO" , 42_700.0), key="LCV_LFO", min_value=0.0)
    l3, l4 = st.columns(2)
    with l3:
        LCV_MGO  = float_text_input("MGO LCV" , _get(DEFAULTS, "LCV_MGO" , 42_700.0), key="LCV_MGO", min_value=0.0)
    with l4:
        LCV_BIO  = float_text_input("BIO LCV" , _get(DEFAULTS, "LCV_BIO" , 38_000.0), key="LCV_BIO", min_value=0.0)

    # WtWs — AR4 and AR5 (you can keep same values; AR5 will be used from 2026 if toggle is ON)
    st.markdown("**WtW intensities [gCO₂e/MJ] — AR4 vs AR5**")
    auto_switch_ar5 = st.checkbox("Apply AR4 in 2025 and AR5 from 2026 onward", value=True)

    w1, w2 = st.columns(2)
    with w1:
        st.markdown("_AR4_")
        WtW_HSFO_AR4 = float_text_input("HSFO WtW (AR4)", _get(DEFAULTS, "WtW_HSFO_AR4", 92.78), key="WtW_HSFO_AR4", min_value=0.0)
        WtW_LFO_AR4  = float_text_input("LFO WtW (AR4)" , _get(DEFAULTS, "WtW_LFO_AR4" , 92.00), key="WtW_LFO_AR4", min_value=0.0)
        WtW_MGO_AR4  = float_text_input("MGO WtW (AR4)" , _get(DEFAULTS, "WtW_MGO_AR4" , 93.93), key="WtW_MGO_AR4", min_value=0.0)
        WtW_BIO_AR4  = float_text_input("BIO WtW (AR4)" , _get(DEFAULTS, "WtW_BIO_AR4" , 70.00), key="WtW_BIO_AR4", min_value=0.0)
    with w2:
        st.markdown("_AR5_")
        WtW_HSFO_AR5 = float_text_input("HSFO WtW (AR5)", _get(DEFAULTS, "WtW_HSFO_AR5", 92.78), key="WtW_HSFO_AR5", min_value=0.0)
        WtW_LFO_AR5  = float_text_input("LFO WtW (AR5)" , _get(DEFAULTS, "WtW_LFO_AR5" , 92.00), key="WtW_LFO_AR5", min_value=0.0)
        WtW_MGO_AR5  = float_text_input("MGO WtW (AR5)" , _get(DEFAULTS, "WtW_MGO_AR5" , 93.93), key="WtW_MGO_AR5", min_value=0.0)
        WtW_BIO_AR5  = float_text_input("BIO WtW (AR5)" , _get(DEFAULTS, "WtW_BIO_AR5" , 70.00), key="WtW_BIO_AR5", min_value=0.0)

    # Compliance Market
    st.markdown("**Compliance Market**")
    c1, c2 = st.columns(2)
    with c1:
        credit_price_eur_per_vlsfo_t = float_text_input(
            "Credit price (€/VLSFO-eq t)",
            _get(DEFAULTS, "credit_price_eur_per_vlsfo_t", 0.0),
            key="credit_price_eur_per_vlsfo_t",
            min_value=0.0,
        )
    with c2:
        # Keep +/- steppers ONLY here
        consecutive_deficit_years = int(
            st.number_input(
                "Consecutive deficit years (n)",
                min_value=1,
                value=int(_get(DEFAULTS, "consecutive_deficit_years", 1)),
                step=1,
            )
        )

    st.markdown("---")
    if st.button("💾 Save current inputs as defaults"):
        defaults_to_save = {
            "voyage_type": voyage_type,
            "credit_price_eur_per_vlsfo_t": credit_price_eur_per_vlsfo_t,
            "consecutive_deficit_years": consecutive_deficit_years,
            "HSFO_t": HSFO_t, "LFO_t": LFO_t, "MGO_t": MGO_t, "BIO_t": BIO_t,
            "LCV_HSFO": LCV_HSFO, "LCV_LFO": LCV_LFO, "LCV_MGO": LCV_MGO, "LCV_BIO": LCV_BIO,
            "WtW_HSFO_AR4": WtW_HSFO_AR4, "WtW_LFO_AR4": WtW_LFO_AR4, "WtW_MGO_AR4": WtW_MGO_AR4, "WtW_BIO_AR4": WtW_BIO_AR4,
            "WtW_HSFO_AR5": WtW_HSFO_AR5, "WtW_LFO_AR5": WtW_LFO_AR5, "WtW_MGO_AR5": WtW_MGO_AR5, "WtW_BIO_AR5": WtW_BIO_AR5,
        }
        try:
            with open(DEFAULTS_PATH, "w", encoding="utf-8") as f:
                json.dump(defaults_to_save, f, indent=2)
            st.success("Defaults saved. They will be used next time the app starts.")
        except Exception as e:
            st.error(f"Could not save defaults: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# Derived energies and scope
# ──────────────────────────────────────────────────────────────────────────────
energies = {
    "HSFO": compute_energy_MJ(HSFO_t, LCV_HSFO),
    "LFO":  compute_energy_MJ(LFO_t,  LCV_LFO),
    "MGO":  compute_energy_MJ(MGO_t,  LCV_MGO),
    "BIO":  compute_energy_MJ(BIO_t,  LCV_BIO),
}
E_total_MJ = sum(energies.values())
E_scope_MJ = E_total_MJ * scope_factor

wtw_ar4 = {"HSFO": WtW_HSFO_AR4, "LFO": WtW_LFO_AR4, "MGO": WtW_MGO_AR4, "BIO": WtW_BIO_AR4}
wtw_ar5 = {"HSFO": WtW_HSFO_AR5, "LFO": WtW_LFO_AR5, "MGO": WtW_MGO_AR5, "BIO": WtW_BIO_AR5}

def wtw_for_year(year: int) -> Dict[str, float]:
    if auto_switch_ar5 and year >= 2026:
        return wtw_ar5
    return wtw_ar4

# Compute per-year actual intensity, emissions, compliance balance, penalty/credit
actual_series = []
emissions_series = []
cb_t_series = []
penalties_eur, credits_eur, net_eur = [], [], []
penalty_rate_eur_per_gj_const = 2400.0 / 41.0  # €/GJ (since 41,000 MJ/t = 41 GJ/t)
penalty_rate_per_tco2e_series = []

multiplier = 1.0 + (max(int(consecutive_deficit_years), 1) - 1) / 10.0

for _, row in LIMITS_DF.iterrows():
    yr = int(row["Year"])
    g_target = float(row["Limit_gCO2e_per_MJ"])
    wtw_year = wtw_for_year(yr)
    g_actual_y = compute_mix_intensity_g_per_MJ(energies, wtw_year)
    actual_series.append(g_actual_y)

    emissions_tco2e_y = (g_actual_y * E_scope_MJ) / 1e6
    emissions_series.append(emissions_tco2e_y)

    CB_g = (g_target - g_actual_y) * E_scope_MJ
    CB_t = CB_g / 1e6
    cb_t_series.append(CB_t)

    # Penalty/credit using per-year g_actual
    pen = penalty_eur_per_year(g_actual_y, g_target, E_scope_MJ) * multiplier
    cre = credit_eur_per_year(g_actual_y, g_target, E_scope_MJ, credit_price_eur_per_vlsfo_t)
    penalties_eur.append(pen)
    credits_eur.append(cre)
    net_eur.append(cre - pen)

    # €/tCO2e rate (per-year, depends on g_actual_y)
    rate_eur_per_tco2e = (2400.0 * 1_000_000.0) / (max(g_actual_y, 1e-12) * 41_000.0)
    penalty_rate_per_tco2e_series.append(rate_eur_per_tco2e)

# ──────────────────────────────────────────────────────────────────────────────
# Plot — GHG Intensity vs Limit (actual now year-dependent)
# ──────────────────────────────────────────────────────────────────────────────
st.header("GHG Intensity vs. FuelEU Limit (2025–2050)")

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=YEARS,
        y=LIMITS_DF["Limit_gCO2e_per_MJ"].tolist(),
        name="FuelEU Limit (step)",
        mode="lines",
        line=dict(shape="hv", width=3),
        hovertemplate="Year=%{x}<br>Limit=%{y:,.2f} gCO₂e/MJ<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=YEARS,
        y=actual_series,
        name="Your Mix (WtW)",
        mode="lines+markers",
        hovertemplate="Year=%{x}<br>Mix=%{y:,.2f} gCO₂e/MJ<extra></extra>",
    )
)

fig.update_yaxes(tickformat=",.2f")
fig.update_layout(
    xaxis_title="Year",
    yaxis_title="GHG Intensity [gCO₂e/MJ]",
    hovermode="x unified",
    title=f"Total Energy (mix): {us2(E_total_MJ)} MJ  •  Voyage: {voyage_type} • AR4→AR5: {'ON' if auto_switch_ar5 else 'OFF'}",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    margin=dict(l=40, r=20, t=80, b=40),
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    f"Energy considered for compliance: {us2(E_scope_MJ)} MJ (scope applied).  "
    f"Penalty reference: 2,400 €/t VLSFO-eq → {us2(penalty_rate_eur_per_gj_const)} €/GJ."
)

# ──────────────────────────────────────────────────────────────────────────────
# Results — single merged table (US format, 2 decimals; Year excluded)
# ──────────────────────────────────────────────────────────────────────────────
st.header("Results (merged per-year table)")

df_results = pd.DataFrame({
    "Year": YEARS,
    "Reduction_%": LIMITS_DF["Reduction_%"].tolist(),
    "Limit_gCO2e_per_MJ": LIMITS_DF["Limit_gCO2e_per_MJ"].tolist(),
    "Actual_gCO2e_per_MJ": actual_series,
    "Emissions_tCO2e": emissions_series,
    "Compliance_Balance_tCO2e": cb_t_series,
    "Penalty_EUR": penalties_eur,
    "Credit_EUR": credits_eur,
    "Net_EUR": net_eur,
    # New informational columns:
    "Penalty_Rate_EUR_per_tCO2e": penalty_rate_per_tco2e_series,  # varies by year (depends on g_actual_y)
    "Penalty_Rate_EUR_per_GJ": [penalty_rate_eur_per_gj_const] * len(YEARS),  # constant
})

# Formatted copy for display/CSV with US format (exclude Year from formatting)
df_fmt = df_results.copy()
for col in df_fmt.columns:
    if col != "Year":
        df_fmt[col] = df_fmt[col].apply(us2)

st.dataframe(df_fmt, use_container_width=True)

st.download_button(
    "Download per-year results (CSV)",
    data=df_fmt.to_csv(index=False),
    file_name="fueleu_results_2025_2050.csv",
    mime="text/csv",
)
