# app.py — FuelEU Maritime Calculator (Simplified, 2025–2050, EUR-only + defaults)
# Implements Extra-EU rule: “BIO first up to the 50% cap; fossil fills the rest”
# Adds top-of-page breakdown: Fossil vs BIO (all & in-scope)
# -------------------------------------------------------------------------------
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
        perc = next(p for s, e, p in REDUCTION_STEPS if s <= y <= e)
        limit = BASELINE_2020_GFI * (1 - perc / 100.0)
        rows.append({"Year": y, "Reduction_%": perc, "Limit_gCO2e_per_MJ": round(limit, 2)})
    return pd.DataFrame(rows)


LIMITS_DF = limits_by_year()

# ──────────────────────────────────────────────────────────────────────────────
# Persistence
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
    """WtW intensity of a mix; energies_MJ are the energies that are IN SCOPE."""
    E_total = sum(energies_MJ.values())
    if E_total <= 0:
        return 0.0
    num = 0.0
    for k, E in energies_MJ.items():
        num += E * max(float(wtw_g_per_MJ.get(k, 0.0)), 0.0)
    return num / E_total


def penalty_eur_per_year(
    g_actual: float,
    g_target: float,
    E_scope_MJ: float,
    penalty_price_eur_per_vlsfo_t: float,
) -> float:
    """
    Penalty = deficit converted to VLSFO-equivalent tons × penalty_price_eur_per_vlsfo_t.
    CB_g = (g_target - g_actual) * E_scope_MJ  (g); if CB_g >= 0 → no penalty.
    VLSFO-eq tons = (-CB_g) / (g_actual [g/MJ] * 41,000 [MJ/t])
    """
    if E_scope_MJ <= 0 or g_actual <= 0 or penalty_price_eur_per_vlsfo_t <= 0:
        return 0.0
    CB_g = (g_target - g_actual) * E_scope_MJ
    if CB_g >= 0:
        return 0.0
    return (-CB_g) / (g_actual * 41_000.0) * penalty_price_eur_per_vlsfo_t


def credit_eur_per_year(
    g_actual: float, g_target: float, E_scope_MJ: float, credit_price_eur_per_vlsfo_t: float
) -> float:
    if E_scope_MJ <= 0 or g_actual <= 0 or credit_price_eur_per_vlsfo_t <= 0:
        return 0.0
    CB_g = (g_target - g_actual) * E_scope_MJ
    if CB_g <= 0:
        return 0.0
    return (CB_g) / (g_actual * 41_000.0) * credit_price_eur_per_vlsfo_t


# ──────────────────────────────────────────────────────────────────────────────
# Formatting helpers (US format, 2 decimals)
# ──────────────────────────────────────────────────────────────────────────────
def us2(x: float) -> str:
    """US-format with thousands commas and 2 decimals."""
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

    st.text_input(label, value=st.session_state[key], key=key, on_change=_normalize, label_visibility="visible")
    return parse_us(st.session_state[key], default=default_val, min_value=min_value)


# ──────────────────────────────────────────────────────────────────────────────
# Scoping logic — implements “BIO first up to the 50% cap; fossil fills the rest”
# ──────────────────────────────────────────────────────────────────────────────
def scoped_energies_for_voyage(energies: Dict[str, float], voyage_type: str) -> Dict[str, float]:
    """
    Returns the energies that are IN SCOPE for FuelEU calculations.
    - Intra-EU / At Berth: 100% counts (return as-is).
    - Extra-EU: Only 50% of total counts. BIO is attributed first up to that 50%;
                the remainder of the scoped half is filled proportionally by HSFO/LFO/MGO.
    """
    E = {k: float(energies.get(k, 0.0)) for k in ["HSFO", "LFO", "MGO", "BIO"]}
    E_tot = sum(E.values())

    if "Extra-EU" in voyage_type:
        E_scope = 0.5 * E_tot
        bio_attr = min(E["BIO"], E_scope)
        remainder = E_scope - bio_attr
        foss_tot = E["HSFO"] + E["LFO"] + E["MGO"]
        if foss_tot > 0 and remainder > 0:
            hsfo_attr = remainder * (E["HSFO"] / foss_tot)
            lfo_attr  = remainder * (E["LFO"]  / foss_tot)
            mgo_attr  = remainder * (E["MGO"]  / foss_tot)
        else:
            hsfo_attr = lfo_attr = mgo_attr = 0.0
        return {"HSFO": hsfo_attr, "LFO": lfo_attr, "MGO": mgo_attr, "BIO": bio_attr}

    # Intra-EU or At Berth: full scope
    return E


# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="FuelEU Maritime Calculator", layout="wide")

st.title("FuelEU Maritime — GHG Intensity & Cost (Simplified)")
st.caption("Period: 2025–2050 • Limits derived from 2020 baseline 91.16 gCO₂e/MJ • WtW basis • Prices in EUR")

# Sidebar — compact inputs high on page
with st.sidebar:
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] div.block-container{
            padding-top: .4rem !important;
            padding-bottom: .4rem !important;
        }
        section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{ gap: .35rem !important; }
        section[data-testid="stSidebar"] [data-testid="column"]{ padding-left:.15rem; padding-right:.15rem; }
        section[data-testid="stSidebar"] label{ font-size: .85rem; margin-bottom: .1rem; }
        section[data-testid="stSidebar"] input[type="text"],
        section[data-testid="stSidebar"] input[type="number"]{
            padding: .2rem .4rem;
            height: 1.6rem;
            min-height: 1.6rem;
        }
        .section-title{ font-weight:600; font-size:.9rem; margin:.25rem 0 .15rem 0; }
        .penalty-label { color: #b91c1c; font-weight: 700; }
        .penalty-note  { color: #b91c1c; font-size: 0.8rem; margin-top:.15rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Voyage scope (adds "At Berth (100%)")
    scope_options = ["Intra-EU (100%)", "Extra-EU (50%)", "At Berth (100%)"]
    saved_scope = _get(DEFAULTS, "voyage_type", scope_options[0])
    try:
        idx = scope_options.index(saved_scope)
    except ValueError:
        idx = 0

    voyage_type = st.radio("Voyage scope", scope_options, index=idx, horizontal=True)

    # Masses
    st.markdown('<div class="section-title">Masses [t]</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="section-title">LCVs [MJ/ton]</div>', unsafe_allow_html=True)
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

    # WtWs
    st.markdown('<div class="section-title">WtW intensities [gCO₂e/MJ]</div>', unsafe_allow_html=True)
    w1, w2 = st.columns(2)
    with w1:
        WtW_HSFO = float_text_input("HSFO WtW", _get(DEFAULTS, "WtW_HSFO", 92.78), key="WtW_HSFO", min_value=0.0)
    with w2:
        WtW_LFO  = float_text_input("LFO WtW" , _get(DEFAULTS, "WtW_LFO" , 92.00), key="WtW_LFO", min_value=0.0)
    w3, w4 = st.columns(2)
    with w3:
        WtW_MGO  = float_text_input("MGO WtW" , _get(DEFAULTS, "WtW_MGO" , 93.93), key="WtW_MGO", min_value=0.0)
    with w4:
        WtW_BIO  = float_text_input("BIO WtW" , _get(DEFAULTS, "WtW_BIO" , 70.00), key="WtW_BIO", min_value=0.0)

    # ── Preview intensity for €/tCO2e ↔ €/VLSFO-eq t conversions (use scoped energies)
    energies_preview_full = {
        "HSFO": compute_energy_MJ(HSFO_t, LCV_HSFO),
        "LFO":  compute_energy_MJ(LFO_t,  LCV_LFO),
        "MGO":  compute_energy_MJ(MGO_t,  LCV_MGO),
        "BIO":  compute_energy_MJ(BIO_t,  LCV_BIO),
    }
    wtw_preview = {"HSFO": WtW_HSFO, "LFO": WtW_LFO, "MGO": WtW_MGO, "BIO": WtW_BIO}
    energies_preview_scoped = scoped_energies_for_voyage(energies_preview_full, voyage_type)
    g_actual_preview = compute_mix_intensity_g_per_MJ(energies_preview_scoped, wtw_preview)
    factor_vlsfo_per_tco2e = (g_actual_preview * 41_000.0) / 1_000_000.0 if g_actual_preview > 0 else 0.0
    st.session_state["factor_vlsfo_per_tco2e"] = factor_vlsfo_per_tco2e  # used by callbacks

    # ── Credits — linked inputs
    st.markdown('<div class="section-title">Compliance Market — Credits</div>', unsafe_allow_html=True)
    if "credit_per_tco2e_str" not in st.session_state:
        st.session_state["credit_per_tco2e_str"] = us2(float(_get(DEFAULTS, "credit_per_tco2e", 200.0)))
    if "credit_per_vlsfo_t_str" not in st.session_state:
        st.session_state["credit_per_vlsfo_t_str"] = us2(float(_get(DEFAULTS, "credit_price_eur_per_vlsfo_t", 0.0)))
    if "credit_sync_guard" not in st.session_state:
        st.session_state["credit_sync_guard"] = False

    def _sync_from_tco2e():
        if st.session_state["credit_sync_guard"]:
            return
        st.session_state["credit_sync_guard"] = True
        per_tco2e = parse_us(st.session_state["credit_per_tco2e_str"], 0.0, 0.0)
        factor = st.session_state.get("factor_vlsfo_per_tco2e", 0.0)
        per_vlsfo = per_tco2e * factor if factor > 0 else 0.0
        st.session_state["credit_per_tco2e_str"] = us2(per_tco2e)
        st.session_state["credit_per_vlsfo_t_str"] = us2(per_vlsfo)
        st.session_state["credit_sync_guard"] = False

    def _sync_from_vlsfo():
        if st.session_state["credit_sync_guard"]:
            return
        st.session_state["credit_sync_guard"] = True
        per_vlsfo = parse_us(st.session_state["credit_per_vlsfo_t_str"], 0.0, 0.0)
        factor = st.session_state.get("factor_vlsfo_per_tco2e", 0.0)
        per_tco2e = (per_vlsfo / factor) if factor > 0 else 0.0
        st.session_state["credit_per_vlsfo_t_str"] = us2(per_vlsfo)
        st.session_state["credit_per_tco2e_str"] = us2(per_tco2e)
        st.session_state["credit_sync_guard"] = False

    c1, c2 = st.columns(2)
    with c1:
        st.text_input("Credit price €/tCO₂e", key="credit_per_tco2e_str", on_change=_sync_from_tco2e)
    with c2:
        st.text_input("Credit price €/VLSFO-eq t", key="credit_per_vlsfo_t_str", on_change=_sync_from_vlsfo)

    credit_per_tco2e = parse_us(st.session_state["credit_per_tco2e_str"], 0.0, 0.0)
    credit_price_eur_per_vlsfo_t = parse_us(st.session_state["credit_per_vlsfo_t_str"], 0.0, 0.0)

    # ── Penalties — linked inputs (red, default 2,400 €/VLSFO-eq t)
    st.markdown('<div class="section-title">Compliance Market — Penalties</div>', unsafe_allow_html=True)
    st.markdown('<div class="penalty-note">Regulated default. Change only if regulation changes.</div>', unsafe_allow_html=True)

    if "penalty_per_vlsfo_t_str" not in st.session_state:
        st.session_state["penalty_per_vlsfo_t_str"] = us2(float(_get(DEFAULTS, "penalty_price_eur_per_vlsfo_t", 2_400.0)))
    if "penalty_per_tco2e_str" not in st.session_state:
        default_pen_vlsfo = parse_us(st.session_state["penalty_per_vlsfo_t_str"], 2_400.0, 0.0)
        default_pen_tco2e = (default_pen_vlsfo / factor_vlsfo_per_tco2e) if factor_vlsfo_per_tco2e > 0 else 0.0
        st.session_state["penalty_per_tco2e_str"] = us2(default_pen_tco2e)
    if "penalty_sync_guard" not in st.session_state:
        st.session_state["penalty_sync_guard"] = False

    def _pen_sync_from_tco2e():
        if st.session_state["penalty_sync_guard"]:
            return
        st.session_state["penalty_sync_guard"] = True
        per_tco2e = parse_us(st.session_state["penalty_per_tco2e_str"], 0.0, 0.0)
        factor = st.session_state.get("factor_vlsfo_per_tco2e", 0.0)
        per_vlsfo = per_tco2e * factor if factor > 0 else 0.0
        st.session_state["penalty_per_tco2e_str"] = us2(per_tco2e)
        st.session_state["penalty_per_vlsfo_t_str"] = us2(per_vlsfo)
        st.session_state["penalty_sync_guard"] = False

    def _pen_sync_from_vlsfo():
        if st.session_state["penalty_sync_guard"]:
            return
        st.session_state["penalty_sync_guard"] = True
        per_vlsfo = parse_us(st.session_state["penalty_per_vlsfo_t_str"], 2_400.0, 0.0)
        factor = st.session_state.get("factor_vlsfo_per_tco2e", 0.0)
        per_tco2e = (per_vlsfo / factor) if factor > 0 else 0.0
        st.session_state["penalty_per_vlsfo_t_str"] = us2(per_vlsfo)
        st.session_state["penalty_per_tco2e_str"] = us2(per_tco2e)
        st.session_state["penalty_sync_guard"] = False

    p1, p2 = st.columns(2)
    with p1:
        st.markdown('<div class="penalty-label">Penalty price €/tCO₂e</div>', unsafe_allow_html=True)
        st.text_input("", key="penalty_per_tco2e_str", on_change=_pen_sync_from_tco2e, placeholder="regulated default")
    with p2:
        st.markdown('<div class="penalty-label">Penalty price €/VLSFO-eq t</div>', unsafe_allow_html=True)
        st.text_input("", key="penalty_per_vlsfo_t_str", on_change=_pen_sync_from_vlsfo, placeholder="regulated default")

    penalty_price_eur_per_vlsfo_t = parse_us(st.session_state["penalty_per_vlsfo_t_str"], 2_400.0, 0.0)
    penalty_price_eur_per_tco2e  = parse_us(st.session_state["penalty_per_tco2e_str"],  0.0, 0.0)

    # Only steppers here
    st.markdown('<div class="section-title">Other</div>', unsafe_allow_html=True)
    consecutive_deficit_years = int(
        st.number_input(
            "Consecutive deficit years (n)",
            min_value=1,
            value=int(_get(DEFAULTS, "consecutive_deficit_years", 1)),
            step=1,
        )
    )

    # Save defaults
    if st.button("💾 Save current inputs as defaults"):
        defaults_to_save = {
            "voyage_type": voyage_type,
            "credit_per_tco2e": credit_per_tco2e,
            "credit_price_eur_per_vlsfo_t": credit_price_eur_per_vlsfo_t,
            "penalty_price_eur_per_vlsfo_t": penalty_price_eur_per_vlsfo_t,
            "consecutive_deficit_years": consecutive_deficit_years,
            "HSFO_t": HSFO_t, "LFO_t": LFO_t, "MGO_t": MGO_t, "BIO_t": BIO_t,
            "LCV_HSFO": LCV_HSFO, "LCV_LFO": LCV_LFO, "LCV_MGO": LCV_MGO, "LCV_BIO": LCV_BIO,
            "WtW_HSFO": WtW_HSFO, "WtW_LFO": WtW_LFO, "WtW_MGO": WtW_MGO, "WtW_BIO": WtW_BIO,
        }
        try:
            with open(DEFAULTS_PATH, "w", encoding="utf-8") as f:
                json.dump(defaults_to_save, f, indent=2)
            st.success("Defaults saved. They will be used next time the app starts.")
        except Exception as e:
            st.error(f"Could not save defaults: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# Derived energies & intensity (using scoped energies for ALL compliance calcs)
# ──────────────────────────────────────────────────────────────────────────────
energies_full = {
    "HSFO": compute_energy_MJ(HSFO_t, LCV_HSFO),
    "LFO":  compute_energy_MJ(LFO_t,  LCV_LFO),
    "MGO":  compute_energy_MJ(MGO_t,  LCV_MGO),
    "BIO":  compute_energy_MJ(BIO_t,  LCV_BIO),
}
wtw = {"HSFO": WtW_HSFO, "LFO": WtW_LFO, "MGO": WtW_MGO, "BIO": WtW_BIO}

scoped_energies = scoped_energies_for_voyage(energies_full, voyage_type)
E_total_MJ = sum(energies_full.values())
E_scope_MJ = sum(scoped_energies.values())
g_actual = compute_mix_intensity_g_per_MJ(scoped_energies, wtw)  # gCO2e/MJ (in-scope mix)

# ── NEW: top-of-page breakdown of fossil vs BIO (all & in-scope)
fossil_all_MJ  = energies_full["HSFO"] + energies_full["LFO"] + energies_full["MGO"]
bio_all_MJ     = energies_full["BIO"]
fossil_scope_MJ = scoped_energies["HSFO"] + scoped_energies["LFO"] + scoped_energies["MGO"]
bio_scope_MJ    = scoped_energies["BIO"]

st.subheader("Energy breakdown (MJ)")
cA, cB, cC, cD, cE, cF = st.columns(6)
with cA: st.metric("Total energy (all)", f"{us2(E_total_MJ)} MJ")
with cB: st.metric("In-scope energy", f"{us2(E_scope_MJ)} MJ")
with cC: st.metric("Fossil — all", f"{us2(fossil_all_MJ)} MJ")
with cD: st.metric("BIO — all", f"{us2(bio_all_MJ)} MJ")
with cE: st.metric("Fossil — in scope", f"{us2(fossil_scope_MJ)} MJ")
with cF: st.metric("BIO — in scope", f"{us2(bio_scope_MJ)} MJ")

# ──────────────────────────────────────────────────────────────────────────────
# Plot — GHG Intensity vs Limit
# ──────────────────────────────────────────────────────────────────────────────
st.header("GHG Intensity vs. FuelEU Limit (2025–2050)")

limit_series = LIMITS_DF["Limit_gCO2e_per_MJ"].tolist()
years = LIMITS_DF["Year"].tolist()
actual_series = [g_actual for _ in years]

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=years,
        y=limit_series,
        name="FuelEU Limit (step)",
        mode="lines",
        line=dict(shape="hv", width=3),
        hovertemplate="Year=%{x}<br>Limit=%{y:,.2f} gCO₂e/MJ<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=years,
        y=actual_series,
        name="Your Mix (WtW, in-scope)",
        mode="lines",
        hovertemplate="Year=%{x}<br>Mix=%{y:,.2f} gCO₂e/MJ<extra></extra>",
    )
)

fig.update_yaxes(tickformat=",.2f")
fig.update_layout(
    xaxis_title="Year",
    yaxis_title="GHG Intensity [gCO₂e/MJ]",
    hovermode="x unified",
    title=(
        f"Total (all): {us2(E_total_MJ)} MJ • In-scope: {us2(E_scope_MJ)} MJ "
        f"• Fossil (all/in-scope): {us2(fossil_all_MJ)} / {us2(fossil_scope_MJ)} MJ "
        f"• BIO (all/in-scope): {us2(bio_all_MJ)} / {us2(bio_scope_MJ)} MJ"
    ),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    margin=dict(l=40, r=20, t=100, b=40),
)
st.plotly_chart(fig, use_container_width=True)
st.caption(f"Energy considered for compliance (in scope): {us2(E_scope_MJ)} MJ.")

# ──────────────────────────────────────────────────────────────────────────────
# Results — single merged table (US format, 2 decimals; Year excluded)
# ──────────────────────────────────────────────────────────────────────────────
st.header("Results (merged per-year table)")

# Per-year emissions (in-scope)
emissions_tco2e = (g_actual * E_scope_MJ) / 1e6  # tCO2e

# Costs/credits per year
penalties_eur, credits_eur, net_eur, cb_t = [], [], [], []
multiplier = 1.0 + (max(int(consecutive_deficit_years), 1) - 1) / 10.0

for _, row in LIMITS_DF.iterrows():
    g_target = float(row["Limit_gCO2e_per_MJ"])
    CB_g = (g_target - g_actual) * E_scope_MJ
    CB_t = CB_g / 1e6  # tCO2e
    cb_t.append(CB_t)

    pen = penalty_eur_per_year(
        g_actual=g_actual,
        g_target=g_target,
        E_scope_MJ=E_scope_MJ,
        penalty_price_eur_per_vlsfo_t=penalty_price_eur_per_vlsfo_t,
    ) * multiplier
    penalties_eur.append(pen)

    cred = credit_eur_per_year(g_actual, g_target, E_scope_MJ, credit_price_eur_per_vlsfo_t)
    credits_eur.append(cred)
    net_eur.append(cred - pen)

df_cost = pd.DataFrame(
    {
        "Year": years,
        "Compliance_Balance_tCO2e": cb_t,
        "Penalty_EUR": penalties_eur,
        "Credit_EUR": credits_eur,
        "Net_EUR": net_eur,
    }
)

df_results = LIMITS_DF[["Year", "Reduction_%", "Limit_gCO2e_per_MJ"]].copy()
df_results["Actual_gCO2e_per_MJ"] = g_actual
df_results["Emissions_tCO2e"] = emissions_tco2e
df_results = df_results.merge(df_cost, on="Year", how="left")

# Display (US format, 2 decimals; Year excluded from formatting)
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
