# app.py — FuelEU Maritime Calculator (Simplified, 2025–2050, EUR-only + defaults)
# Scope logic:
#   • Intra-EU: 100% of fuels + 100% of EU OPS electricity (WtW=0)
#   • Extra-EU: EU OPS electricity 100%; fuels split into:
#       – At-berth fuels (EU ports): 100% scope
#       – Voyage fuels: 50% scope with “BIO first; fossil fills remainder proportionally”
# UI/formatting:
#   • Intra-EU shows only “Masses [t] — total (voyage + berth)”
#   • Extra-EU shows “Masses [t] — voyage (excluding at-berth)” + “At-berth masses [t] (EU ports)”
#   • EU OPS electricity input in kWh (converted to MJ)
#   • US-format numbers (1,234.56); no +/- steppers except for “Consecutive deficit years”
#   • Linked credit/penalty price inputs (€/tCO2e ↔ €/VLSFO-eq t), penalty default 2,400
#   • Inputs spaced a bit more; Energy breakdown metrics now BIGGER & BOLD (per request)
#   • Chart: “Your Mix” renamed to “Attained GHG”, dashed line, and inline value labels
# --------------------------------------------------------------------------------------
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

def _get_ops_kwh_default() -> float:
    """Backward-compatible default for OPS electricity: prefer kWh; if only MWh exists, convert."""
    if "OPS_kWh" in DEFAULTS:
        try:
            return float(DEFAULTS["OPS_kWh"])
        except Exception:
            return 0.0
    if "OPS_MWh" in DEFAULTS:
        try:
            return float(DEFAULTS["OPS_MWh"]) * 1_000.0  # MWh → kWh
        except Exception:
            return 0.0
    return 0.0

# ──────────────────────────────────────────────────────────────────────────────
# Calculations
# ──────────────────────────────────────────────────────────────────────────────
def compute_energy_MJ(mass_t: float, lcv_MJ_per_t: float) -> float:
    mass_t = max(float(mass_t), 0.0)
    lcv = max(float(lcv_MJ_per_t), 0.0)
    return mass_t * lcv

def compute_mix_intensity_g_per_MJ(energies_MJ: dict, wtw_g_per_MJ: dict) -> float:
    """WtW intensity of a mix; energies_MJ must be IN-SCOPE energies."""
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
    Penalty = deficit (g) converted to VLSFO-eq tons × penalty_price_eur_per_vlsfo_t.
    CB_g = (g_target − g_actual) × E_scope_MJ; if CB_g ≥ 0 → no penalty.
    VLSFO-eq tons = (−CB_g) / (g_actual [g/MJ] × 41,000 [MJ/t]).
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
    """Text input that displays and preserves US formatting with 2 decimals (no steppers)."""
    if key not in st.session_state:
        st.session_state[key] = us2(default_val)
    def _normalize():
        val = parse_us(st.session_state[key], default=default_val, min_value=min_value)
        st.session_state[key] = us2(val)
    st.text_input(label, value=st.session_state[key], key=key, on_change=_normalize, label_visibility="visible")
    return parse_us(st.session_state[key], default=default_val, min_value=min_value)

# ──────────────────────────────────────────────────────────────────────────────
# Scoping helpers
# ──────────────────────────────────────────────────────────────────────────────
def scoped_energies_extra_eu(energies_fuel_voyage: Dict[str, float],
                             energies_fuel_berth: Dict[str, float],
                             elec_MJ: float) -> Dict[str, float]:
    """
    Extra-EU scope:
      • 100% scope for at-berth fuels (EU ports) + ELEC (OPS)
      • 50% scope for VOYAGE fuels with BIO first; fossil fills remainder proportionally
    """
    scoped = {
        "HSFO": energies_fuel_berth["HSFO"],
        "LFO":  energies_fuel_berth["LFO"],
        "MGO":  energies_fuel_berth["MGO"],
        "BIO":  energies_fuel_berth["BIO"],
        "ELEC": elec_MJ,
    }
    fuel_voy_tot = sum(energies_fuel_voyage.values())
    half_scope   = 0.5 * fuel_voy_tot

    bio_attr = min(energies_fuel_voyage["BIO"], half_scope)
    remainder = half_scope - bio_attr

    foss_voy_tot = energies_fuel_voyage["HSFO"] + energies_fuel_voyage["LFO"] + energies_fuel_voyage["MGO"]
    if foss_voy_tot > 0 and remainder > 0:
        hsfo_attr = remainder * (energies_fuel_voyage["HSFO"] / foss_voy_tot)
        lfo_attr  = remainder * (energies_fuel_voyage["LFO"]  / foss_voy_tot)
        mgo_attr  = remainder * (energies_fuel_voyage["MGO"]  / foss_voy_tot)
    else:
        hsfo_attr = lfo_attr = mgo_attr = 0.0

    scoped["HSFO"] += hsfo_attr
    scoped["LFO"]  += lfo_attr
    scoped["MGO"]  += mgo_attr
    scoped["BIO"]  += bio_attr
    return scoped

# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="FuelEU Maritime Calculator", layout="wide")

st.title("FuelEU Maritime — GHG Intensity & Cost (Simplified)")
st.caption("Period: 2025–2050 • Limits derived from 2020 baseline 91.16 gCO₂e/MJ • WtW basis • Prices in EUR")

# Global CSS:
#  • Inputs: spaced a bit more (but still compact)
#  • Metrics: BIGGER & BOLD (per latest request)
#  • Muted note style (at-berth explanation)
st.markdown(
    """
    <style>
    /* Metric styling — bigger & bold */
    [data-testid="stMetricLabel"] { font-size: 1.05rem !important; font-weight: 700 !important; color: #111827 !important; }
    [data-testid="stMetricValue"] { font-size: 1.35rem !important; font-weight: 800 !important; line-height: 1.1 !important; }
    [data-testid="stMetric"] { padding: .35rem .5rem !important; }

    /* Sidebar input spacing */
    section[data-testid="stSidebar"] div.block-container{
        padding-top: .6rem !important; padding-bottom: .6rem !important;
    }
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{ gap: .6rem !important; }
    section[data-testid="stSidebar"] [data-testid="column"]{ padding-left:.25rem; padding-right:.25rem; }
    section[data-testid="stSidebar"] label{ font-size: .95rem; margin-bottom: .2rem; font-weight: 600; }
    section[data-testid="stSidebar"] input[type="text"],
    section[data-testid="stSidebar"] input[type="number"]{
        padding: .32rem .55rem; height: 2.0rem; min-height: 2.0rem;
    }
    .section-title{ font-weight:700; font-size:1.0rem; margin:.35rem 0 .25rem 0; }
    .muted-note{ font-size:.86rem; color:#6b7280; margin:-.05rem 0 .4rem 0; }
    .penalty-label { color: #b91c1c; font-weight: 800; }
    .penalty-note  { color: #b91c1c; font-size: 0.9rem; margin-top:.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar — structured groups with conditional rendering
with st.sidebar:
    scope_options = ["Intra-EU (100%)", "Extra-EU (50%)"]
    saved_scope = _get(DEFAULTS, "voyage_type", scope_options[0])
    try:
        idx = scope_options.index(saved_scope)
    except ValueError:
        idx = 0
    voyage_type = st.radio("Voyage scope", scope_options, index=idx, horizontal=True)
    st.divider()

    # ── Masses UI depends on scope
    if "Extra-EU" in voyage_type:
        # Voyage masses (excluding at-berth) — subject to 50%
        st.markdown('<div class="section-title">Masses [t] — voyage (excluding at-berth)</div>', unsafe_allow_html=True)
        vm1, vm2 = st.columns(2)
        with vm1:
            HSFO_voy_t = float_text_input("HSFO voyage [t]", _get(DEFAULTS, "HSFO_voy_t", _get(DEFAULTS, "HSFO_t", 5_000.0)),
                                          key="HSFO_voy_t", min_value=0.0)
        with vm2:
            LFO_voy_t  = float_text_input("LFO voyage [t]" , _get(DEFAULTS, "LFO_voy_t" , _get(DEFAULTS, "LFO_t" , 0.0)),
                                          key="LFO_voy_t", min_value=0.0)
        vm3, vm4 = st.columns(2)
        with vm3:
            MGO_voy_t  = float_text_input("MGO voyage [t]" , _get(DEFAULTS, "MGO_voy_t" , _get(DEFAULTS, "MGO_t" , 0.0)),
                                          key="MGO_voy_t", min_value=0.0)
        with vm4:
            BIO_voy_t  = float_text_input("BIO voyage [t]" , _get(DEFAULTS, "BIO_voy_t" , _get(DEFAULTS, "BIO_t" , 0.0)),
                                          key="BIO_voy_t", min_value=0.0)
        st.divider()

        # At-berth masses (EU ports) — 100% scope in Extra-EU
        st.markdown('<div class="section-title">At-berth masses [t] (EU ports)</div>', unsafe_allow_html=True)
        st.markdown('<div class="muted-note">Always 100% in scope; particularly relevant for Extra-EU voyages.</div>',
                    unsafe_allow_html=True)
        bm1, bm2 = st.columns(2)
        with bm1:
            HSFO_berth_t = float_text_input("HSFO at berth [t]", _get(DEFAULTS, "HSFO_berth_t", 0.0),
                                            key="HSFO_berth_t", min_value=0.0)
        with bm2:
            LFO_berth_t  = float_text_input("LFO at berth [t]" , _get(DEFAULTS, "LFO_berth_t" , 0.0),
                                            key="LFO_berth_t", min_value=0.0)
        bm3, bm4 = st.columns(2)
        with bm3:
            MGO_berth_t  = float_text_input("MGO at berth [t]" , _get(DEFAULTS, "MGO_berth_t" , 0.0),
                                            key="MGO_berth_t", min_value=0.0)
        with bm4:
            BIO_berth_t  = float_text_input("BIO at berth [t]" , _get(DEFAULTS, "BIO_berth_t" , 0.0),
                                            key="BIO_berth_t", min_value=0.0)

    else:
        # Intra-EU: a single combined block (voyage + berth)
        st.markdown('<div class="section-title">Masses [t] — total (voyage + berth)</div>', unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        with m1:
            HSFO_total_t = float_text_input("HSFO [t]", _get(DEFAULTS, "HSFO_t", 5_000.0),
                                            key="HSFO_total_t", min_value=0.0)
        with m2:
            LFO_total_t  = float_text_input("LFO [t]" , _get(DEFAULTS, "LFO_t" , 0.0),
                                            key="LFO_total_t", min_value=0.0)
        m3, m4 = st.columns(2)
        with m3:
            MGO_total_t  = float_text_input("MGO [t]" , _get(DEFAULTS, "MGO_t" , 0.0),
                                            key="MGO_total_t", min_value=0.0)
        with m4:
            BIO_total_t  = float_text_input("BIO [t]" , _get(DEFAULTS, "BIO_t" , 0.0),
                                            key="BIO_total_t", min_value=0.0)

        # For calculations, set voyage = total and berth = 0 in Intra-EU
        HSFO_voy_t, LFO_voy_t, MGO_voy_t, BIO_voy_t = HSFO_total_t, LFO_total_t, MGO_total_t, BIO_total_t
        HSFO_berth_t = LFO_berth_t = MGO_berth_t = BIO_berth_t = 0.0

    st.divider()

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
    st.divider()

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
    st.divider()

    # EU OPS electricity — input in kWh; converted to MJ (1 kWh = 3.6 MJ); 100% scope; WtW = 0 g/MJ
    st.markdown('<div class="section-title">EU OPS electricity</div>', unsafe_allow_html=True)
    OPS_kWh = float_text_input("Electricity delivered (kWh)", _get_ops_kwh_default(), key="OPS_kWh", min_value=0.0)
    OPS_MJ  = OPS_kWh * 3.6  # kWh → MJ
    st.divider()

    # Preview factor for €/tCO2e ↔ €/VLSFO-eq t (based on IN-SCOPE mix incl. ELEC and at-berth fuels)
    if "Extra-EU" in voyage_type:
        energies_preview_fuel_voyage = {
            "HSFO": compute_energy_MJ(HSFO_voy_t, LCV_HSFO),
            "LFO":  compute_energy_MJ(LFO_voy_t,  LCV_LFO),
            "MGO":  compute_energy_MJ(MGO_voy_t,  LCV_MGO),
            "BIO":  compute_energy_MJ(BIO_voy_t,  LCV_BIO),
        }
        energies_preview_fuel_berth = {
            "HSFO": compute_energy_MJ(HSFO_berth_t, LCV_HSFO),
            "LFO":  compute_energy_MJ(LFO_berth_t,  LCV_LFO),
            "MGO":  compute_energy_MJ(MGO_berth_t,  LCV_MGO),
            "BIO":  compute_energy_MJ(BIO_berth_t,  LCV_BIO),
        }
        wtw_preview = {"HSFO": WtW_HSFO, "LFO": WtW_LFO, "MGO": WtW_MGO, "BIO": WtW_BIO, "ELEC": 0.0}
        energies_preview_scoped = scoped_energies_extra_eu(energies_preview_fuel_voyage,
                                                           energies_preview_fuel_berth,
                                                           OPS_MJ)
    else:
        energies_preview_fuel_full = {
            "HSFO": compute_energy_MJ(HSFO_voy_t, LCV_HSFO),  # voy_t == total_t
            "LFO":  compute_energy_MJ(LFO_voy_t,  LCV_LFO),
            "MGO":  compute_energy_MJ(MGO_voy_t,  LCV_MGO),
            "BIO":  compute_energy_MJ(BIO_voy_t,  LCV_BIO),
            "ELEC": OPS_MJ,
        }
        wtw_preview = {"HSFO": WtW_HSFO, "LFO": WtW_LFO, "MGO": WtW_MGO, "BIO": WtW_BIO, "ELEC": 0.0}
        energies_preview_scoped = energies_preview_fuel_full

    g_actual_preview = compute_mix_intensity_g_per_MJ(energies_preview_scoped, wtw_preview)
    factor_vlsfo_per_tco2e = (g_actual_preview * 41_000.0) / 1_000_000.0 if g_actual_preview > 0 else 0.0
    st.session_state["factor_vlsfo_per_tco2e"] = factor_vlsfo_per_tco2e
    st.divider()

    # Credits — linked inputs
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
    st.divider()

    # Penalties — linked inputs (red, default 2,400 €/VLSFO-eq t)
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
    st.divider()

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

    # Save defaults (persist both sets so switching is seamless)
    if st.button("💾 Save current inputs as defaults"):
        if "Extra-EU" in voyage_type:
            hsfo_t, lfo_t, mgo_t, bio_t = HSFO_voy_t + HSFO_berth_t, LFO_voy_t + LFO_berth_t, MGO_voy_t + MGO_berth_t, BIO_voy_t + BIO_berth_t
        else:
            hsfo_t, lfo_t, mgo_t, bio_t = HSFO_voy_t, LFO_voy_t, MGO_voy_t, BIO_voy_t  # here voy_t == total_t; berth zeros

        defaults_to_save = {
            "voyage_type": voyage_type,
            # Persist both patterns (total for intra, voyage+berth for extra)
            "HSFO_t": hsfo_t, "LFO_t": lfo_t, "MGO_t": mgo_t, "BIO_t": bio_t,
            "HSFO_voy_t": HSFO_voy_t, "LFO_voy_t": LFO_voy_t, "MGO_voy_t": MGO_voy_t, "BIO_voy_t": BIO_voy_t,
            "HSFO_berth_t": HSFO_berth_t, "LFO_berth_t": LFO_berth_t, "MGO_berth_t": MGO_berth_t, "BIO_berth_t": BIO_berth_t,

            "LCV_HSFO": LCV_HSFO, "LCV_LFO": LCV_LFO, "LCV_MGO": LCV_MGO, "LCV_BIO": LCV_BIO,
            "WtW_HSFO": WtW_HSFO, "WtW_LFO": WtW_LFO, "WtW_MGO": WtW_MGO, "WtW_BIO": WtW_BIO,

            "credit_per_tco2e": credit_per_tco2e,
            "credit_price_eur_per_vlsfo_t": credit_price_eur_per_vlsfo_t,
            "penalty_price_eur_per_vlsfo_t": penalty_price_eur_per_vlsfo_t,
            "consecutive_deficit_years": consecutive_deficit_years,

            "OPS_kWh": OPS_kWh,  # save in kWh
        }
        try:
            with open(DEFAULTS_PATH, "w", encoding="utf-8") as f:
                json.dump(defaults_to_save, f, indent=2)
            st.success("Defaults saved. They will be used next time the app starts.")
        except Exception as e:
            st.error(f"Could not save defaults: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# Derived energies & intensity (use IN-SCOPE energies for compliance)
# ──────────────────────────────────────────────────────────────────────────────
# Build energies from inputs above
energies_fuel_voyage = {
    "HSFO": compute_energy_MJ(HSFO_voy_t, LCV_HSFO),
    "LFO":  compute_energy_MJ(LFO_voy_t,  LCV_LFO),
    "MGO":  compute_energy_MJ(MGO_voy_t,  LCV_MGO),
    "BIO":  compute_energy_MJ(BIO_voy_t,  LCV_BIO),
}
energies_fuel_berth = {
    "HSFO": compute_energy_MJ(HSFO_berth_t, LCV_HSFO),
    "LFO":  compute_energy_MJ(LFO_berth_t,  LCV_LFO),
    "MGO":  compute_energy_MJ(MGO_berth_t,  LCV_MGO),
    "BIO":  compute_energy_MJ(BIO_berth_t,  LCV_BIO),
}
# Totals for display (voyage + berth)
energies_fuel_full = {
    k: energies_fuel_voyage[k] + energies_fuel_berth[k] for k in ["HSFO","LFO","MGO","BIO"]
}
# ELEC (OPS) — always 100% scope; WtW = 0
ELEC_MJ = OPS_MJ

# Build in-scope energies
if "Extra-EU" in voyage_type:
    scoped_energies = scoped_energies_extra_eu(energies_fuel_voyage, energies_fuel_berth, ELEC_MJ)
else:  # Intra-EU
    scoped_energies = {**energies_fuel_full, "ELEC": ELEC_MJ}

# Totals for display
energies_full = {**energies_fuel_full, "ELEC": ELEC_MJ}  # for total MJ readouts
wtw = {"HSFO": WtW_HSFO, "LFO": WtW_LFO, "MGO": WtW_MGO, "BIO": WtW_BIO, "ELEC": 0.0}

E_total_MJ = sum(energies_full.values())
E_scope_MJ = sum(scoped_energies.values())
g_actual = compute_mix_intensity_g_per_MJ(scoped_energies, wtw)  # gCO2e/MJ (in-scope mix)

# Top-of-page breakdown (now bigger & bold via CSS above)
st.subheader("Energy breakdown (MJ)")
cA, cB, cC, cD, cE, cF = st.columns(6)
with cA: st.metric("Total energy (all)", f"{us2(E_total_MJ)} MJ")
with cB: st.metric("In-scope energy", f"{us2(E_scope_MJ)} MJ")
with cC: st.metric("Fossil — all", f"{us2(energies_fuel_full['HSFO'] + energies_fuel_full['LFO'] + energies_fuel_full['MGO'])} MJ")
with cD: st.metric("BIO — all", f"{us2(energies_fuel_full['BIO'])} MJ")
with cE: st.metric("Fossil — in scope", f"{us2(scoped_energies.get('HSFO',0)+scoped_energies.get('LFO',0)+scoped_energies.get('MGO',0))} MJ")
with cF: st.metric("BIO — in scope", f"{us2(scoped_energies.get('BIO',0))} MJ")

# ──────────────────────────────────────────────────────────────────────────────
# Plot — GHG Intensity vs Limit (with inline value labels + dashed “Attained GHG”)
# ──────────────────────────────────────────────────────────────────────────────
st.header("GHG Intensity vs. FuelEU Limit (2025–2050)")

limit_series = LIMITS_DF["Limit_gCO2e_per_MJ"].tolist()
years = LIMITS_DF["Year"].tolist()
actual_series = [g_actual for _ in years]

# Step years where labels should appear on the limit (the change points)
step_years = [2025, 2030, 2035, 2040, 2045, 2050]
limit_text = [f"{limit_series[i]:,.2f}" if years[i] in step_years else "" for i in range(len(years))]

# For Attained GHG text: show only at the last year to avoid clutter
attained_text = ["" for _ in years]
attained_text[-1] = f"{g_actual:,.2f}"

fig = go.Figure()

# Limit (step) — solid line, labels at step years, show markers so labels anchor nicely
fig.add_trace(
    go.Scatter(
        x=years,
        y=limit_series,
        name="FuelEU Limit (step)",
        mode="lines+markers+text",
        line=dict(shape="hv", width=3),
        text=limit_text,
        textposition="top left",
        textfont=dict(size=12),
        hovertemplate="Year=%{x}<br>Limit=%{y:,.2f} gCO₂e/MJ<extra></extra>",
    )
)

# Attained GHG — dashed line, with a value label at the last point
fig.add_trace(
    go.Scatter(
        x=years,
        y=actual_series,
        name="Attained GHG",
        mode="lines+text",
        line=dict(dash="dash", width=3),
        text=attained_text,
        textposition="middle right",
        textfont=dict(size=12),
        hovertemplate="Year=%{x}<br>Attained=%{y:,.2f} gCO₂e/MJ<extra></extra>",
    )
)

fig.update_yaxes(tickformat=",.2f")
fig.update_layout(
    xaxis_title="Year",
    yaxis_title="GHG Intensity [gCO₂e/MJ]",
    hovermode="x unified",
    title=(
        f"Total (all): {us2(E_total_MJ)} MJ • In-scope: {us2(E_scope_MJ)} MJ • "
        f"Fossil (all/in-scope): {us2(energies_fuel_full['HSFO'] + energies_fuel_full['LFO'] + energies_fuel_full['MGO'])} / "
        f"{us2(scoped_energies.get('HSFO',0)+scoped_energies.get('LFO',0)+scoped_energies.get('MGO',0))} MJ • "
        f"BIO (all/in-scope): {us2(energies_fuel_full['BIO'])} / {us2(scoped_energies.get('BIO',0))} MJ • "
        f"ELEC (OPS): {us2(ELEC_MJ)} MJ"
    ),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    margin=dict(l=40, r=20, t=120, b=40),
)
st.plotly_chart(fig, use_container_width=True)
st.caption("ELEC (OPS) is always 100% in scope. For Extra-EU, at-berth fuels are 100% scope; voyage fuels follow the 50% rule with BIO priority.")

# ──────────────────────────────────────────────────────────────────────────────
# Results — merged per-year table (US format, 2 decimals; Year excluded from formatting)
# ──────────────────────────────────────────────────────────────────────────────
st.header("Results (merged per-year table)")

emissions_tco2e = (g_actual * E_scope_MJ) / 1e6  # tCO2e

penalties_eur, credits_eur, net_eur, cb_t = [], [], [], []
multiplier = 1.0 + (max(int(consecutive_deficit_years), 1) - 1) / 10.0

for _, row in LIMITS_DF.iterrows():
    g_target = float(row["Limit_gCO2e_per_MJ"])
    CB_g = (g_target - g_actual) * E_scope_MJ
    CB_t = CB_g / 1e6
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
