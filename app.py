# app.py — FuelEU Maritime Calculator (Simplified, 2025–2050, EUR-only + defaults)
# Scope logic:
#   • Intra-EU: 100% of fuels + 100% of EU OPS electricity (WtW = 0)
#   • Extra-EU: EU OPS electricity 100%; fuels split into:
#       – At-berth fuels (EU ports): 100% scope
#       – Voyage fuels: 50% scope with “Renewables first (BIO then RFNBO); fossil fills remainder pro-rata”
# RFNBO reward:
#   • Until end-2033, RFNBO gets a ×2 reward in the intensity denominator (compliance only, not physical emissions).
# UI/formatting:
#   • Intra-EU → only “Masses [t] — total (voyage + berth)”
#   • Extra-EU → “Masses [t] — voyage (excluding at-berth)” + “At-berth masses [t] (EU ports)”
#   • EU OPS electricity input in kWh (converted to MJ)
#   • US-format numbers (1,234.56); no +/- steppers except for “Consecutive deficit years”
#   • Linked credit/penalty price inputs (€/tCO2e ↔ €/VLSFO-eq t), penalty default 2,400
#   • Energy breakdown labels bigger & bold; numbers smaller to avoid truncation
#   • Chart: “Attained GHG” dashed; step labels below/above; compact header & margins
# Added features (2025-10-08):
#   • Pooling & Banking are independent (each capped vs pre-adjustment surplus):
#       – Pooling: +uptake applies as entered (can overshoot); −provide capped to pre-surplus (never flips to deficit).
#       – Banking: capped to pre-surplus; creates carry-in for next year equal to the final (post-safety-clamp) banked amount.
#   • Processing order per year: Carry-in → Independent Pooling & Banking (vs pre-surplus) → Safety clamp → €.
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

def euros_from_tco2e(balance_tco2e_positive: float, g_attained: float, price_eur_per_vlsfo_t: float) -> float:
    """
    Convert a *positive* tCO2e quantity (surplus for credits, deficit for penalties)
    to euros using the year's attained GHG intensity.
    tCO2e per VLSFO-eq ton = (g_attained [g/MJ] * 41,000 [MJ/t]) / 1e6.
    """
    if balance_tco2e_positive <= 0 or price_eur_per_vlsfo_t <= 0 or g_attained <= 0:
        return 0.0
    tco2e_per_vlsfot = (g_attained * 41_000.0) / 1_000_000.0
    if tco2e_per_vlsfot <= 0:
        return 0.0
    vlsfo_eq_t = balance_tco2e_positive / tco2e_per_vlsfot
    return vlsfo_eq_t * price_eur_per_vlsfo_t

# ──────────────────────────────────────────────────────────────────────────────
# Formatting helpers (US format, 2 decimals)
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

def parse_us_any(s: str, default: float = 0.0) -> float:
    try:
        return float(str(s).replace(",", ""))
    except Exception:
        return float(default)

def float_text_input(label: str, default_val: float, key: str, min_value: float = 0.0) -> float:
    if key not in st.session_state:
        st.session_state[key] = us2(default_val)
    def _normalize():
        val = parse_us(st.session_state[key], default=default_val, min_value=min_value)
        st.session_state[key] = us2(val)
    st.text_input(label, value=st.session_state[key], key=key, on_change=_normalize, label_visibility="visible")
    return parse_us(st.session_state[key], default=default_val, min_value=min_value)

def float_text_input_signed(label: str, default_val: float, key: str) -> float:
    if key not in st.session_state:
        st.session_state[key] = us2(default_val)
    def _normalize():
        val = parse_us_any(st.session_state[key], default=default_val)
        st.session_state[key] = us2(val)
    st.text_input(label, value=st.session_state[key], key=key, on_change=_normalize, label_visibility="visible")
    return parse_us_any(st.session_state[key], default=default_val)

# ──────────────────────────────────────────────────────────────────────────────
# Scoping helpers
# ──────────────────────────────────────────────────────────────────────────────
def scoped_energies_extra_eu(energies_fuel_voyage: Dict[str, float],
                             energies_fuel_berth: Dict[str, float],
                             elec_MJ: float) -> Dict[str, float]:
    """
    Extra-EU scope:
      • 100% scope for at-berth fuels (EU ports) + ELEC (OPS)
      • 50% scope for VOYAGE fuels with Renewables first (BIO then RFNBO); fossil fills remainder pro-rata
    """
    scoped = {
        "HSFO": energies_fuel_berth.get("HSFO", 0.0),
        "LFO":  energies_fuel_berth.get("LFO",  0.0),
        "MGO":  energies_fuel_berth.get("MGO",  0.0),
        "BIO":  energies_fuel_berth.get("BIO",  0.0),
        "RFNBO":energies_fuel_berth.get("RFNBO",0.0),
        "ELEC": elec_MJ,
    }

    fuel_voy_tot = sum(energies_fuel_voyage.values())
    half_scope   = 0.5 * fuel_voy_tot

    # Renewables pool with BIO priority, then RFNBO
    bio_v    = energies_fuel_voyage.get("BIO",   0.0)
    rfnbo_v  = energies_fuel_voyage.get("RFNBO", 0.0)
    ren_total = bio_v + rfnbo_v

    ren_attr = min(ren_total, half_scope)
    bio_attr   = min(bio_v, ren_attr)
    rfnbo_attr = ren_attr - bio_attr

    remainder = half_scope - ren_attr

    # Fossil remainder pro-rata
    hsfo_v = energies_fuel_voyage.get("HSFO", 0.0)
    lfo_v  = energies_fuel_voyage.get("LFO",  0.0)
    mgo_v  = energies_fuel_voyage.get("MGO",  0.0)
    foss_v = hsfo_v + lfo_v + mgo_v

    if foss_v > 0 and remainder > 0:
        hsfo_attr = remainder * (hsfo_v / foss_v)
        lfo_attr  = remainder * (lfo_v  / foss_v)
        mgo_attr  = remainder * (mgo_v  / foss_v)
    else:
        hsfo_attr = lfo_attr = mgo_attr = 0.0

    scoped["HSFO"] += hsfo_attr
    scoped["LFO"]  += lfo_attr
    scoped["MGO"]  += mgo_attr
    scoped["BIO"]  += bio_attr
    scoped["RFNBO"]+= rfnbo_attr
    return scoped

# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="FuelEU Maritime Calculator", layout="wide")

st.title("FuelEU Maritime — GHG Intensity & Cost — TMS DRY")
st.caption("Period: 2025–2050 • Limits derived from 2020 baseline 91.16 gCO₂e/MJ • WtW basis • Prices in EUR")

# Global CSS
st.markdown(
    """
    <style>
    [data-testid="stMetricLabel"] { font-size: 1.00rem !important; font-weight: 800 !important; color: #111827 !important; }
    [data-testid="stMetricValue"] { font-size: 0.95rem !important; font-weight: 700 !important; line-height: 1.10 !important; }
    [data-testid="stMetric"] { padding: .25rem .40rem !important; }
    section[data-testid="stSidebar"] div.block-container{ padding-top: .6rem !important; padding-bottom: .6rem !important; }
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{ gap: .6rem !important; }
    section[data-testid="stSidebar"] [data-testid="column"]{ padding-left:.25rem; padding-right:.25rem; }
    section[data-testid="stSidebar"] label{ font-size: .95rem; margin-bottom: .2rem; font-weight: 600; }
    section[data-testid="stSidebar"] input[type="text"], section[data-testid="stSidebar"] input[type="number"]{
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

    # ── Masses
    if "Extra-EU" in voyage_type:
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
        vm5, _ = st.columns(2)
        with vm5:
            RFNBO_voy_t = float_text_input("RFNBO voyage [t]", _get(DEFAULTS, "RFNBO_voy_t", _get(DEFAULTS, "RFNBO_t", 0.0)),
                                           key="RFNBO_voy_t", min_value=0.0)
        st.divider()

        st.markdown('<div class="section-title">At-berth masses [t] (EU ports)</div>', unsafe_allow_html=True)
        st.markdown('<div class="muted-note">100% in scope; particularly relevant for Extra-EU voyages.</div>', unsafe_allow_html=True)
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
        bm5, _ = st.columns(2)
        with bm5:
            RFNBO_berth_t = float_text_input("RFNBO at berth [t]", _get(DEFAULTS, "RFNBO_berth_t", 0.0),
                                             key="RFNBO_berth_t", min_value=0.0)
    else:
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
        m5, _ = st.columns(2)
        with m5:
            RFNBO_total_t = float_text_input("RFNBO [t]" , _get(DEFAULTS, "RFNBO_t", 0.0),
                                             key="RFNBO_total_t", min_value=0.0)
        HSFO_voy_t, LFO_voy_t, MGO_voy_t, BIO_voy_t, RFNBO_voy_t = HSFO_total_t, LFO_total_t, MGO_total_t, BIO_total_t, RFNBO_total_t
        HSFO_berth_t = LFO_berth_t = MGO_berth_t = BIO_berth_t = RFNBO_berth_t = 0.0

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
    l5, _ = st.columns(2)
    with l5:
        LCV_RFNBO = float_text_input("RFNBO LCV" , _get(DEFAULTS, "LCV_RFNBO", 30_000.0), key="LCV_RFNBO", min_value=0.0)
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
    w5, _ = st.columns(2)
    with w5:
        WtW_RFNBO = float_text_input("RFNBO WtW" , _get(DEFAULTS, "WtW_RFNBO", 20.00), key="WtW_RFNBO", min_value=0.0)
    st.divider()

    # EU OPS electricity
    st.markdown('<div class="section-title">EU OPS electricity</div>', unsafe_allow_html=True)
    OPS_kWh = float_text_input("Electricity delivered (kWh)", _get_ops_kwh_default(), key="OPS_kWh", min_value=0.0)
    OPS_MJ  = OPS_kWh * 3.6
    st.divider()

    # Preview factor for €/tCO2e ↔ €/VLSFO-eq t linking
    if "Extra-EU" in voyage_type:
        energies_preview_fuel_voyage = {
            "HSFO": compute_energy_MJ(HSFO_voy_t,  LCV_HSFO),
            "LFO":  compute_energy_MJ(LFO_voy_t,   LCV_LFO),
            "MGO":  compute_energy_MJ(MGO_voy_t,   LCV_MGO),
            "BIO":  compute_energy_MJ(BIO_voy_t,   LCV_BIO),
            "RFNBO":compute_energy_MJ(RFNBO_voy_t, LCV_RFNBO),
        }
        energies_preview_fuel_berth = {
            "HSFO": compute_energy_MJ(HSFO_berth_t,  LCV_HSFO),
            "LFO":  compute_energy_MJ(LFO_berth_t,   LCV_LFO),
            "MGO":  compute_energy_MJ(MGO_berth_t,   LCV_MGO),
            "BIO":  compute_energy_MJ(BIO_berth_t,   LCV_BIO),
            "RFNBO":compute_energy_MJ(RFNBO_berth_t, LCV_RFNBO),
        }
        energies_preview_scoped = scoped_energies_extra_eu(energies_preview_fuel_voyage,
                                                           energies_preview_fuel_berth,
                                                           OPS_MJ)
    else:
        energies_preview_scoped = {
            "HSFO": compute_energy_MJ(HSFO_voy_t,  LCV_HSFO),
            "LFO":  compute_energy_MJ(LFO_voy_t,   LCV_LFO),
            "MGO":  compute_energy_MJ(MGO_voy_t,   LCV_MGO),
            "BIO":  compute_energy_MJ(BIO_voy_t,   LCV_BIO),
            "RFNBO":compute_energy_MJ(RFNBO_voy_t, LCV_RFNBO),
            "ELEC": OPS_MJ,
        }

    wtw_preview = {"HSFO": WtW_HSFO, "LFO": WtW_LFO, "MGO": WtW_MGO, "BIO": WtW_BIO, "RFNBO": WtW_RFNBO, "ELEC": 0.0}
    num_prev = sum(energies_preview_scoped.get(k,0.0) * wtw_preview.get(k,0.0) for k in wtw_preview.keys())
    den_prev = sum(energies_preview_scoped.values())
    E_rfnbo_prev = energies_preview_scoped.get("RFNBO", 0.0)
    den_prev_rwd = den_prev + E_rfnbo_prev  # r=2
    g_actual_preview = (num_prev / den_prev_rwd) if den_prev_rwd > 0 else 0.0
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
    with c1: st.text_input("Credit price €/tCO₂e", key="credit_per_tco2e_str", on_change=_sync_from_tco2e)
    with c2: st.text_input("Credit price €/VLSFO-eq t", key="credit_per_vlsfo_t_str", on_change=_sync_from_vlsfo)

    credit_per_tco2e = parse_us(st.session_state["credit_per_tco2e_str"], 0.0, 0.0)
    credit_price_eur_per_vlsfo_t = parse_us(st.session_state["credit_per_vlsfo_t_str"], 0.0, 0.0)
    st.divider()

    # Penalties — linked inputs (default 2,400 €/VLSFO-eq t)
    st.markdown('<div class="section-title">Compliance Market — Penalties</div>', unsafe_allow_html=True)
    st.markdown('<div class="penalty-note">Regulated default. Change only if regulation changes.</div>', unsafe_allow_html=True)
    if "penalty_per_vlsfo_t_str" not in st.session_state:
        st.session_state["penalty_per_vlsfo_t_str"] = us2(float(_get(DEFAULTS, "penalty_price_eur_per_vlsfo_t", 2_400.0)))
    if "penalty_per_tco2e_str" not in st.session_state:
        default_pen_vlsfo = parse_us(st.session_state["penalty_per_vlsfo_t_str"], 2400.0, 0.0)
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

    # Other
    st.markdown('<div class="section-title">Other</div>', unsafe_allow_html=True)
    consecutive_deficit_years = int(
        st.number_input("Consecutive deficit years (n)", min_value=1,
                        value=int(_get(DEFAULTS, "consecutive_deficit_years", 1)), step=1)
    )

    # Banking & Pooling (tCO2e) — independent caps (vs pre-adjustment surplus)
    st.divider()
    st.markdown('<div class="section-title">Banking & Pooling (tCO₂e)</div>', unsafe_allow_html=True)

    pooling_tco2e_input = float_text_input_signed(
        "Pooling [tCO₂e]: + uptake tCO2e, − provide tCO2e",
        _get(DEFAULTS, "pooling_tco2e", 0.0),
        key="POOL_T"
    )

    banking_tco2e_input = float_text_input(
        "Banking to next year [tCO₂e] (capped vs pre-surplus)",
        _get(DEFAULTS, "banking_tco2e", 0.0),
        key="BANK_T",
        min_value=0.0
    )

    # Save defaults
    if st.button("💾 Save current inputs as defaults"):
        if "Extra-EU" in voyage_type:
            hsfo_t = HSFO_voy_t + HSFO_berth_t
            lfo_t  = LFO_voy_t  + LFO_berth_t
            mgo_t  = MGO_voy_t  + MGO_berth_t
            bio_t  = BIO_voy_t  + BIO_berth_t
            rfnbo_t= RFNBO_voy_t+ RFNBO_berth_t
        else:
            hsfo_t, lfo_t, mgo_t, bio_t, rfnbo_t = HSFO_voy_t, LFO_voy_t, MGO_voy_t, BIO_voy_t, RFNBO_voy_t
        defaults_to_save = {
            "voyage_type": voyage_type,
            "HSFO_t": hsfo_t, "LFO_t": lfo_t, "MGO_t": mgo_t, "BIO_t": bio_t, "RFNBO_t": rfnbo_t,
            "HSFO_voy_t": HSFO_voy_t, "LFO_voy_t": LFO_voy_t, "MGO_voy_t": MGO_voy_t, "BIO_voy_t": BIO_voy_t, "RFNBO_voy_t": RFNBO_voy_t,
            "HSFO_berth_t": HSFO_berth_t, "LFO_berth_t": LFO_berth_t, "MGO_berth_t": MGO_berth_t, "BIO_berth_t": BIO_berth_t, "RFNBO_berth_t": RFNBO_berth_t,
            "LCV_HSFO": LCV_HSFO, "LCV_LFO": LCV_LFO, "LCV_MGO": LCV_MGO, "LCV_BIO": LCV_BIO, "LCV_RFNBO": LCV_RFNBO,
            "WtW_HSFO": WtW_HSFO, "WtW_LFO": WtW_LFO, "WtW_MGO": WtW_MGO, "WtW_BIO": WtW_BIO, "WtW_RFNBO": WtW_RFNBO,
            "credit_per_tco2e": credit_per_tco2e,
            "credit_price_eur_per_vlsfo_t": credit_price_eur_per_vlsfo_t,
            "penalty_price_eur_per_vlsfo_t": penalty_price_eur_per_vlsfo_t,
            "consecutive_deficit_years": consecutive_deficit_years,
            "OPS_kWh": OPS_kWh,
            "banking_tco2e": banking_tco2e_input,
            "pooling_tco2e": pooling_tco2e_input,
        }
        try:
            with open(DEFAULTS_PATH, "w", encoding="utf-8") as f:
                json.dump(defaults_to_save, f, indent=2)
            st.success("Defaults saved. They will be used next time the app starts.")
        except Exception as e:
            st.error(f"Could not save defaults: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# Derived energies & intensity (IN-SCOPE for compliance)
# ──────────────────────────────────────────────────────────────────────────────
energies_fuel_voyage = {
    "HSFO": compute_energy_MJ(HSFO_voy_t,  LCV_HSFO),
    "LFO":  compute_energy_MJ(LFO_voy_t,   LCV_LFO),
    "MGO":  compute_energy_MJ(MGO_voy_t,   LCV_MGO),
    "BIO":  compute_energy_MJ(BIO_voy_t,   LCV_BIO),
    "RFNBO":compute_energy_MJ(RFNBO_voy_t, LCV_RFNBO),
}
energies_fuel_berth = {
    "HSFO": compute_energy_MJ(HSFO_berth_t,  LCV_HSFO),
    "LFO":  compute_energy_MJ(LFO_berth_t,   LCV_LFO),
    "MGO":  compute_energy_MJ(MGO_berth_t,   LCV_MGO),
    "BIO":  compute_energy_MJ(BIO_berth_t,   LCV_BIO),
    "RFNBO":compute_energy_MJ(RFNBO_berth_t, LCV_RFNBO),
}
energies_fuel_full = {k: energies_fuel_voyage.get(k,0.0) + energies_fuel_berth.get(k,0.0) for k in ["HSFO","LFO","MGO","BIO","RFNBO"]}
ELEC_MJ = OPS_MJ

if "Extra-EU" in voyage_type:
    scoped_energies = scoped_energies_extra_eu(energies_fuel_voyage, energies_fuel_berth, ELEC_MJ)
else:
    scoped_energies = {**energies_fuel_full, "ELEC": ELEC_MJ}

energies_full = {**energies_fuel_full, "ELEC": ELEC_MJ}
wtw = {"HSFO": WtW_HSFO, "LFO": WtW_LFO, "MGO": WtW_MGO, "BIO": WtW_BIO, "RFNBO": WtW_RFNBO, "ELEC": 0.0}

E_total_MJ = sum(energies_full.values())
E_scope_MJ = sum(scoped_energies.values())

num_phys = sum(scoped_energies.get(k,0.0) * wtw.get(k,0.0) for k in wtw.keys())
den_phys = E_scope_MJ
g_base = (num_phys / den_phys) if den_phys > 0 else 0.0

E_rfnbo_scope = scoped_energies.get("RFNBO", 0.0)
def attained_intensity_for_year(y: int) -> float:
    if den_phys <= 0:
        return 0.0
    r = 2.0 if y <= 2033 else 1.0
    den_rwd = den_phys + (r - 1.0) * E_rfnbo_scope
    return num_phys / den_rwd if den_rwd > 0 else 0.0

# ──────────────────────────────────────────────────────────────────────────────
# Top breakdown
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("Energy breakdown (MJ)")
cA, cB, cC, cD, cE, cF, cG, cH = st.columns(8)
with cA: st.metric("Total energy (all)", f"{us2(E_total_MJ)} MJ")
with cB: st.metric("In-scope energy", f"{us2(E_scope_MJ)} MJ")
with cC: st.metric("Fossil — all", f"{us2(energies_fuel_full['HSFO'] + energies_fuel_full['LFO'] + energies_fuel_full['MGO'])} MJ")
with cD: st.metric("BIO — all", f"{us2(energies_fuel_full['BIO'])} MJ")
with cE: st.metric("RFNBO — all", f"{us2(energies_fuel_full['RFNBO'])} MJ")
with cF: st.metric("Fossil — in scope", f"{us2(scoped_energies.get('HSFO',0)+scoped_energies.get('LFO',0)+scoped_energies.get('MGO',0))} MJ")
with cG: st.metric("BIO — in scope", f"{us2(scoped_energies.get('BIO',0))} MJ")
with cH: st.metric("RFNBO — in scope", f"{us2(scoped_energies.get('RFNBO',0))} MJ")

# ──────────────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<h2 style="margin:0 0 .25rem 0;">GHG Intensity vs. FuelEU Limit (2025–2050)</h2>', unsafe_allow_html=True)
years = LIMITS_DF["Year"].tolist()
limit_series = LIMITS_DF["Limit_gCO2e_per_MJ"].tolist()
actual_series = [attained_intensity_for_year(y) for y in years]
step_years = [2025, 2030, 2035, 2040, 2045, 2050]
limit_text = [f"{limit_series[i]:,.2f}" if years[i] in step_years else "" for i in range(len(years))]
attained_text = [f"{actual_series[i]:,.2f}" if years[i] in step_years else "" for i in range(len(years))]

fig = go.Figure()
fig.add_trace(go.Scatter(x=years, y=limit_series, name="FuelEU Limit (step)",
                         mode="lines+markers+text", line=dict(shape="hv", width=3),
                         text=limit_text, textposition="bottom center", textfont=dict(size=12),
                         hovertemplate="Year=%{x}<br>Limit=%{y:,.2f} gCO₂e/MJ<extra></extra>"))
fig.add_trace(go.Scatter(x=years, y=actual_series, name="Attained GHG",
                         mode="lines+text", line=dict(dash="dash", width=3),
                         text=attained_text, textposition="top center", textfont=dict(size=12),
                         hovertemplate="Year=%{x}<br>Attained=%{y:,.2f} gCO₂e/MJ<extra></extra>"))
fig.update_yaxes(tickformat=",.2f")
fig.update_layout(xaxis_title="Year", yaxis_title="GHG Intensity [gCO₂e/MJ]",
                  hovermode="x unified",
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
                  margin=dict(l=40, r=20, t=50, b=40))
st.plotly_chart(fig, use_container_width=True)
st.caption("ELEC (OPS) is always 100% in scope. For Extra-EU, at-berth fuels are 100% scope; voyage fuels follow the 50% rule with renewables first (BIO, then RFNBO).")

# ──────────────────────────────────────────────────────────────────────────────
# Results — Banking/Pooling (independent, capped vs pre-surplus)
# ──────────────────────────────────────────────────────────────────────────────
st.header("Results (merged per-year table)")

emissions_tco2e = (g_base * E_scope_MJ) / 1e6  # physical
multiplier = 1.0 + (max(int(consecutive_deficit_years), 1) - 1) / 10.0

cb_raw_t, carry_in_list, cb_eff_t = [], [], []
pool_applied, bank_applied = [], []
final_balance_t, penalties_eur, credits_eur, net_eur, g_att_list = [], [], [], [], []

info_provide_capped = 0
info_bank_capped = 0
info_bank_ignored_no_surplus = 0
info_final_safety_trim = 0

carry = 0.0  # tCO2e banked from previous year only

for _, row in LIMITS_DF.iterrows():
    year = int(row["Year"])
    g_target = float(row["Limit_gCO2e_per_MJ"])
    g_att = attained_intensity_for_year(year)
    g_att_list.append(g_att)

    # Raw compliance balance (no adjustments)
    CB_g = (g_target - g_att) * E_scope_MJ
    CB_t_raw = CB_g / 1e6
    cb_raw_t.append(CB_t_raw)

    # Pre-adjustment effective balance (anchor for independence)
    cb_eff = CB_t_raw + carry
    carry_in_list.append(carry)
    cb_eff_t.append(cb_eff)

    # ---------- P O O L I N G  (independent, cap vs pre-surplus) ----------
    if pooling_tco2e_input >= 0:
        # Uptake can overshoot (allowed)
        pool_use = pooling_tco2e_input
    else:
        provide_abs = abs(pooling_tco2e_input)
        pre_surplus = max(cb_eff, 0.0)              # cap vs pre-adjustment surplus
        applied_provide = min(provide_abs, pre_surplus)
        if applied_provide < provide_abs:
            info_provide_capped += 1
        pool_use = -applied_provide

    # ---------- B A N K I N G  (independent, cap vs pre-surplus) ----------
    pre_surplus = max(cb_eff, 0.0)                  # same anchor as for pooling
    requested_bank = max(banking_tco2e_input, 0.0)
    bank_use = min(requested_bank, pre_surplus)
    if requested_bank > pre_surplus:
        info_bank_capped += 1
    if pre_surplus == 0.0 and requested_bank > 0.0:
        info_bank_ignored_no_surplus += 1

    # ---------- F I N A L   B A L A N C E  &  € ----------
    final_bal = cb_eff + pool_use - bank_use

    # Safety clamp: never flip a surplus to a deficit (trim bank first, then provide)
    if final_bal < 0:
        needed = -final_bal
        trim_bank = min(needed, bank_use)
        bank_use -= trim_bank
        needed -= trim_bank
        if needed > 0 and pool_use < 0:
            # Reduce the absolute provide (i.e., provide less)
            pool_use += needed
            needed = 0.0
        final_bal = cb_eff + pool_use - bank_use
        info_final_safety_trim += 1

    # Carry to next year is the *final* banked amount (after any trims)
    carry_next = bank_use

    final_balance_t.append(final_bal)

    # Money
    if final_bal > 0:
        credit_val = euros_from_tco2e(final_bal, g_att, credit_price_eur_per_vlsfo_t)
        penalty_val = 0.0
    elif final_bal < 0:
        penalty_val = euros_from_tco2e(-final_bal, g_att, penalty_price_eur_per_vlsfo_t) * multiplier
        credit_val = 0.0
    else:
        credit_val = penalty_val = 0.0

    pool_applied.append(pool_use)
    bank_applied.append(bank_use)
    penalties_eur.append(penalty_val)
    credits_eur.append(credit_val)
    net_eur.append(credit_val - penalty_val)

    # Prepare next loop
    carry = carry_next

# Informational notes (no hard errors)
if info_provide_capped > 0:
    st.info(f"Pooling (provide < 0) capped vs pre-surplus in {info_provide_capped} year(s).")
if info_bank_capped > 0:
    st.info(f"Banking capped vs pre-surplus in {info_bank_capped} year(s).")
if info_bank_ignored_no_surplus > 0:
    st.info(f"Banking request ignored (no pre-surplus) in {info_bank_ignored_no_surplus} year(s).")
if info_final_safety_trim > 0:
    st.info(f"Final safety trim applied in {info_final_safety_trim} year(s) to avoid flipping surplus to deficit.")

# Table
df_cost = pd.DataFrame(
    {
        "Year": years,
        "Reduction_%": LIMITS_DF["Reduction_%"].tolist(),
        "Limit_gCO2e_per_MJ": LIMITS_DF["Limit_gCO2e_per_MJ"].tolist(),
        "Actual_gCO2e_per_MJ": g_att_list,
        "Emissions_tCO2e": [emissions_tco2e]*len(years),
        "Compliance_Balance_tCO2e": cb_raw_t,   # raw
        "CarryIn_Banked_tCO2e": carry_in_list,  # from previous year
        "Banked_to_Next_Year_tCO2e": bank_applied,
        "Effective_Balance_tCO2e": cb_eff_t,    # before adjustments
        "Pooling_tCO2e_Applied": pool_applied,  # +uptake / −provide (capped vs pre-surplus)
        "Final_Balance_tCO2e_for_€": final_balance_t,  # after independent pooling & banking (+ safety)
        "Penalty_EUR": penalties_eur,
        "Credit_EUR": credits_eur,
        "Net_EUR": net_eur,
    }
)

# US-format (except Year)
df_fmt = df_cost.copy()
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
