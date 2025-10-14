# app.py — FuelEU Maritime Calculator (Simplified, 2025–2050, EUR-only + defaults)
# Scope logic:
#   • Intra-EU: 100% of fuels + 100% of EU OPS electricity (WtW = 0)
#   • Extra-EU: EU OPS electricity 100%; fuels split into:
#       – At-berth fuels (EU ports): 100% scope
#       – Voyage fuels: 50% scope
#     ⤷ Requested allocator (WtW-prioritized, pool-then-fill):
#         1) Build in-scope fuel pool = 100% at-berth fuels + 50% of total voyage fuels (ELEC always 100% in scope).
#         2) Fill the pool (without changing its total) by WtW priority:
#            a) Renewables (RFNBO vs BIO): lower WtW first; at-berth 100% first, then voyage up to spare capacity
#               (leaving enough room to ensure all fossil at-berth still fit 100%).
#            b) Fossil at-berth (HSFO, LFO, MGO): 100% in ascending WtW.
#            c) Fossil voyage  (HSFO, LFO, MGO): 50% per fuel in ascending WtW, partial on the last if needed.
# RFNBO reward:
#   • Until end-2033, RFNBO gets a ×2 reward in the intensity denominator (compliance only, not physical emissions).
# UI/formatting:
#   • Intra-EU → only “Masses [t] — total (voyage + berth)”
#   • Extra-EU → “Masses [t] — voyage (excluding at-berth)” + “At-berth masses [t] (EU ports)”
#   • EU OPS electricity input in kWh (converted to MJ)
#   • US-format numbers (1,234.56); no +/- steppers except for “Consecutive deficit years (seed)”
#   • Linked price inputs (see below priorities)
#   • Energy breakdown labels bigger & bold; numbers smaller to avoid truncation
#   • Chart: “Attained GHG” dashed; step labels below/above; compact header & margins
#
# Added features (2025-10-08):
#   • Pooling & Banking are independent (each capped vs pre-adjustment surplus):
#       – Pooling: +uptake applies as entered (can overshoot); −provide capped to pre-surplus (never flips to deficit).
#       – Banking: capped vs pre-surplus; creates carry-in for next year equal to *final* banked amount.
#   • Start-year selectors for Pooling and Banking (placed just below each input).
#   • Price linking (one-way priorities, updated automatically without on_change):
#       – Penalties: user edits €/VLSFO-eq t; €/tCO2e is derived (read-only).
#       – Credits:   user edits €/tCO2e;   €/VLSFO-eq t is derived (read-only).
#   • Processing per year: Carry-in → Independent Pooling & Banking (vs pre-surplus) → Safety clamp → €.
#   • Consecutive-deficit multiplier (automatic): +10% per additional consecutive deficit year; seeded from UI.
# --------------------------------------------------------------------------------------
from __future__ import annotations

import json
import os
from typing import Dict, Any, Tuple

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

def float_text_input(label: str, default_val: float, key: str, min_value: float = 0.0, label_visibility: str = "visible") -> float:
    if key not in st.session_state:
        st.session_state[key] = us2(default_val)
    def _normalize():
        val = parse_us(st.session_state[key], default=default_val, min_value=min_value)
        st.session_state[key] = us2(val)
    st.text_input(label, value=st.session_state[key], key=key, on_change=_normalize, label_visibility=label_visibility)
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
                             elec_MJ: float,
                             wtw: Dict[str, float]) -> Dict[str, float]:
    """
    Extra-EU (requested variant, with berth-100% guarantee):
      • Build one in-scope fuel pool: ELEC (100%) + at-berth fuels (100%) + 50% of *total* voyage fuels
      • Fill the pool by WtW priority, without changing its total:
          1) Renewables (RFNBO, BIO), lower WtW first:
             - take AT-BERTH part first (100%),
             - then take VOYAGE part but only up to the spare pool after reserving all fossil at-berth.
          2) Fossil AT-BERTH (HSFO, LFO, MGO): 100% in ascending WtW.
          3) Fossil VOYAGE (HSFO, LFO, MGO): take 50% of each fuel in ascending WtW; partial on last to close pool.
      • ELEC is always 100% in-scope and excluded from the competition.
    """
    def g(d, k): return float(d.get(k, 0.0))

    fossils = ["HSFO", "LFO", "MGO"]
    foss_sorted = sorted(fossils, key=lambda f: wtw.get(f, float("inf")))

    total_voy = sum(energies_fuel_voyage.values())
    half_voy  = 0.5 * total_voy
    total_berth = sum(energies_fuel_berth.values())
    berth_fossil_total = sum(g(energies_fuel_berth, f) for f in fossils)

    pool_total = total_berth + half_voy

    scoped = {k: 0.0 for k in ["HSFO","LFO","MGO","BIO","RFNBO","ELEC"]}
    scoped["ELEC"] = max(elec_MJ, 0.0)

    remaining = pool_total  # fuels only

    # 1) Renewables (at-berth first, then voyage up to spare after reserving berth fossils)
    ren = ["RFNBO", "BIO"]
    ren_sorted = sorted(ren, key=lambda f: wtw.get(f, float("inf")))

    for f in ren_sorted:
        amt_b = g(energies_fuel_berth, f)
        take_b = min(amt_b, remaining)
        if take_b > 0:
            scoped[f] += take_b
            remaining -= take_b
        if remaining <= 0:
            return scoped

        amt_v = g(energies_fuel_voyage, f)
        spare_for_voy_ren = max(0.0, remaining - berth_fossil_total)
        take_v = min(amt_v, spare_for_voy_ren)
        if take_v > 0:
            scoped[f] += take_v
            remaining -= take_v
        if remaining <= 0:
            return scoped

    # 2) Fossil at-berth — 100% in ascending WtW
    for f in foss_sorted:
        amt = g(energies_fuel_berth, f)
        take = min(amt, remaining)
        if take > 0:
            scoped[f] += take
            remaining -= take
        if remaining <= 0:
            return scoped

    # 3) Fossil voyage — 50% per fuel in ascending WtW, partial on last to close pool
    for f in foss_sorted:
        half_voy_fuel = 0.5 * g(energies_fuel_voyage, f)
        if half_voy_fuel <= 0 or remaining <= 0:
            continue
        take = min(half_voy_fuel, remaining)
        scoped[f] += take
        remaining -= take
        if remaining <= 0:
            return scoped

    return scoped

# ──────────────────────────────────────────────────────────────────────────────
# PRECOMPUTE PREVIEW FACTOR (for first-render derived fields)
# Uses session_state if present, otherwise defaults; runs before rendering sidebar.
# ──────────────────────────────────────────────────────────────────────────────
def _ss_or_def(key: str, default_val: float) -> float:
    if key in st.session_state:
        try:
            return parse_us_any(st.session_state[key], default=default_val)
        except Exception:
            return float(default_val)
    return float(_get(DEFAULTS, key, default_val))

def _compute_preview_factor_first_pass() -> float:
    # Voyage scope
    voyage_type_saved = st.session_state.get("voyage_type", _get(DEFAULTS, "voyage_type", "Intra-EU (100%)"))

    # LCVs
    LCV_HSFO_pre = _ss_or_def("LCV_HSFO", _get(DEFAULTS, "LCV_HSFO", 40200.0))
    LCV_LFO_pre  = _ss_or_def("LCV_LFO",  _get(DEFAULTS, "LCV_LFO",  42700.0))
    LCV_MGO_pre  = _ss_or_def("LCV_MGO",  _get(DEFAULTS, "LCV_MGO",  42700.0))
    LCV_BIO_pre  = _ss_or_def("LCV_BIO",  _get(DEFAULTS, "LCV_BIO",  38000.0))
    LCV_RFN_pre  = _ss_or_def("LCV_RFNBO",_get(DEFAULTS, "LCV_RFNBO",30000.0))

    # WtW
    W_HSFO_pre = _ss_or_def("WtW_HSFO", _get(DEFAULTS, "WtW_HSFO", 92.78))
    W_LFO_pre  = _ss_or_def("WtW_LFO",  _get(DEFAULTS, "WtW_LFO",  92.00))
    W_MGO_pre  = _ss_or_def("WtW_MGO",  _get(DEFAULTS, "WtW_MGO",  93.93))
    W_BIO_pre  = _ss_or_def("WtW_BIO",  _get(DEFAULTS, "WtW_BIO",  70.00))
    W_RFN_pre  = _ss_or_def("WtW_RFNBO",_get(DEFAULTS, "WtW_RFNBO",20.00))
    wtw_pre = {"HSFO": W_HSFO_pre, "LFO": W_LFO_pre, "MGO": W_MGO_pre, "BIO": W_BIO_pre, "RFNBO": W_RFN_pre, "ELEC": 0.0}

    # Masses + OPS
    OPS_kWh_pre = _ss_or_def("OPS_kWh", _get_ops_kwh_default())
    OPS_MJ_pre  = OPS_kWh_pre * 3.6

    if "Extra-EU" in voyage_type_saved:
        HSFO_v = _ss_or_def("HSFO_voy_t", _get(DEFAULTS, "HSFO_voy_t", _get(DEFAULTS, "HSFO_t", 5000.0)))
        LFO_v  = _ss_or_def("LFO_voy_t",  _get(DEFAULTS, "LFO_voy_t",  _get(DEFAULTS, "LFO_t", 0.0)))
        MGO_v  = _ss_or_def("MGO_voy_t",  _get(DEFAULTS, "MGO_voy_t",  _get(DEFAULTS, "MGO_t", 0.0)))
        BIO_v  = _ss_or_def("BIO_voy_t",  _get(DEFAULTS, "BIO_voy_t",  _get(DEFAULTS, "BIO_t", 0.0)))
        RFN_v  = _ss_or_def("RFNBO_voy_t",_get(DEFAULTS, "RFNBO_voy_t",_get(DEFAULTS, "RFNBO_t", 0.0)))

        HSFO_b = _ss_or_def("HSFO_berth_t", _get(DEFAULTS, "HSFO_berth_t", 0.0))
        LFO_b  = _ss_or_def("LFO_berth_t",  _get(DEFAULTS, "LFO_berth_t",  0.0))
        MGO_b  = _ss_or_def("MGO_berth_t",  _get(DEFAULTS, "MGO_berth_t",  0.0))
        BIO_b  = _ss_or_def("BIO_berth_t",  _get(DEFAULTS, "BIO_berth_t",  0.0))
        RFN_b  = _ss_or_def("RFNBO_berth_t",_get(DEFAULTS, "RFNBO_berth_t",0.0))

        energies_v = {
            "HSFO": compute_energy_MJ(HSFO_v, LCV_HSFO_pre),
            "LFO":  compute_energy_MJ(LFO_v,  LCV_LFO_pre),
            "MGO":  compute_energy_MJ(MGO_v,  LCV_MGO_pre),
            "BIO":  compute_energy_MJ(BIO_v,  LCV_BIO_pre),
            "RFNBO":compute_energy_MJ(RFN_v,  LCV_RFN_pre),
        }
        energies_b = {
            "HSFO": compute_energy_MJ(HSFO_b, LCV_HSFO_pre),
            "LFO":  compute_energy_MJ(LFO_b,  LCV_LFO_pre),
            "MGO":  compute_energy_MJ(MGO_b,  LCV_MGO_pre),
            "BIO":  compute_energy_MJ(BIO_b,  LCV_BIO_pre),
            "RFNBO":compute_energy_MJ(RFN_b,  LCV_RFN_pre),
        }
        scoped_pre = scoped_energies_extra_eu(energies_v, energies_b, OPS_MJ_pre, wtw_pre)
    else:
        HSFO_t = _ss_or_def("HSFO_total_t", _get(DEFAULTS, "HSFO_t", 5000.0))
        LFO_t  = _ss_or_def("LFO_total_t",  _get(DEFAULTS, "LFO_t",  0.0))
        MGO_t  = _ss_or_def("MGO_total_t",  _get(DEFAULTS, "MGO_t",  0.0))
        BIO_t  = _ss_or_def("BIO_total_t",  _get(DEFAULTS, "BIO_t",  0.0))
        RFN_t  = _ss_or_def("RFNBO_total_t",_get(DEFAULTS, "RFNBO_t",0.0))
        full = {
            "HSFO": compute_energy_MJ(HSFO_t, LCV_HSFO_pre),
            "LFO":  compute_energy_MJ(LFO_t,  LCV_LFO_pre),
            "MGO":  compute_energy_MJ(MGO_t,  LCV_MGO_pre),
            "BIO":  compute_energy_MJ(BIO_t,  LCV_BIO_pre),
            "RFNBO":compute_energy_MJ(RFN_t,  LCV_RFN_pre),
            "ELEC": OPS_MJ_pre,
        }
        scoped_pre = full

    num_pre = sum(scoped_pre.get(k,0.0) * wtw_pre.get(k,0.0) for k in wtw_pre.keys())
    den_pre = sum(scoped_pre.values())
    E_rfnbo_pre = scoped_pre.get("RFNBO", 0.0)
    den_pre_rwd = den_pre + E_rfnbo_pre  # r=2 preview effect
    g_preview = (num_pre / den_pre_rwd) if den_pre_rwd > 0 else 0.0
    return (g_preview * 41_000.0) / 1_000_000.0 if g_preview > 0 else 0.0

# Store factor for immediate first render
st.session_state["factor_vlsfo_per_tco2e"] = _compute_preview_factor_first_pass()

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
st.markdown("""
<style>
[data-testid="stDataFrame"] div[role="columnheader"],
[data-testid="stDataFrame"] div[role="gridcell"]{
  padding: 2px 6px !important;
}
[data-testid="stDataFrame"] div[role="columnheader"] *{
  white-space: normal !important;
  overflow-wrap: anywhere !important;
  word-break: break-word !important;
  text-align: center !important;
}
[data-testid="stDataFrame"] div[role="columnheader"]{
  justify-content: center !important;
}
[data-testid="stDataFrame"] div[role="gridcell"] *{
  white-space: normal !important;
  overflow-wrap: anywhere !important;
  word-break: break-word !important;
  text-align: center !important;
}
[data-testid="stDataFrame"] div[role="gridcell"]{
  justify-content: center !important;
}
[data-testid="stDataFrame"] { font-size: 0.85rem !important; }
</style>
""", unsafe_allow_html=True)

# Sidebar — structured groups with conditional rendering
with st.sidebar:
    # ── Currency (at top)
    st.markdown('<div class="section-title">Exchange rate</div>', unsafe_allow_html=True)
    eur_usd_fx = float_text_input("1 Euro = … USD", _get(DEFAULTS, "eur_usd_fx", 1.00),
                                  key="eur_usd_fx", min_value=0.0)

    # Use precomputed factor for first render (then it will be recomputed again below)
    factor_vlsfo_per_tco2e_prev = float(st.session_state.get("factor_vlsfo_per_tco2e", 0.0))

    # ── Compliance Market — Credits (moved up)
    st.markdown('<div class="section-title">Compliance Market — Credits</div>', unsafe_allow_html=True)
    credit_per_tco2e = float_text_input("Credit price €/tCO₂e",
                                        _get(DEFAULTS, "credit_per_tco2e", 200.0),
                                        key="credit_per_tco2e_str", min_value=0.0)
    credit_price_eur_per_vlsfo_t = credit_per_tco2e * factor_vlsfo_per_tco2e_prev if factor_vlsfo_per_tco2e_prev > 0 else 0.0
    st.text_input("Credit price €/VLSFO-eq t (derived)", value=us2(credit_price_eur_per_vlsfo_t), disabled=True)

    # ── Compliance Market — Penalties (moved up)
    st.markdown('<div class="section-title">Compliance Market — Penalties</div>', unsafe_allow_html=True)
    st.markdown('<div class="penalty-note">Regulated default. Change only if regulation changes.</div>', unsafe_allow_html=True)
    penalty_price_eur_per_vlsfo_t = float_text_input("Penalty price €/VLSFO-eq t",
                                                     _get(DEFAULTS, "penalty_price_eur_per_vlsfo_t", 2_400.0),
                                                     key="penalty_per_vlsfo_t_str", min_value=0.0)
    penalty_price_eur_per_tco2e_prev = (penalty_price_eur_per_vlsfo_t / factor_vlsfo_per_tco2e_prev) if factor_vlsfo_per_tco2e_prev > 0 else 0.0
    st.text_input("Penalty price €/tCO₂e (derived)", value=us2(penalty_price_eur_per_tco2e_prev), disabled=True)

    # ── BIO premium (bold label + collapsed input label)
    st.markdown('<div class="section-title"><span style="font-weight:800;">Premium BIO vs HSFO [USD/ton]</span></div>', unsafe_allow_html=True)
    bio_premium_usd_per_t = float_text_input(
        "",
        _get(DEFAULTS, "bio_premium_usd_per_t", 0.0),
        key="bio_premium_usd_per_t",
        min_value=0.0,
        label_visibility="collapsed"
    )

    st.divider()

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
        # Map totals to voyage variables for unified downstream handling
        HSFO_voy_t, LFO_voy_t, MGO_voy_t, BIO_voy_t, RFNBO_voy_t = HSFO_total_t, LFO_total_t, MGO_total_t, BIO_total_t, RFNBO_total_t
        HSFO_berth_t = LFO_berth_t = MGO_berth_t = BIO_berth_t = RFNBO_berth_t = 0.0

    # ── LCVs
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

    # ── WtWs
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

    # ── EU OPS electricity
    st.markdown('<div class="section-title">EU OPS electricity</div>', unsafe_allow_html=True)
    OPS_kWh = float_text_input("Electricity delivered (kWh)", _get_ops_kwh_default(), key="OPS_kWh", min_value=0.0)
    OPS_MJ  = OPS_kWh * 3.6
    st.text_input("Electricity delivered (MJ) (derived)", value=us2(OPS_MJ), disabled=True)
    # (no extra divider here to keep spacing tight)

    # Preview factor (recompute with CURRENT sidebar values; also updates session_state)
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
        wtw_preview = {"HSFO": WtW_HSFO, "LFO": WtW_LFO, "MGO": WtW_MGO, "BIO": WtW_BIO, "RFNBO": WtW_RFNBO, "ELEC": 0.0}
        energies_preview_scoped = scoped_energies_extra_eu(
            energies_preview_fuel_voyage,
            energies_preview_fuel_berth,
            OPS_MJ,
            wtw_preview
        )
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
    den_prev_rwd = den_prev + E_rfnbo_prev  # r=2 in preview
    g_actual_preview = (num_prev / den_prev_rwd) if den_prev_rwd > 0 else 0.0
    factor_vlsfo_per_tco2e = (g_actual_preview * 41_000.0) / 1_000_000.0 if g_actual_preview > 0 else 0.0
    st.session_state["factor_vlsfo_per_tco2e"] = factor_vlsfo_per_tco2e
    st.divider()

    # Other
    st.markdown('<div class="section-title">Other</div>', unsafe_allow_html=True)
    consecutive_deficit_years_seed = int(
        st.number_input("Consecutive deficit years (seed)", min_value=1,
                        value=int(_get(DEFAULTS, "consecutive_deficit_years", 1)), step=1)
    )

    # NEW — Fuel to reduce (placed exactly here)
    opt_fuels = ["HSFO", "LFO", "MGO"]
    _saved_opt_fuel = _get(DEFAULTS, "opt_reduce_fuel", "HSFO")
    try:
        _idx = opt_fuels.index(_saved_opt_fuel)
    except ValueError:
        _idx = 0
    selected_fuel_for_opt = st.selectbox("Fuel to reduce (for optimization)", opt_fuels, index=_idx)

    # Banking & Pooling (tCO2e)
    st.divider()
    st.markdown('<div class="section-title">Banking & Pooling (tCO₂e)</div>', unsafe_allow_html=True)
    pooling_tco2e_input = float_text_input_signed(
        "Pooling [tCO₂e]: + uptake tCO2e, − provide tCO2e",
        _get(DEFAULTS, "pooling_tco2e", 0.0),
        key="POOL_T"
    )
    pooling_start_year = st.selectbox("Pooling starts from year",
                                      YEARS,
                                      index=YEARS.index(int(_get(DEFAULTS, "pooling_start_year", YEARS[0]))))
    banking_tco2e_input = float_text_input(
        "Banking to next year [tCO₂e]",
        _get(DEFAULTS, "banking_tco2e", 0.0),
        key="BANK_T",
        min_value=0.0
    )
    banking_start_year = st.selectbox("Banking starts from year",
                                      YEARS,
                                      index=YEARS.index(int(_get(DEFAULTS, "banking_start_year", YEARS[0]))))

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
            "eur_usd_fx": eur_usd_fx,
            "bio_premium_usd_per_t": bio_premium_usd_per_t,
            "HSFO_t": hsfo_t, "LFO_t": lfo_t, "MGO_t": mgo_t, "BIO_t": bio_t, "RFNBO_t": rfnbo_t,
            "HSFO_voy_t": HSFO_voy_t, "LFO_voy_t": LFO_voy_t, "MGO_voy_t": MGO_voy_t, "BIO_voy_t": BIO_voy_t, "RFNBO_voy_t": RFNBO_voy_t,
            "HSFO_berth_t": HSFO_berth_t, "LFO_berth_t": LFO_berth_t, "MGO_berth_t": MGO_berth_t, "BIO_berth_t": BIO_berth_t, "RFNBO_berth_t": RFNBO_berth_t,
            "LCV_HSFO": LCV_HSFO, "LCV_LFO": LCV_LFO, "LCV_MGO": LCV_MGO, "LCV_BIO": LCV_BIO, "LCV_RFNBO": LCV_RFNBO,
            "WtW_HSFO": WtW_HSFO, "WtW_LFO": WtW_LFO, "WtW_MGO": WtW_MGO, "WtW_BIO": WtW_BIO, "WtW_RFNBO": WtW_RFNBO,
            "credit_per_tco2e": credit_per_tco2e,
            "penalty_price_eur_per_vlsfo_t": penalty_price_eur_per_vlsfo_t,
            "consecutive_deficit_years": consecutive_deficit_years_seed,
            "OPS_kWh": OPS_kWh,
            "banking_tco2e": banking_tco2e_input,
            "pooling_tco2e": pooling_tco2e_input,
            "pooling_start_year": int(pooling_start_year),
            "banking_start_year": int(banking_start_year),
            "opt_reduce_fuel": selected_fuel_for_opt,
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

wtw = {"HSFO": WtW_HSFO, "LFO": WtW_LFO, "MGO": WtW_MGO, "BIO": WtW_BIO, "RFNBO": WtW_RFNBO, "ELEC": 0.0}
if "Extra-EU" in voyage_type:
    scoped_energies = scoped_energies_extra_eu(
        energies_fuel_voyage,
        energies_fuel_berth,
        ELEC_MJ,
        wtw
    )
else:
    scoped_energies = {**energies_fuel_full, "ELEC": ELEC_MJ}

energies_full = {**energies_fuel_full, "ELEC": ELEC_MJ}

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
# Visual — Two stacked columns with dashed connectors & % labels (centers)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<h2 style="margin:0 0 .25rem 0;">Energy composition — all vs in-scope</h2>',
            unsafe_allow_html=True)

categories = ["All energy", "In-scope energy"]
fuels = ["RFNBO", "BIO", "HSFO", "LFO", "MGO"]
fuels_sorted = sorted(fuels, key=lambda f: wtw.get(f, float("inf")))
stack_layers = [("ELEC", "ELEC (OPS)")] + [(f, f) for f in fuels_sorted]

COLORS = {
    "ELEC":  "#FACC15",
    "RFNBO": "#86EFAC",
    "BIO":   "#065F46",
    "MGO":   "#93C5FD",
    "LFO":   "#2563EB",
    "HSFO":  "#1E3A8A",
}

left_vals = {
    "ELEC":  ELEC_MJ,
    "RFNBO": energies_fuel_full.get("RFNBO", 0.0),
    "BIO":   energies_fuel_full.get("BIO",   0.0),
    "HSFO":  energies_fuel_full.get("HSFO",  0.0),
    "LFO":   energies_fuel_full.get("LFO",   0.0),
    "MGO":   energies_fuel_full.get("MGO",   0.0),
}
right_vals = {
    "ELEC":  scoped_energies.get("ELEC",  0.0),
    "RFNBO": scoped_energies.get("RFNBO", 0.0),
    "BIO":   scoped_energies.get("BIO",   0.0),
    "HSFO":  scoped_energies.get("HSFO",  0.0),
    "LFO":   scoped_energies.get("LFO",   0.0),
    "MGO":   scoped_energies.get("MGO",   0.0),
}

fig_stacks = go.Figure()
for key, label in stack_layers:
    fig_stacks.add_trace(
        go.Bar(
            x=categories,
            y=[left_vals.get(key, 0.0), right_vals.get(key, 0.0)],
            name=label,
            marker_color=COLORS.get(key, None),
            hovertemplate=f"{label}<br>%{{x}}<br>%{{y:,.2f}} MJ<extra></extra>",
        )
    )

total_all = sum(left_vals.values())
total_scope = sum(right_vals.values())
fig_stacks.add_annotation(x=categories[0], y=total_all,  text=f"{us2(total_all)} MJ",  showarrow=False, yshift=10, font=dict(size=12))
fig_stacks.add_annotation(x=categories[1], y=total_scope, text=f"{us2(total_scope)} MJ", showarrow=False, yshift=10, font=dict(size=12))

cum_left = 0.0
cum_right = 0.0
for key, label in stack_layers:
    layer_left = float(left_vals.get(key, 0.0))
    layer_right = float(right_vals.get(key, 0.0))
    if layer_left <= 0.0 and layer_right <= 0.0:
        cum_left += layer_left; cum_right += layer_right
        continue
    y_center_left = cum_left + (layer_left / 2.0)
    y_center_right = cum_right + (layer_right / 2.0)

    fig_stacks.add_trace(
        go.Scatter(x=categories, y=[y_center_left, y_center_right], mode="lines",
                   line=dict(dash="dot", width=2), hoverinfo="skip", showlegend=False)
    )
    fig_stacks.add_annotation(x=categories[1], y=y_center_right, ax=categories[0], ay=y_center_left,
                              xref="x", yref="y", axref="x", ayref="y", text="", showarrow=True,
                              arrowhead=3, arrowsize=1.2, arrowwidth=2, arrowcolor="rgba(0,0,0,0.65)")
    pct = (layer_right / layer_left) * 100.0 if layer_left > 0 else 100.0
    pct = max(min(pct, 100.0), 0.0)
    y_mid = 0.5 * (y_center_left + y_center_right)
    fig_stacks.add_annotation(xref="paper", yref="y", x=0.5, y=y_mid, text=f"{pct:.0f}%", showarrow=False,
                              font=dict(size=11, color="#374151"),
                              bgcolor="rgba(255,255,255,0.65)", bordercolor="rgba(0,0,0,0)", borderpad=1)
    cum_left += layer_left; cum_right += layer_right

fig_stacks.update_layout(
    barmode="stack", xaxis_title="", yaxis_title="Energy [MJ]", hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    margin=dict(l=40, r=20, t=50, b=20), bargap=0.35,
)
st.plotly_chart(fig_stacks, use_container_width=True)

if "Extra-EU" in voyage_type:
    st.caption("Left = total energy (voyage + at-berth + OPS). Right = in-scope after priority: RFNBO/BIO (at-berth then voyage up to spare), then all at-berth fossils (100%), then 50% of voyage fossils (ascending WtW; partial on last). ELEC is always 100%.")
else:
    st.caption("Intra-EU: all energy is in scope. Bars ordered by ELEC then ascending WtW; dashed labels should read 100% for each fuel.")

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
st.caption("ELEC (OPS) is always 100% in scope. For Extra-EU, at-berth fuels are 100% scope; voyage fuels are 50% scope; allocation follows the priority described above.")

# ──────────────────────────────────────────────────────────────────────────────
# Results — Banking/Pooling + auto deficit multiplier
# ──────────────────────────────────────────────────────────────────────────────
st.header("Results (merged per-year table)")

emissions_tco2e = (g_base * E_scope_MJ) / 1e6  # physical

cb_raw_t, carry_in_list, cb_eff_t = [], [], []
pool_applied, bank_applied = [], []
final_balance_t, penalties_eur, credits_eur, g_att_list = [], [], [], []
penalties_usd, credits_usd = [], []

info_provide_capped = 0
info_bank_capped = 0
info_bank_ignored_no_surplus = 0
info_final_safety_trim = 0

carry = 0.0  # tCO2e banked from previous year only

prior_seed = max(int(consecutive_deficit_years_seed) - 1, 0)
applied_seed = False
deficit_run = 0

for _, row in LIMITS_DF.iterrows():
    year = int(row["Year"])
    g_target = float(row["Limit_gCO2e_per_MJ"])
    g_att = attained_intensity_for_year(year)
    g_att_list.append(g_att)

    CB_g = (g_target - g_att) * E_scope_MJ
    CB_t_raw = CB_g / 1e6
    cb_raw_t.append(CB_t_raw)

    cb_eff = CB_t_raw + carry
    carry_in_list.append(carry)
    cb_eff_t.append(cb_eff)

    # Pooling
    if year >= int(pooling_start_year):
        if pooling_tco2e_input >= 0:
            pool_use = pooling_tco2e_input
        else:
            provide_abs = abs(pooling_tco2e_input)
            pre_surplus = max(cb_eff, 0.0)
            applied_provide = min(provide_abs, pre_surplus)
            if applied_provide < provide_abs:
                info_provide_capped += 1
            pool_use = -applied_provide
    else:
        pool_use = 0.0

    # Banking
    if year >= int(banking_start_year):
        pre_surplus = max(cb_eff, 0.0)
        requested_bank = max(banking_tco2e_input, 0.0)
        bank_use = min(requested_bank, pre_surplus)
        if requested_bank > pre_surplus:
            info_bank_capped += 1
        if pre_surplus == 0.0 and requested_bank > 0.0:
            info_bank_ignored_no_surplus += 1
    else:
        bank_use = 0.0

    # Final balance & clamp
    final_bal = cb_eff + pool_use - bank_use
    if final_bal < 0:
        needed = -final_bal
        trim_bank = min(needed, bank_use)
        bank_use -= trim_bank
        needed -= trim_bank
        if needed > 0 and pool_use < 0:
            pool_use += needed
            needed = 0.0
        final_bal = cb_eff + pool_use - bank_use
        info_final_safety_trim += 1

    carry_next = bank_use

    # Consecutive-deficit multiplier
    if final_bal < 0:
        if not applied_seed:
            deficit_run = prior_seed + 1
            applied_seed = True
        else:
            deficit_run += 1
        multiplier_y = 1.0 + max(deficit_run - 1, 0) * 0.10
    else:
        deficit_run = 0
        multiplier_y = 1.0

    # € → USD
    if final_bal > 0:
        credit_val = euros_from_tco2e(final_bal, g_att, (credit_per_tco2e * st.session_state["factor_vlsfo_per_tco2e"] if st.session_state["factor_vlsfo_per_tco2e"]>0 else 0.0))
        penalty_val = 0.0
    elif final_bal < 0:
        penalty_val = euros_from_tco2e(-final_bal, g_att, penalty_price_eur_per_vlsfo_t) * multiplier_y
        credit_val = 0.0
    else:
        credit_val = penalty_val = 0.0

    pool_applied.append(pool_use)
    bank_applied.append(bank_use)
    final_balance_t.append(final_bal)
    penalties_eur.append(penalty_val)
    credits_eur.append(credit_val)
    penalties_usd.append(penalty_val * eur_usd_fx)
    credits_usd.append(credit_val * eur_usd_fx)

    carry = carry_next

if info_provide_capped > 0:
    st.info(f"Pooling (provide < 0) capped vs pre-surplus in {info_provide_capped} year(s).")
if info_bank_capped > 0:
    st.info(f"Banking capped vs pre-surplus in {info_bank_capped} year(s).")
if info_bank_ignored_no_surplus > 0:
    st.info(f"Banking request ignored (no pre-surplus) in {info_bank_ignored_no_surplus} year(s).")
if info_final_safety_trim > 0:
    st.info(f"Final safety trim applied in {info_final_safety_trim} year(s) to avoid flipping surplus to deficit.")

# ──────────────────────────────────────────────────────────────────────────────
# BIO premium cost (USD) and Total_Cost_USD
# ──────────────────────────────────────────────────────────────────────────────
if "Extra-EU" in voyage_type:
    bio_mass_total_t_base = BIO_voy_t + BIO_berth_t
else:
    bio_mass_total_t_base = BIO_voy_t

bio_premium_cost_usd_base = bio_mass_total_t_base * bio_premium_usd_per_t
bio_premium_cost_usd_col = [bio_premium_cost_usd_base] * len(years)
total_cost_usd_col = [penalties_usd[i] + bio_premium_cost_usd_col[i] for i in range(len(years))]

# ──────────────────────────────────────────────────────────────────────────────
# Optimizer utilities (generic fuel → BIO, berth-first BIO placement for Extra-EU)
# ──────────────────────────────────────────────────────────────────────────────
def scoped_and_intensity_from_masses(h_v, l_v, m_v, b_v, r_v, h_b, l_b, m_b, b_b, r_b, elec_MJ, wtw_dict, year) -> Tuple[float,float,float]:
    energies_v = {
        "HSFO": compute_energy_MJ(h_v, LCV_HSFO),
        "LFO":  compute_energy_MJ(l_v, LCV_LFO),
        "MGO":  compute_energy_MJ(m_v, LCV_MGO),
        "BIO":  compute_energy_MJ(b_v, LCV_BIO),
        "RFNBO":compute_energy_MJ(r_v, LCV_RFNBO),
    }
    energies_b = {
        "HSFO": compute_energy_MJ(h_b, LCV_HSFO),
        "LFO":  compute_energy_MJ(l_b, LCV_LFO),
        "MGO":  compute_energy_MJ(m_b, LCV_MGO),
        "BIO":  compute_energy_MJ(b_b, LCV_BIO),
        "RFNBO":compute_energy_MJ(r_b, LCV_RFNBO),
    }
    if "Extra-EU" in voyage_type:
        scoped_x = scoped_energies_extra_eu(energies_v, energies_b, elec_MJ, wtw_dict)
    else:
        full = {k: energies_v.get(k,0.0) + energies_b.get(k,0.0) for k in ["HSFO","LFO","MGO","BIO","RFNBO"]}
        scoped_x = {**full, "ELEC": elec_MJ}

    E_scope_x = sum(scoped_x.values())
    num_phys_x = sum(scoped_x.get(k,0.0) * wtw_dict.get(k,0.0) for k in wtw_dict.keys())
    E_rfnbo_scope_x = scoped_x.get("RFNBO", 0.0)
    return E_scope_x, num_phys_x, E_rfnbo_scope_x

def penalty_usd_with_masses_for_year(year_idx: int,
                                     h_v, l_v, m_v, b_v, r_v,
                                     h_b, l_b, m_b, b_b, r_b) -> Tuple[float, float]:
    year = years[year_idx]
    g_target = limit_series[year_idx]
    E_scope_x, num_phys_x, E_rfnbo_scope_x = scoped_and_intensity_from_masses(
        h_v, l_v, m_v, b_v, r_v, h_b, l_b, m_b, b_b, r_b, ELEC_MJ, wtw, year
    )
    if E_scope_x <= 0:
        return 0.0, 0.0

    r = 2.0 if year <= 2033 else 1.0
    den_rwd_x = E_scope_x + (r - 1.0) * E_rfnbo_scope_x
    g_att_x = (num_phys_x / den_rwd_x) if den_rwd_x > 0 else 0.0

    CB_g_x = (g_target - g_att_x) * E_scope_x
    CB_t_raw_x = CB_g_x / 1e6
    cb_eff_x = CB_t_raw_x + carry_in_list[year_idx]

    if years[year_idx] >= int(pooling_start_year):
        if pooling_tco2e_input >= 0:
            pool_use_x = pooling_tco2e_input
        else:
            provide_abs = abs(pooling_tco2e_input)
            pre_surplus = max(cb_eff_x, 0.0)
            pool_use_x = -min(provide_abs, pre_surplus)
    else:
        pool_use_x = 0.0

    if years[year_idx] >= int(banking_start_year):
        pre_surplus = max(cb_eff_x, 0.0)
        requested_bank = max(banking_tco2e_input, 0.0)
        bank_use_x = min(requested_bank, pre_surplus)
    else:
        bank_use_x = 0.0

    final_bal_x = cb_eff_x + pool_use_x - bank_use_x
    if final_bal_x < 0:
        needed = -final_bal_x
        trim_bank = min(needed, bank_use_x)
        bank_use_x -= trim_bank
        needed -= trim_bank
        if needed > 0 and pool_use_x < 0:
            pool_use_x += needed
            needed = 0.0
        final_bal_x = cb_eff_x + pool_use_x - bank_use_x

    if final_bal_x < 0:
        penalty_eur_x = euros_from_tco2e(-final_bal_x, g_att_x, penalty_price_eur_per_vlsfo_t) * 1.0
        penalty_usd_x = penalty_eur_x * eur_usd_fx
    else:
        penalty_usd_x = 0.0

    return penalty_usd_x, g_att_x

def masses_after_shift_generic(fuel: str, x_decrease_t: float) -> Tuple[float,float,float,float,float,float,float,float,float,float]:
    """
    Reduce `fuel` mass by x (t) and add BIO mass scaled by LCV ratio to keep MJ constant.
    Extra-EU policy (as chosen): reduce selected on VOYAGE first, then BERTH; add BIO on BERTH first, then VOYAGE.
    Returns tuple in the same order as before:
      (h_v, l_v, m_v, b_v, r_v, h_b, l_b, m_b, b_b, r_b)
    """
    # Current masses (working copies)
    h_v, l_v, m_v, b_v, r_v = HSFO_voy_t, LFO_voy_t, MGO_voy_t, BIO_voy_t, RFNBO_voy_t
    h_b, l_b, m_b, b_b, r_b = HSFO_berth_t, LFO_berth_t, MGO_berth_t, BIO_berth_t, RFNBO_berth_t

    # Selected fuel references
    if fuel == "HSFO":
        s_v, s_b, LCV_S = h_v, h_b, LCV_HSFO
    elif fuel == "LFO":
        s_v, s_b, LCV_S = l_v, l_b, LCV_LFO
    else:  # MGO
        s_v, s_b, LCV_S = m_v, m_b, LCV_MGO

    # Bound decrease
    x = max(0.0, float(x_decrease_t))
    x = min(x, s_v + s_b)

    # BIO mass to add (energy neutrality)
    bio_increase_t = (x * LCV_S / LCV_BIO) if LCV_BIO > 0 else 0.0

    if "Extra-EU" in voyage_type:
        # Reduce selected fuel on VOYAGE first, then BERTH
        take_v = min(x, s_v)
        s_v -= take_v
        rem = x - take_v
        s_b = max(0.0, s_b - rem)

        # Add BIO on BERTH first, then VOYAGE
        add_b = min(bio_increase_t, float("inf"))
        b_b += add_b
        rem_bio = bio_increase_t - add_b
        if rem_bio > 0:
            b_v += rem_bio
    else:
        # Intra-EU: single bucket modeled via voyage vars
        s_v = max(0.0, s_v - x)
        b_v += bio_increase_t

    # Write-back
    if fuel == "HSFO":
        h_v, h_b = s_v, s_b
    elif fuel == "LFO":
        l_v, l_b = s_v, s_b
    else:
        m_v, m_b = s_v, s_b

    return h_v, l_v, m_v, b_v, r_v, h_b, l_b, m_b, b_b, r_b

# ──────────────────────────────────────────────────────────────────────────────
# Optimizer (generic fuel → BIO; berth-first BIO placement for Extra-EU)
# ──────────────────────────────────────────────────────────────────────────────
dec_opt_list, bio_inc_opt_list = [], []

for i in range(len(years)):
    # Total available tons of the selected fuel + its LCV
    if selected_fuel_for_opt == "HSFO":
        total_avail = HSFO_voy_t + HSFO_berth_t
        LCV_SEL = LCV_HSFO
    elif selected_fuel_for_opt == "LFO":
        total_avail = LFO_voy_t + LFO_berth_t
        LCV_SEL = LCV_LFO
    else:
        total_avail = MGO_voy_t + MGO_berth_t
        LCV_SEL = LCV_MGO

    if total_avail <= 0 or LCV_BIO <= 0:
        dec_opt_list.append(0.0)
        bio_inc_opt_list.append(0.0)
        continue

    steps_coarse = 60
    x_max = total_avail
    best_x, best_cost = 0.0, float("inf")

    # Coarse scan
    for s in range(steps_coarse + 1):
        x = x_max * s / steps_coarse
        masses = masses_after_shift_generic(selected_fuel_for_opt, x)
        penalty_usd_x, _ = penalty_usd_with_masses_for_year(i, *masses)

        # New total BIO tons under candidate x
        if "Extra-EU" in voyage_type:
            new_bio_total_t = (masses[3] + masses[8])  # b_v + b_b
        else:
            new_bio_total_t = masses[3]                # b_v
        total_cost_x = penalty_usd_x + new_bio_total_t * bio_premium_usd_per_t

        if total_cost_x < best_cost:
            best_cost, best_x = total_cost_x, x

    # Fine scan around best coarse x
    delta = x_max / steps_coarse * 2.0
    left = max(0.0, best_x - delta)
    right = min(x_max, best_x + delta)
    steps_fine = 80
    for s in range(steps_fine + 1):
        x = left + (right - left) * s / steps_fine
        masses = masses_after_shift_generic(selected_fuel_for_opt, x)
        penalty_usd_x, _ = penalty_usd_with_masses_for_year(i, *masses)
        if "Extra-EU" in voyage_type:
            new_bio_total_t = (masses[3] + masses[8])
        else:
            new_bio_total_t = masses[3]
        total_cost_x = penalty_usd_x + new_bio_total_t * bio_premium_usd_per_t
        if total_cost_x < best_cost:
            best_cost, best_x = total_cost_x, x

    dec_opt = best_x
    bio_inc_opt = dec_opt * (LCV_SEL / LCV_BIO) if LCV_BIO > 0 else 0.0
    dec_opt_list.append(dec_opt)
    bio_inc_opt_list.append(bio_inc_opt)

# ──────────────────────────────────────────────────────────────────────────────
# Table
# ──────────────────────────────────────────────────────────────────────────────
decrease_col_name = f"{selected_fuel_for_opt}_decrease(t)_for_Opt_Cost"

df_cost = pd.DataFrame(
    {
        "Year": years,
        "Reduction_%": LIMITS_DF["Reduction_%"].tolist(),
        "Limit_gCO2e_per_MJ": LIMITS_DF["Limit_gCO2e_per_MJ"].tolist(),
        "Actual_gCO2e_per_MJ": g_att_list,
        "Emissions_tCO2e": [emissions_tco2e]*len(years),
        "Compliance_Balance_tCO2e": cb_raw_t,
        "CarryIn_Banked_tCO2e": carry_in_list,
        "Effective_Balance_tCO2e": cb_eff_t,
        "Banked_to_Next_Year_tCO2e": bank_applied,
        "Pooling_tCO2e_Applied": pool_applied,
        "Final_Balance_tCO2e": final_balance_t,
        "Penalty_USD": penalties_usd,
        "Credit_USD": credits_usd,
        "BIO Premium Cost_USD": bio_premium_cost_usd_col,
        "Total_Cost_USD": total_cost_usd_col,
        decrease_col_name: dec_opt_list,
        "BIO_Increase(t)_For_Opt_Cost": bio_inc_opt_list,
    }
)

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
