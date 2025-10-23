# app.py — FuelEU Maritime Calculator (ABS-only, segments builder), 2025–2050
# Keeps your allocator/optimizer/pooling/banking intact. Sidebar has colored panels.

from __future__ import annotations

import json
import os
from typing import Dict, Any, Tuple, List

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

def _step_of_year(y: int) -> int:
    for i, (s, e, _) in enumerate(REDUCTION_STEPS):
        if s <= y <= e:
            return i
    return -1

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
    if "OPS_kWh" in DEFAULTS:
        try:
            return float(DEFAULTS["OPS_kWh"])
        except Exception:
            return 0.0
    if "OPS_MWh" in DEFAULTS:
        try:
            return float(DEFAULTS["OPS_MWh"]) * 1_000.0
        except Exception:
            return 0.0
    return 0.0

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def compute_energy_MJ(mass_t: float, lcv_MJ_per_t: float) -> float:
    mass_t = max(float(mass_t), 0.0)
    lcv = max(float(lcv_MJ_per_t), 0.0)
    return mass_t * lcv

def euros_from_tco2e(balance_tco2e_positive: float, g_attained: float, price_eur_per_vlsfo_t: float) -> float:
    if balance_tco2e_positive <= 0 or price_eur_per_vlsfo_t <= 0 or g_attained <= 0:
        return 0.0
    tco2e_per_vlsfot = (g_attained * 41_000.0) / 1_000_000.0
    if tco2e_per_vlsfot <= 0:
        return 0.0
    vlsfo_eq_t = balance_tco2e_positive / tco2e_per_vlsfot
    return vlsfo_eq_t * price_eur_per_vlsfo_t

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
# Allocator (unchanged Extra-EU logic)
# ──────────────────────────────────────────────────────────────────────────────
def scoped_energies_extra_eu(energies_fuel_voyage: Dict[str, float],
                             energies_fuel_berth: Dict[str, float],
                             elec_MJ: float,
                             wtw: Dict[str, float]) -> Dict[str, float]:
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

    for f in foss_sorted:
        amt = g(energies_fuel_berth, f)
        take = min(amt, remaining)
        if take > 0:
            scoped[f] += take
            remaining -= take
        if remaining <= 0:
            return scoped

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
# ABS-style segments
# ──────────────────────────────────────────────────────────────────────────────
SEG_TYPES = [
    "Intra-EU voyage",        # 100% scope
    "EU→non-EU voyage",       # Extra-EU (50% voyage scope)
    "non-EU→EU voyage",       # Extra-EU (50% voyage scope)
    "EU at-berth (port stay)" # 100% scope + OPS 100%
]

def _default_segment() -> Dict[str, Any]:
    return {
        "type": SEG_TYPES[0],
        "HSFO_t": 0.0, "LFO_t": 0.0, "MGO_t": 0.0, "BIO_t": 0.0, "RFNBO_t": 0.0,
        "OPS_kWh": 0.0
    }

def _ensure_segments_state():
    if "abs_segments" not in st.session_state:
        # Start from defaults if present
        if "abs_segments" in DEFAULTS and isinstance(DEFAULTS["abs_segments"], list):
            st.session_state["abs_segments"] = DEFAULTS["abs_segments"]
        else:
            st.session_state["abs_segments"] = []

def _segments_totals_masses() -> Dict[str, Dict[str, float]]:
    res = {
        "intra_voy": {k: 0.0 for k in ["HSFO","LFO","MGO","BIO","RFNBO"]},
        "extra_voy": {k: 0.0 for k in ["HSFO","LFO","MGO","BIO","RFNBO"]},
        "eu_berth":  {k: 0.0 for k in ["HSFO","LFO","MGO","BIO","RFNBO"]},
        "ops_kwh_total": 0.0
    }
    for seg in st.session_state.get("abs_segments", []):
        t = seg.get("type", SEG_TYPES[0])
        bucket = "intra_voy" if t == "Intra-EU voyage" else ("eu_berth" if t == "EU at-berth (port stay)" else "extra_voy")
        for f in ["HSFO","LFO","MGO","BIO","RFNBO"]:
            res[bucket][f] += float(seg.get(f + "_t", 0.0)) or 0.0
        res["ops_kwh_total"] += float(seg.get("OPS_kWh", 0.0)) or 0.0
    return res

def _masses_to_energies(masses: Dict[str, float], LCVs: Dict[str, float]) -> Dict[str, float]:
    return {f: compute_energy_MJ(masses.get(f, 0.0), LCVs.get(f, 0.0)) for f in ["HSFO","LFO","MGO","BIO","RFNBO"]}

# ──────────────────────────────────────────────────────────────────────────────
# Preview factor (first pass) — uses DEFAULTS + stored segments
# ──────────────────────────────────────────────────────────────────────────────
def _compute_preview_factor_first_pass_abs() -> float:
    # LCVs & WtWs from defaults only (first render)
    LCV_HSFO_pre = float(_get(DEFAULTS, "LCV_HSFO", 40200.0))
    LCV_LFO_pre  = float(_get(DEFAULTS, "LCV_LFO",  42700.0))
    LCV_MGO_pre  = float(_get(DEFAULTS, "LCV_MGO",  42700.0))
    LCV_BIO_pre  = float(_get(DEFAULTS, "LCV_BIO",  38000.0))
    LCV_RFN_pre  = float(_get(DEFAULTS, "LCV_RFNBO",30000.0))
    W_HSFO_pre = float(_get(DEFAULTS, "WtW_HSFO", 92.78))
    W_LFO_pre  = float(_get(DEFAULTS, "WtW_LFO",  92.00))
    W_MGO_pre  = float(_get(DEFAULTS, "WtW_MGO",  93.93))
    W_BIO_pre  = float(_get(DEFAULTS, "WtW_BIO",  70.00))
    W_RFN_pre  = float(_get(DEFAULTS, "WtW_RFNBO",20.00))
    OPS_kWh_pre = _get_ops_kwh_default()
    wtw_pre = {"HSFO": W_HSFO_pre, "LFO": W_LFO_pre, "MGO": W_MGO_pre, "BIO": W_BIO_pre, "RFNBO": W_RFN_pre, "ELEC": 0.0}
    LCVs_now = {"HSFO": LCV_HSFO_pre, "LFO": LCV_LFO_pre, "MGO": LCV_MGO_pre, "BIO": LCV_BIO_pre, "RFNBO": LCV_RFN_pre}

    segs = DEFAULTS.get("abs_segments", [])
    if not isinstance(segs, list) or len(segs) == 0:
        return 0.0

    totals_mass = {
        "intra_voy": {k:0.0 for k in ["HSFO","LFO","MGO","BIO","RFNBO"]},
        "extra_voy": {k:0.0 for k in ["HSFO","LFO","MGO","BIO","RFNBO"]},
        "eu_berth":  {k:0.0 for k in ["HSFO","LFO","MGO","BIO","RFNBO"]},
        "ops_kwh_total": 0.0
    }
    for seg in segs:
        t = seg.get("type", SEG_TYPES[0])
        bucket = "intra_voy" if t == "Intra-EU voyage" else ("eu_berth" if t == "EU at-berth (port stay)" else "extra_voy")
        for f in ["HSFO","LFO","MGO","BIO","RFNBO"]:
            totals_mass[bucket][f] += float(seg.get(f + "_t", 0.0)) or 0.0
        totals_mass["ops_kwh_total"] += float(seg.get("OPS_kWh", 0.0)) or 0.0

    energies_extra_voy = _masses_to_energies(totals_mass["extra_voy"], LCVs_now)
    energies_eu_berth  = _masses_to_energies(totals_mass["eu_berth"],  LCVs_now)
    energies_intra_voy = _masses_to_energies(totals_mass["intra_voy"], LCVs_now)
    OPS_MJ_pre = (totals_mass["ops_kwh_total"] or 0.0) * 3.6

    scoped_extra = scoped_energies_extra_eu(energies_extra_voy, energies_eu_berth, OPS_MJ_pre, wtw_pre)
    for f in ["HSFO","LFO","MGO","BIO","RFNBO"]:
        scoped_extra[f] = scoped_extra.get(f,0.0) + energies_intra_voy.get(f,0.0)

    num_pre = sum(scoped_extra.get(k,0.0) * wtw_pre.get(k,0.0) for k in wtw_pre.keys())
    den_pre = sum(scoped_extra.values())
    E_rfnbo_pre = scoped_extra.get("RFNBO", 0.0)
    den_pre_rwd = den_pre + E_rfnbo_pre
    g_preview = (num_pre / den_pre_rwd) if den_pre_rwd > 0 else 0.0
    return (g_preview * 41_000.0) / 1_000_000.0 if g_preview > 0 else 0.0

# First-pass factor for price linking
st.session_state["factor_vlsfo_per_tco2e"] = _compute_preview_factor_first_pass_abs()

# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="FuelEU Maritime Calculator — ABS Segments", layout="wide")

st.title("FuelEU Maritime — ABS-style Segments — GHG Intensity & Cost")
st.caption("2025–2050 • Limits from 2020 baseline 91.16 gCO₂e/MJ • WtW • Prices in EUR")

# Global CSS + Sidebar colored panels
st.markdown("""
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
.panel{ padding:.6rem .7rem; border-radius:.6rem; border:1px solid rgba(0,0,0,.08); }
.panel h4{ margin:.1rem 0 .4rem 0; font-size:1.0rem; font-weight:800; }
.panel small{ color:#374151; }
.p-green { background: #ecfdf5; border-left: 6px solid #10b981; }
.p-blue  { background: #eff6ff; border-left: 6px solid #3b82f6; }
.p-amber { background: #fffbeb; border-left: 6px solid #f59e0b; }
.p-purple{ background: #f5f3ff; border-left: 6px solid #8b5cf6; }
.p-gray  { background: #f9fafb; border-left: 6px solid #9ca3af; }
.badge{ display:inline-block; padding:.12rem .4rem; border-radius:9999px; font-size:.78rem; font-weight:700; }
.bg-green{ background:#10b981; color:white; }
.bg-blue { background:#3b82f6; color:white; }
.bg-amber{ background:#f59e0b; color:white; }
.bg-purple{ background:#8b5cf6; color:white; }
.panel .subtle{ font-size:.86rem; color:#6b7280; margin-top:.1rem; }
[data-testid="stDataFrame"] div[role="columnheader"],[data-testid="stDataFrame"] div[role="gridcell"]{ padding:2px 6px !important; }
[data-testid="stDataFrame"] { font-size: 0.85rem !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar (ABS-only)
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    _ensure_segments_state()

    # Exchange rate
    st.markdown('<div class="panel p-gray"><h4>Exchange rate</h4></div>', unsafe_allow_html=True)
    eur_usd_fx = float_text_input("1 Euro = … USD", _get(DEFAULTS, "eur_usd_fx", 1.00), key="eur_usd_fx", min_value=0.0)

    # Credits
    st.markdown('<div class="panel p-green"><h4>Compliance Market — Credits</h4><div class="subtle">Edit €/tCO₂e; €/VLSFO-eq t is derived.</div></div>', unsafe_allow_html=True)
    credit_per_tco2e = float_text_input("Credit price €/tCO₂e", _get(DEFAULTS, "credit_per_tco2e", 200.0), key="credit_per_tco2e_str", min_value=0.0)
    factor_vlsfo_per_tco2e_prev = float(st.session_state.get("factor_vlsfo_per_tco2e", 0.0))
    credit_price_eur_per_vlsfo_t = credit_per_tco2e * factor_vlsfo_per_tco2e_prev if factor_vlsfo_per_tco2e_prev > 0 else 0.0
    st.text_input("Credit price €/VLSFO-eq t (derived)", value=us2(credit_price_eur_per_vlsfo_t), disabled=True)

    # Penalties
    st.markdown('<div class="panel p-amber"><h4>Compliance Market — Penalties</h4><div class="subtle">Regulated default. Change only if regulation changes.</div></div>', unsafe_allow_html=True)
    penalty_price_eur_per_vlsfo_t = float_text_input("Penalty price €/VLSFO-eq t", _get(DEFAULTS, "penalty_price_eur_per_vlsfo_t", 2_400.0), key="penalty_per_vlsfo_t_str", min_value=0.0)
    penalty_price_eur_per_tco2e_prev = (penalty_price_eur_per_vlsfo_t / factor_vlsfo_per_tco2e_prev) if factor_vlsfo_per_tco2e_prev > 0 else 0.0
    st.text_input("Penalty price €/tCO₂e (derived)", value=us2(penalty_price_eur_per_tco2e_prev), disabled=True)

    # BIO premium
    st.markdown('<div class="panel p-blue"><h4>Premium BIO vs HSFO</h4><div class="subtle">USD per ton of BIO (fuel cost premium).</div></div>', unsafe_allow_html=True)
    bio_premium_usd_per_t = float_text_input("", _get(DEFAULTS, "bio_premium_usd_per_t", 0.0), key="bio_premium_usd_per_t", min_value=0.0, label_visibility="collapsed")

    st.divider()

    # Segments builder panel
    st.markdown('<div class="panel p-purple"><h4>ABS segments</h4><div class="subtle">Add voyages and EU at-berth stays one by one. All totals aggregate to the common dashboard.</div></div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("➕ Add segment"):
            st.session_state["abs_segments"].append(_default_segment())
    with c2:
        if st.button("🗑️ Clear all"):
            st.session_state["abs_segments"] = []

    # Render segments
    to_remove: List[int] = []
    for i, seg in enumerate(st.session_state["abs_segments"]):
        # colored badge
        t = seg.get("type", SEG_TYPES[0])
        badge = '<span class="badge bg-green">Intra</span>' if t=="Intra-EU voyage" else ('<span class="badge bg-blue">Cross-border</span>' if "voyage" in t else '<span class="badge bg-purple">EU berth</span>')
        with st.expander(f"Segment {i+1} {badge}", expanded=True):
            seg["type"] = st.selectbox("Segment type", SEG_TYPES, index=SEG_TYPES.index(t), key=f"seg_type_{i}")
            cA, cB = st.columns(2)
            with cA:
                seg["HSFO_t"]  = float_text_input("HSFO [t]" , seg.get("HSFO_t", 0.0), key=f"seg_hsfo_{i}",  min_value=0.0)
                seg["MGO_t"]   = float_text_input("MGO [t]"  , seg.get("MGO_t",  0.0), key=f"seg_mgo_{i}",   min_value=0.0)
                seg["RFNBO_t"] = float_text_input("RFNBO [t]", seg.get("RFNBO_t",0.0), key=f"seg_rfn_{i}",   min_value=0.0)
            with cB:
                seg["LFO_t"]   = float_text_input("LFO [t]"  , seg.get("LFO_t",  0.0), key=f"seg_lfo_{i}",   min_value=0.0)
                seg["BIO_t"]   = float_text_input("BIO [t]"  , seg.get("BIO_t",  0.0), key=f"seg_bio_{i}",   min_value=0.0)
                seg["OPS_kWh"] = float_text_input("OPS electricity (kWh)", seg.get("OPS_kWh", 0.0), key=f"seg_ops_{i}", min_value=0.0)
            if st.button("Remove this segment", key=f"seg_remove_{i}"):
                to_remove.append(i)
    if to_remove:
        st.session_state["abs_segments"] = [s for j,s in enumerate(st.session_state["abs_segments"]) if j not in to_remove]

    st.divider()

    # LCVs
    st.markdown('<div class="panel p-gray"><h4>LCVs [MJ/ton]</h4></div>', unsafe_allow_html=True)
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

    # WtWs
    st.markdown('<div class="panel p-gray"><h4>WtW intensities [gCO₂e/MJ]</h4></div>', unsafe_allow_html=True)
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

    # OPS (derived from segments)
    totals_mass_preview = _segments_totals_masses()
    OPS_kWh_total = totals_mass_preview["ops_kwh_total"]
    OPS_MJ = OPS_kWh_total * 3.6
    st.text_input("Total OPS electricity (kWh) (derived from segments)", value=us2(OPS_kWh_total), disabled=True)
    st.text_input("Electricity delivered (MJ) (derived)", value=us2(OPS_MJ), disabled=True)

    # Update preview factor with current values (used on next rerun)
    LCVs_now = {"HSFO": LCV_HSFO, "LFO": LCV_LFO, "MGO": LCV_MGO, "BIO": LCV_BIO, "RFNBO": LCV_RFNBO}
    wtw_preview = {"HSFO": WtW_HSFO, "LFO": WtW_LFO, "MGO": WtW_MGO, "BIO": WtW_BIO, "RFNBO": WtW_RFNBO, "ELEC": 0.0}
    energies_extra_voy_prev = _masses_to_energies(totals_mass_preview["extra_voy"], LCVs_now)
    energies_eu_berth_prev  = _masses_to_energies(totals_mass_preview["eu_berth"],  LCVs_now)
    energies_intra_voy_prev = _masses_to_energies(totals_mass_preview["intra_voy"], LCVs_now)
    scoped_prev = scoped_energies_extra_eu(energies_extra_voy_prev, energies_eu_berth_prev, OPS_MJ, wtw_preview)
    for f in ["HSFO","LFO","MGO","BIO","RFNBO"]:
        scoped_prev[f] = scoped_prev.get(f,0.0) + energies_intra_voy_prev.get(f,0.0)
    num_prev = sum(scoped_prev.get(k,0.0) * wtw_preview.get(k,0.0) for k in wtw_preview.keys())
    den_prev = sum(scoped_prev.values())
    E_rfnbo_prev = scoped_prev.get("RFNBO", 0.0)
    den_prev_rwd = den_prev + E_rfnbo_prev
    g_actual_preview = (num_prev / den_prev_rwd) if den_prev_rwd > 0 else 0.0
    st.session_state["factor_vlsfo_per_tco2e"] = (g_actual_preview * 41_000.0) / 1_000_000.0 if g_actual_preview > 0 else 0.0

    # Other
    st.markdown('<div class="panel p-gray"><h4>Other</h4></div>', unsafe_allow_html=True)
    consecutive_deficit_years_seed = int(st.number_input("Consecutive deficit years (seed)", min_value=1, value=int(_get(DEFAULTS, "consecutive_deficit_years", 1)), step=1))

    # Optimizer target fuel
    opt_fuels = ["HSFO", "LFO", "MGO"]
    _saved_opt_fuel = _get(DEFAULTS, "opt_reduce_fuel", "HSFO")
    try:
        _idx = opt_fuels.index(_saved_opt_fuel)
    except ValueError:
        _idx = 0
    selected_fuel_for_opt = st.selectbox("Fuel to reduce (for optimization)", opt_fuels, index=_idx)

    # Banking & Pooling
    st.markdown('<div class="panel p-amber"><h4>Banking & Pooling (tCO₂e)</h4><div class="subtle">Independent & capped vs pre-surplus (no flip to deficit). Banking creates carry-in.</div></div>', unsafe_allow_html=True)
    pooling_price_eur_per_tco2e = float_text_input("Pooling price €/tCO₂e", _get(DEFAULTS, "pooling_price_eur_per_tco2e", 200.0), key="pooling_price_eur_per_tco2e", min_value=0.0)
    pooling_tco2e_input = float_text_input_signed("Pooling [tCO₂e]: + uptake, − provide", _get(DEFAULTS, "pooling_tco2e", 0.0), key="POOL_T")
    pooling_start_year = st.selectbox("Pooling starts from year", YEARS, index=YEARS.index(int(_get(DEFAULTS, "pooling_start_year", YEARS[0]))))
    banking_tco2e_input = float_text_input("Banking to next year [tCO₂e]", _get(DEFAULTS, "banking_tco2e", 0.0), key="BANK_T", min_value=0.0)
    banking_start_year = st.selectbox("Banking starts from year", YEARS, index=YEARS.index(int(_get(DEFAULTS, "banking_start_year", YEARS[0]))))

    # Save defaults (including segments)
    if st.button("💾 Save current inputs as defaults"):
        defaults_to_save = {
            "voyage_type": "ABS",
            "eur_usd_fx": eur_usd_fx,
            "bio_premium_usd_per_t": bio_premium_usd_per_t,
            "LCV_HSFO": LCV_HSFO, "LCV_LFO": LCV_LFO, "LCV_MGO": LCV_MGO, "LCV_BIO": LCV_BIO, "LCV_RFNBO": LCV_RFNBO,
            "WtW_HSFO": WtW_HSFO, "WtW_LFO": WtW_LFO, "WtW_MGO": WtW_MGO, "WtW_BIO": WtW_BIO, "WtW_RFNBO": WtW_RFNBO,
            "credit_per_tco2e": credit_per_tco2e,
            "penalty_price_eur_per_vlsfo_t": penalty_price_eur_per_vlsfo_t,
            "consecutive_deficit_years": consecutive_deficit_years_seed,
            "pooling_price_eur_per_tco2e": pooling_price_eur_per_tco2e,
            "banking_tco2e": banking_tco2e_input,
            "pooling_tco2e": pooling_tco2e_input,
            "pooling_start_year": int(pooling_start_year),
            "banking_start_year": int(banking_start_year),
            "abs_segments": st.session_state.get("abs_segments", []),
        }
        try:
            with open(DEFAULTS_PATH, "w", encoding="utf-8") as f:
                json.dump(defaults_to_save, f, indent=2)
            st.success("Defaults saved.")
        except Exception as e:
            st.error(f"Could not save defaults: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# Derived energies & intensity (ABS aggregation)
# ──────────────────────────────────────────────────────────────────────────────
# Mass totals
totals_mass = _segments_totals_masses()

# Publish classic-like variables for downstream (optimizer etc.)
HSFO_voy_t = totals_mass["intra_voy"]["HSFO"] + totals_mass["extra_voy"]["HSFO"]
LFO_voy_t  = totals_mass["intra_voy"]["LFO"]  + totals_mass["extra_voy"]["LFO"]
MGO_voy_t  = totals_mass["intra_voy"]["MGO"]  + totals_mass["extra_voy"]["MGO"]
BIO_voy_t  = totals_mass["intra_voy"]["BIO"]  + totals_mass["extra_voy"]["BIO"]
RFNBO_voy_t= totals_mass["intra_voy"]["RFNBO"]+ totals_mass["extra_voy"]["RFNBO"]
HSFO_berth_t = totals_mass["eu_berth"]["HSFO"]
LFO_berth_t  = totals_mass["eu_berth"]["LFO"]
MGO_berth_t  = totals_mass["eu_berth"]["MGO"]
BIO_berth_t  = totals_mass["eu_berth"]["BIO"]
RFNBO_berth_t= totals_mass["eu_berth"]["RFNBO"]
ELEC_MJ = (totals_mass["ops_kwh_total"] or 0.0) * 3.6

# LCVs/WtWs
LCV_HSFO = float(_get(DEFAULTS, "LCV_HSFO", 40200.0)) if "LCV_HSFO" not in st.session_state else parse_us_any(st.session_state["LCV_HSFO"], 40200.0)
LCV_LFO  = float(_get(DEFAULTS, "LCV_LFO",  42700.0)) if "LCV_LFO"  not in st.session_state else parse_us_any(st.session_state["LCV_LFO"],  42700.0)
LCV_MGO  = float(_get(DEFAULTS, "LCV_MGO",  42700.0)) if "LCV_MGO"  not in st.session_state else parse_us_any(st.session_state["LCV_MGO"],  42700.0)
LCV_BIO  = float(_get(DEFAULTS, "LCV_BIO",  38000.0)) if "LCV_BIO"  not in st.session_state else parse_us_any(st.session_state["LCV_BIO"],  38000.0)
LCV_RFNBO= float(_get(DEFAULTS, "LCV_RFNBO",30000.0)) if "LCV_RFNBO"not in st.session_state else parse_us_any(st.session_state["LCV_RFNBO"],30000.0)

WtW_HSFO = float(_get(DEFAULTS, "WtW_HSFO", 92.78)) if "WtW_HSFO" not in st.session_state else parse_us_any(st.session_state["WtW_HSFO"], 92.78)
WtW_LFO  = float(_get(DEFAULTS, "WtW_LFO",  92.00)) if "WtW_LFO"  not in st.session_state else parse_us_any(st.session_state["WtW_LFO"],  92.00)
WtW_MGO  = float(_get(DEFAULTS, "WtW_MGO",  93.93)) if "WtW_MGO"  not in st.session_state else parse_us_any(st.session_state["WtW_MGO"],  93.93)
WtW_BIO  = float(_get(DEFAULTS, "WtW_BIO",  70.00)) if "WtW_BIO"  not in st.session_state else parse_us_any(st.session_state["WtW_BIO"],  70.00)
WtW_RFNBO= float(_get(DEFAULTS, "WtW_RFNBO",20.00)) if "WtW_RFNBO"not in st.session_state else parse_us_any(st.session_state["WtW_RFNBO"],20.00)
wtw = {"HSFO": WtW_HSFO, "LFO": WtW_LFO, "MGO": WtW_MGO, "BIO": WtW_BIO, "RFNBO": WtW_RFNBO, "ELEC": 0.0}
LCVs_now = {"HSFO": LCV_HSFO, "LFO": LCV_LFO, "MGO": LCV_MGO, "BIO": LCV_BIO, "RFNBO": LCV_RFNBO}

energies_extra_voy = _masses_to_energies(totals_mass["extra_voy"], LCVs_now)
energies_eu_berth  = _masses_to_energies(totals_mass["eu_berth"],  LCVs_now)
energies_intra_voy = _masses_to_energies(totals_mass["intra_voy"], LCVs_now)

scoped_extra = scoped_energies_extra_eu(energies_extra_voy, energies_eu_berth, ELEC_MJ, wtw)
scoped_energies = dict(scoped_extra)
for f in ["HSFO","LFO","MGO","BIO","RFNBO"]:
    scoped_energies[f] = scoped_energies.get(f,0.0) + energies_intra_voy.get(f,0.0)

energies_fuel_full = {f: energies_extra_voy.get(f,0.0) + energies_eu_berth.get(f,0.0) + energies_intra_voy.get(f,0.0)
                      for f in ["HSFO","LFO","MGO","BIO","RFNBO"]}
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
# Headline metrics
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
# Bars: all vs in-scope
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<h2 style="margin:0 0 .25rem 0;">Energy composition (all vs in-scope) </h2>', unsafe_allow_html=True)

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

    fig_stacks.add_trace(go.Scatter(x=categories, y=[y_center_left, y_center_right], mode="lines",
                    line=dict(dash="dot", width=2), hoverinfo="skip", showlegend=False))
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
st.caption("ABS mode: Intra-EU voyage energy is 100% in scope; cross-border voyages + EU at-berth follow the Extra-EU allocator (renewables first, then at-berth fossils, then 50% of voyage fossils by ascending WtW). ELEC is always 100%.")

# ──────────────────────────────────────────────────────────────────────────────
# Plot — GHG Intensity vs Limit
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

# ──────────────────────────────────────────────────────────────────────────────
# Results — Banking/Pooling + per-step multiplier
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

carry = 0.0
fixed_multiplier_by_step = {}

eur_usd_fx = float(parse_us_any(st.session_state.get("eur_usd_fx", _get(DEFAULTS,"eur_usd_fx",1.0)), 1.0))
credit_per_tco2e = float(parse_us_any(st.session_state.get("credit_per_tco2e_str", _get(DEFAULTS,"credit_per_tco2e",200.0)), 200.0))
penalty_price_eur_per_vlsfo_t = float(parse_us_any(st.session_state.get("penalty_per_vlsfo_t_str", _get(DEFAULTS,"penalty_price_eur_per_vlsfo_t",2400.0)), 2400.0))
pooling_price_eur_per_tco2e = float(parse_us_any(st.session_state.get("pooling_price_eur_per_tco2e", _get(DEFAULTS,"pooling_price_eur_per_tco2e",200.0)), 200.0))
pooling_tco2e_input = float(parse_us_any(st.session_state.get("POOL_T", _get(DEFAULTS,"pooling_tco2e",0.0)), 0.0))
banking_tco2e_input = float(parse_us_any(st.session_state.get("BANK_T", _get(DEFAULTS,"banking_tco2e",0.0)), 0.0))
pooling_start_year = int(st.session_state.get("pooling_start_year", _get(DEFAULTS,"pooling_start_year", YEARS[0])))
banking_start_year = int(st.session_state.get("banking_start_year", _get(DEFAULTS,"banking_start_year", YEARS[0])))
consecutive_deficit_years_seed = int(st.session_state.get("Consecutive deficit years (seed)", _get(DEFAULTS,"consecutive_deficit_years",1)))

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

    # Final clamp
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

    # Per-step penalty multiplier (fixed within each step)
    if final_bal < 0:
        step_idx = _step_of_year(year)
        if step_idx not in fixed_multiplier_by_step:
            start_count = max(int(consecutive_deficit_years_seed), 1)
            fixed_multiplier_by_step[step_idx] = 1.0 + (start_count - 1) * 0.10
        multiplier_y = fixed_multiplier_by_step[step_idx]
    else:
        multiplier_y = 1.0

    # € → USD
    if final_bal > 0:
        factor = float(st.session_state.get("factor_vlsfo_per_tco2e", 0.0))
        credit_val = euros_from_tco2e(final_bal, g_att, (credit_per_tco2e * factor if factor > 0 else 0.0))
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

# BIO premium & pooling cost
bio_mass_total_t_base = totals_mass["intra_voy"]["BIO"] + totals_mass["extra_voy"]["BIO"] + totals_mass["eu_berth"]["BIO"]
bio_premium_usd_per_t = float(parse_us_any(st.session_state.get("bio_premium_usd_per_t", _get(DEFAULTS,"bio_premium_usd_per_t",0.0)), 0.0))
bio_premium_cost_usd_base = bio_mass_total_t_base * bio_premium_usd_per_t
bio_premium_cost_usd_col = [bio_premium_cost_usd_base] * len(years)

pooling_cost_usd_col = [pool_applied[i] * pooling_price_eur_per_tco2e * eur_usd_fx for i in range(len(years))]
total_cost_usd_col = [penalties_usd[i] + bio_premium_cost_usd_col[i] + pooling_cost_usd_col[i] for i in range(len(years))]

# ──────────────────────────────────────────────────────────────────────────────
# Optimizer (generic fuel → BIO; berth-first BIO placement)
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
    scoped_x = scoped_energies_extra_eu(energies_v, energies_b, elec_MJ, wtw_dict)
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
        step_idx = _step_of_year(year)
        start_count = max(int(consecutive_deficit_years_seed), 1)
        step_mult = 1.0 + (start_count - 1) * 0.10
        penalty_eur_x = euros_from_tco2e(-final_bal_x, g_att_x, penalty_price_eur_per_vlsfo_t) * step_mult
        penalty_usd_x = penalty_eur_x * eur_usd_fx
    else:
        penalty_usd_x = 0.0

    return penalty_usd_x, g_att_x

def masses_after_shift_generic(fuel: str, x_decrease_t: float) -> Tuple[float,float,float,float,float,float,float,float,float,float]:
    # working copies: voyage = intra+extra; berth = EU berth
    h_v, l_v, m_v, b_v, r_v = HSFO_voy_t, LFO_voy_t, MGO_voy_t, BIO_voy_t, RFNBO_voy_t
    h_b, l_b, m_b, b_b, r_b = HSFO_berth_t, LFO_berth_t, MGO_berth_t, BIO_berth_t, RFNBO_berth_t

    if fuel == "HSFO":
        s_v, s_b, LCV_S = h_v, h_b, LCV_HSFO
    elif fuel == "LFO":
        s_v, s_b, LCV_S = l_v, l_b, LCV_LFO
    else:
        s_v, s_b, LCV_S = m_v, m_b, LCV_MGO

    x = max(0.0, float(x_decrease_t))
    x = min(x, s_v + s_b)

    bio_increase_t = (x * LCV_S / LCV_BIO) if LCV_BIO > 0 else 0.0

    take_v = min(x, s_v); s_v -= take_v
    rem = x - take_v; s_b = max(0.0, s_b - rem)

    add_b = min(bio_increase_t, float("inf"))
    b_b += add_b
    rem_bio = bio_increase_t - add_b
    if rem_bio > 0:
        b_v += rem_bio

    if fuel == "HSFO":
        h_v, h_b = s_v, s_b
    elif fuel == "LFO":
        l_v, l_b = s_v, s_b
    else:
        m_v, m_b = s_v, s_b

    return h_v, l_v, m_v, b_v, r_v, h_b, l_b, m_b, b_b, r_b

# Optimizer scan
selected_fuel_for_opt = st.session_state.get("Fuel to reduce (for optimization)", _get(DEFAULTS,"opt_reduce_fuel","HSFO"))
dec_opt_list, bio_inc_opt_list = [], []
for i in range(len(years)):
    if selected_fuel_for_opt == "HSFO":
        total_avail, LCV_SEL = HSFO_voy_t + HSFO_berth_t, LCV_HSFO
    elif selected_fuel_for_opt == "LFO":
        total_avail, LCV_SEL = LFO_voy_t + LFO_berth_t, LCV_LFO
    else:
        total_avail, LCV_SEL = MGO_voy_t + MGO_berth_t, LCV_MGO

    if total_avail <= 0 or LCV_BIO <= 0:
        dec_opt_list.append(0.0); bio_inc_opt_list.append(0.0); continue

    steps_coarse = 60
    x_max = total_avail
    best_x, best_cost = 0.0, float("inf")

    for s in range(steps_coarse + 1):
        x = x_max * s / steps_coarse
        masses = masses_after_shift_generic(selected_fuel_for_opt, x)
        penalty_usd_x, _ = penalty_usd_with_masses_for_year(i, *masses)
        new_bio_total_t = (masses[3] + masses[8])  # b_v + b_b
        total_cost_x = penalty_usd_x + new_bio_total_t * bio_premium_usd_per_t
        if total_cost_x < best_cost:
            best_cost, best_x = total_cost_x, x

    delta = x_max / steps_coarse * 2.0
    left = max(0.0, best_x - delta); right = min(x_max, best_x + delta)
    steps_fine = 80
    for s in range(steps_fine + 1):
        x = left + (right - left) * s / steps_fine
        masses = masses_after_shift_generic(selected_fuel_for_opt, x)
        penalty_usd_x, _ = penalty_usd_with_masses_for_year(i, *masses)
        new_bio_total_t = (masses[3] + masses[8])
        total_cost_x = penalty_usd_x + new_bio_total_t * bio_premium_usd_per_t
        if total_cost_x < best_cost:
            best_cost, best_x = total_cost_x, x

    dec_opt = best_x
    bio_inc_opt = dec_opt * (LCV_SEL / LCV_BIO) if LCV_BIO > 0 else 0.0
    dec_opt_list.append(dec_opt); bio_inc_opt_list.append(bio_inc_opt)

# Recompute optimized costs
penalty_usd_opt_col, bio_premium_cost_usd_opt_col, total_cost_usd_opt_col = [], [], []
for i in range(len(years)):
    x_opt = dec_opt_list[i]
    if x_opt <= 0.0 or LCV_BIO <= 0.0:
        penalty_usd_opt = penalties_usd[i]
        bio_premium_usd_opt = bio_premium_cost_usd_col[i]
    else:
        masses_opt = masses_after_shift_generic(selected_fuel_for_opt, x_opt)
        penalty_usd_opt, _ = penalty_usd_with_masses_for_year(i, *masses_opt)
        new_bio_total_t_opt = (masses_opt[3] + masses_opt[8])
        bio_premium_usd_opt = new_bio_total_t_opt * bio_premium_usd_per_t
    penalty_usd_opt_col.append(penalty_usd_opt)
    bio_premium_cost_usd_opt_col.append(bio_premium_usd_opt)
    total_cost_usd_opt_col.append(penalty_usd_opt + bio_premium_usd_opt + pooling_cost_usd_col[i])

# ──────────────────────────────────────────────────────────────────────────────
# Table
# ──────────────────────────────────────────────────────────────────────────────
decrease_col_name = f"{selected_fuel_for_opt}_decrease(t)_for_Opt_Cost"

df_cost = pd.DataFrame(
    {
        "Year": years,
        "Reduction_%": LIMITS_DF["Reduction_%"].tolist(),
        "Limit_gCO2e_per_MJ": LIMITS_DF["Limit_gCO2e_per_MJ"].tolist(),
        "Actual_gCO2e_per_MJ": [attained_intensity_for_year(y) for y in years],
        "Emissions_tCO2e": [ (g_base * E_scope_MJ) / 1e6 ]*len(years),
        "Compliance_Balance_tCO2e": cb_raw_t,
        "CarryIn_Banked_tCO2e": carry_in_list,
        "Effective_Balance_tCO2e": cb_eff_t,
        "Banked_to_Next_Year_tCO2e": bank_applied,
        "Pooling_tCO2e_Applied": pool_applied,
        "Final_Balance_tCO2e": final_balance_t,
        "Pooling_Cost_USD": pooling_cost_usd_col,
        "Penalty_USD": penalties_usd,
        "Credit_USD": credits_usd,
        "BIO Premium Cost_USD": bio_premium_cost_usd_col,
        "Total_Cost_USD": total_cost_usd_col,
        decrease_col_name: dec_opt_list,
        "BIO_Increase(t)_For_Opt_Cost": bio_inc_opt_list,
        "Total_Cost_USD_Opt": total_cost_usd_opt_col,
    }
)

df_fmt = df_cost.copy()
for col in df_fmt.columns:
    if col != "Year":
        df_fmt[col] = df_fmt[col].apply(us2)

st.dataframe(df_fmt, use_container_width=True)
st.download_button("Download per-year results (CSV)", data=df_fmt.to_csv(index=False), file_name="fueleu_results_2025_2050.csv", mime="text/csv")
