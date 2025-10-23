# app.py — FuelEU Maritime Calculator (ABS segments only, tidy sidebar)
# Keeps your allocator/optimizer/pooling/banking logic. OPS is a single global input (kWh→MJ).

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

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def compute_energy_MJ(mass_t: float, lcv_MJ_per_t: float) -> float:
    mass_t = max(float(mass_t), 0.0)
    lcv = max(float(lcv_MJ_per_t), 0.0)
    return mass_t * lcv

def euros_from_tco2e(balance_tco2e_positive: float, g_attained: float, price_eur_per_vlsfo_t: float) -> float:
    """
    Convert a *positive* tCO2e quantity to euros using the year's attained GHG.
    tCO2e per VLSFO-eq ton = (g_attained [g/MJ] * 41,000 [MJ/t]) / 1e6.
    """
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
# Allocator (Extra-EU pool priority, unchanged)
# ──────────────────────────────────────────────────────────────────────────────
def scoped_energies_extra_eu(energies_fuel_voyage: Dict[str, float],
                             energies_fuel_berth: Dict[str, float],
                             elec_MJ: float,
                             wtw: Dict[str, float]) -> Dict[str, float]:
    """
    Extra-EU (requested variant, berth-100% guarantee):
      • Build pool: ELEC (100%) + at-berth fuels (100%) + 50% of *total* voyage fuels.
      • Fill by WtW priority (renewables first, then at-berth fossils, then 50% voyage fossils).
    """
    def g(d, k): return float(d.get(k, 0.0))

    fossils = ["HSFO", "LFO", "MGO"]
    foss_sorted = sorted(fossils, key=lambda f: wtw.get(f, float("inf")))

    total_voy = sum(energies_fuel_voyage.values())
    half_voy  = 0.5 * total_voy
    berth_fossil_total = sum(g(energies_fuel_berth, f) for f in fossils)
    pool_total = sum(energies_fuel_berth.values()) + half_voy

    scoped = {k: 0.0 for k in ["HSFO","LFO","MGO","BIO","RFNBO","ELEC"]}
    scoped["ELEC"] = max(elec_MJ, 0.0)

    remaining = pool_total  # fuels only

    # 1) Renewables: at-berth first, then voyage up to spare after reserving berth fossils
    ren_sorted = sorted(["RFNBO","BIO"], key=lambda f: wtw.get(f, float("inf")))
    for f in ren_sorted:
        take_b = min(g(energies_fuel_berth, f), remaining)
        if take_b > 0:
            scoped[f] += take_b; remaining -= take_b
        if remaining <= 0: return scoped

        spare_for_voy_ren = max(0.0, remaining - berth_fossil_total)
        take_v = min(g(energies_fuel_voyage, f), spare_for_voy_ren)
        if take_v > 0:
            scoped[f] += take_v; remaining -= take_v
        if remaining <= 0: return scoped

    # 2) Fossil at-berth — 100% in ascending WtW
    for f in foss_sorted:
        take = min(g(energies_fuel_berth, f), remaining)
        if take > 0:
            scoped[f] += take; remaining -= take
        if remaining <= 0: return scoped

    # 3) Fossil voyage — 50% per fuel in ascending WtW
    for f in foss_sorted:
        half_v = 0.5 * g(energies_fuel_voyage, f)
        if half_v <= 0 or remaining <= 0: continue
        take = min(half_v, remaining)
        scoped[f] += take; remaining -= take
        if remaining <= 0: return scoped

    return scoped

# ──────────────────────────────────────────────────────────────────────────────
# ABS-style segments
# ──────────────────────────────────────────────────────────────────────────────
SEG_TYPES = [
    "Intra-EU voyage",        # 100% scope
    "EU→non-EU voyage",       # Extra-EU (50% voyage scope in pool)
    "non-EU→EU voyage",       # Extra-EU (50% voyage scope in pool)
    "EU at-berth (port stay)" # 100% scope + contributes to pool
]

def _default_segment() -> Dict[str, Any]:
    return {"type": SEG_TYPES[0], "HSFO_t": 0.0, "LFO_t": 0.0, "MGO_t": 0.0, "BIO_t": 0.0, "RFNBO_t": 0.0}

def _ensure_segments_state():
    if "abs_segments" not in st.session_state:
        if "abs_segments" in DEFAULTS and isinstance(DEFAULTS["abs_segments"], list):
            st.session_state["abs_segments"] = DEFAULTS["abs_segments"]
        else:
            st.session_state["abs_segments"] = []

def _segments_totals_masses() -> Dict[str, Dict[str, float]]:
    res = {
        "intra_voy": {k: 0.0 for k in ["HSFO","LFO","MGO","BIO","RFNBO"]},
        "extra_voy": {k: 0.0 for k in ["HSFO","LFO","MGO","BIO","RFNBO"]},
        "eu_berth":  {k: 0.0 for k in ["HSFO","LFO","MGO","BIO","RFNBO"]},
    }
    for seg in st.session_state.get("abs_segments", []):
        t = seg.get("type", SEG_TYPES[0])
        bucket = "intra_voy" if t == "Intra-EU voyage" else ("eu_berth" if t == "EU at-berth (port stay)" else "extra_voy")
        for f in ["HSFO","LFO","MGO","BIO","RFNBO"]:
            res[bucket][f] += float(seg.get(f + "_t", 0.0)) or 0.0
    return res

def _masses_to_energies(masses: Dict[str, float], LCVs: Dict[str, float]) -> Dict[str, float]:
    return {f: compute_energy_MJ(masses.get(f, 0.0), LCVs.get(f, 0.0)) for f in ["HSFO","LFO","MGO","BIO","RFNBO"]}

# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="FuelEU Maritime — ABS Segments", layout="wide")
st.title("FuelEU Maritime — ABS Segments — GHG Intensity & Cost")
st.caption("2025–2050 • Limits from 2020 baseline 91.16 gCO₂e/MJ • WtW • Prices in EUR")

# Sidebar layout — compact “cards”, clear order, no colors
st.markdown("""
<style>
/* Compact, consistent sidebar spacing */
section[data-testid="stSidebar"] div.block-container{ padding-top:.6rem; padding-bottom:.6rem; }
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{ gap:.6rem; }
section[data-testid="stSidebar"] label{ font-size:.95rem; margin-bottom:.2rem; font-weight:600; }
section[data-testid="stSidebar"] input[type="text"],
section[data-testid="stSidebar"] input[type="number"]{ height:2.0rem; min-height:2.0rem; padding:.32rem .55rem; }
.card{ padding:.65rem .75rem; border:1px solid #e5e7eb; border-radius:.6rem; background:#fbfbfb; }
.card h4{ margin:.15rem 0 .4rem 0; font-size:1.0rem; font-weight:800; }
.card .help{ font-size:.86rem; color:#6b7280; margin-top:.1rem; }
hr{ border:none; border-top:1px solid #e5e7eb; margin:.4rem 0; }
[data-testid="stDataFrame"] div[role="columnheader"],[data-testid="stDataFrame"] div[role="gridcell"]{ padding:2px 6px !important; }
[data-testid="stDataFrame"] { font-size: 0.85rem !important; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    _ensure_segments_state()

    # 1) Segments builder
    st.markdown('<div class="card"><h4>ABS segments</h4><div class="help">Add voyages and EU at-berth stays one by one. All results aggregate below.</div>', unsafe_allow_html=True)
    col_add, col_clear = st.columns([1,1])
    with col_add:
        if st.button("➕ Add segment"):
            st.session_state["abs_segments"].append(_default_segment())
    with col_clear:
        if st.button("🗑️ Clear all"):
            st.session_state["abs_segments"] = []
    # Render segments (compact)
    to_remove: List[int] = []
    for i, seg in enumerate(st.session_state["abs_segments"]):
        with st.expander(f"Segment {i+1}", expanded=True):
            seg["type"] = st.selectbox("Type", SEG_TYPES, index=SEG_TYPES.index(seg.get("type", SEG_TYPES[0])), key=f"seg_type_{i}")
            cA, cB = st.columns(2)
            with cA:
                seg["HSFO_t"]  = float_text_input("HSFO [t]" , seg.get("HSFO_t", 0.0), key=f"seg_hsfo_{i}",  min_value=0.0)
                seg["MGO_t"]   = float_text_input("MGO [t]"  , seg.get("MGO_t",  0.0), key=f"seg_mgo_{i}",   min_value=0.0)
                seg["RFNBO_t"] = float_text_input("RFNBO [t]", seg.get("RFNBO_t",0.0), key=f"seg_rfn_{i}",   min_value=0.0)
            with cB:
                seg["LFO_t"]   = float_text_input("LFO [t]"  , seg.get("LFO_t",  0.0), key=f"seg_lfo_{i}",   min_value=0.0)
                seg["BIO_t"]   = float_text_input("BIO [t]"  , seg.get("BIO_t",  0.0), key=f"seg_bio_{i}",   min_value=0.0)
            if st.button("Remove this segment", key=f"seg_remove_{i}"):
                to_remove.append(i)
    if to_remove:
        st.session_state["abs_segments"] = [s for j, s in enumerate(st.session_state["abs_segments"]) if j not in to_remove]
    st.markdown("</div>", unsafe_allow_html=True)

    # 2) Fuel properties
    st.markdown('<div class="card"><h4>Fuel properties</h4>', unsafe_allow_html=True)
    l1, l2 = st.columns(2)
    with l1:
        LCV_HSFO = float_text_input("HSFO LCV [MJ/t]", _get(DEFAULTS, "LCV_HSFO", 40_200.0), key="LCV_HSFO", min_value=0.0)
        LCV_MGO  = float_text_input("MGO LCV [MJ/t]" , _get(DEFAULTS, "LCV_MGO" , 42_700.0), key="LCV_MGO", min_value=0.0)
        WtW_HSFO = float_text_input("HSFO WtW [g/MJ]", _get(DEFAULTS, "WtW_HSFO", 92.78),    key="WtW_HSFO", min_value=0.0)
        WtW_MGO  = float_text_input("MGO WtW [g/MJ]" , _get(DEFAULTS, "WtW_MGO" , 93.93),    key="WtW_MGO",  min_value=0.0)
    with l2:
        LCV_LFO  = float_text_input("LFO LCV [MJ/t]" , _get(DEFAULTS, "LCV_LFO" , 42_700.0), key="LCV_LFO", min_value=0.0)
        LCV_BIO  = float_text_input("BIO LCV [MJ/t]" , _get(DEFAULTS, "LCV_BIO" , 38_000.0), key="LCV_BIO", min_value=0.0)
        WtW_LFO  = float_text_input("LFO WtW [g/MJ]" , _get(DEFAULTS, "WtW_LFO" , 92.00),    key="WtW_LFO",  min_value=0.0)
        WtW_RFNBO= float_text_input("RFNBO WtW [g/MJ]",_get(DEFAULTS, "WtW_RFNBO",20.00),    key="WtW_RFNBO",min_value=0.0)
    st.markdown("</div>", unsafe_allow_html=True)

    # 3) OPS (global)
    st.markdown('<div class="card"><h4>EU OPS electricity</h4>', unsafe_allow_html=True)
    OPS_kWh = float_text_input("Electricity delivered (kWh)", _get(DEFAULTS, "OPS_kWh", 0.0), key="OPS_kWh", min_value=0.0)
    OPS_MJ  = OPS_kWh * 3.6
    st.text_input("Electricity delivered (MJ) (derived)", value=us2(OPS_MJ), disabled=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # 4) Market prices (derived values use *live* factor from current mix — not zero anymore)
    st.markdown('<div class="card"><h4>Market prices</h4>', unsafe_allow_html=True)
    eur_usd_fx = float_text_input("1 Euro = … USD", _get(DEFAULTS, "eur_usd_fx", 1.00), key="eur_usd_fx", min_value=0.0)
    credit_per_tco2e = float_text_input("Credit price €/tCO₂e", _get(DEFAULTS, "credit_per_tco2e", 200.0), key="credit_per_tco2e_str", min_value=0.0)
    penalty_price_eur_per_vlsfo_t = float_text_input("Penalty price €/VLSFO-eq t", _get(DEFAULTS, "penalty_price_eur_per_vlsfo_t", 2_400.0), key="penalty_per_vlsfo_t_str", min_value=0.0)
    bio_premium_usd_per_t = float_text_input("Premium BIO vs HSFO [USD/ton]", _get(DEFAULTS, "bio_premium_usd_per_t", 0.0), key="bio_premium_usd_per_t", min_value=0.0)
    st.markdown('<div class="help">Derived below are computed from the current mix (with RFNBO ×2 reward through 2033).</div>', unsafe_allow_html=True)
    # (Derived fields filled after we compute the factor — see below.)
    st.markdown("</div>", unsafe_allow_html=True)

    # 5) Other + Optimizer
    st.markdown('<div class="card"><h4>Other settings</h4>', unsafe_allow_html=True)
    consecutive_deficit_years_seed = int(st.number_input("Consecutive deficit years (seed)", min_value=1, value=int(_get(DEFAULTS, "consecutive_deficit_years", 1)), step=1))
    opt_fuels = ["HSFO", "LFO", "MGO"]
    try:
        _idx = opt_fuels.index(_get(DEFAULTS, "opt_reduce_fuel", "HSFO"))
    except ValueError:
        _idx = 0
    selected_fuel_for_opt = st.selectbox("Fuel to reduce (for optimization)", opt_fuels, index=_idx)
    st.markdown("</div>", unsafe_allow_html=True)

    # 6) Banking & Pooling
    st.markdown('<div class="card"><h4>Banking & Pooling (tCO₂e)</h4>', unsafe_allow_html=True)
    pooling_price_eur_per_tco2e = float_text_input("Pooling price €/tCO₂e", _get(DEFAULTS, "pooling_price_eur_per_tco2e", 200.0), key="pooling_price_eur_per_tco2e", min_value=0.0)
    pooling_tco2e_input = float_text_input_signed("Pooling [tCO₂e]: + uptake, − provide", _get(DEFAULTS, "pooling_tco2e", 0.0), key="POOL_T")
    pooling_start_year = st.selectbox("Pooling starts from year", YEARS, index=YEARS.index(int(_get(DEFAULTS, "pooling_start_year", YEARS[0]))))
    banking_tco2e_input = float_text_input("Banking to next year [tCO₂e]", _get(DEFAULTS, "banking_tco2e", 0.0), key="BANK_T", min_value=0.0)
    banking_start_year = st.selectbox("Banking starts from year", YEARS, index=YEARS.index(int(_get(DEFAULTS, "banking_start_year", YEARS[0]))))
    st.markdown("</div>", unsafe_allow_html=True)

    # 7) Save
    if st.button("💾 Save current inputs as defaults"):
        defaults_to_save = {
            "eur_usd_fx": eur_usd_fx,
            "bio_premium_usd_per_t": bio_premium_usd_per_t,
            "OPS_kWh": OPS_kWh,
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
            "opt_reduce_fuel": selected_fuel_for_opt,
            "abs_segments": st.session_state.get("abs_segments", []),
        }
        try:
            with open(DEFAULTS_PATH, "w", encoding="utf-8") as f:
                json.dump(defaults_to_save, f, indent=2)
            st.success("Defaults saved.")
        except Exception as e:
            st.error(f"Could not save defaults: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# Aggregation (ABS → totals) and attained GHG (global, correct)
# ──────────────────────────────────────────────────────────────────────────────
# Mass totals
totals_mass = _segments_totals_masses()

# Publish voyage/berth split for downstream logic
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

# LCVs/WtWs (live values)
LCV_HSFO = parse_us_any(st.session_state.get("LCV_HSFO", _get(DEFAULTS,"LCV_HSFO",40200.0)), 40200.0)
LCV_LFO  = parse_us_any(st.session_state.get("LCV_LFO" , _get(DEFAULTS,"LCV_LFO" ,42700.0)), 42700.0)
LCV_MGO  = parse_us_any(st.session_state.get("LCV_MGO" , _get(DEFAULTS,"LCV_MGO" ,42700.0)), 42700.0)
LCV_BIO  = parse_us_any(st.session_state.get("LCV_BIO" , _get(DEFAULTS,"LCV_BIO" ,38000.0)), 38000.0)
LCV_RFNBO= parse_us_any(st.session_state.get("LCV_RFNBO", _get(DEFAULTS,"LCV_RFNBO",30000.0)),30000.0)

WtW_HSFO = parse_us_any(st.session_state.get("WtW_HSFO", _get(DEFAULTS,"WtW_HSFO",92.78)), 92.78)
WtW_LFO  = parse_us_any(st.session_state.get("WtW_LFO" , _get(DEFAULTS,"WtW_LFO" ,92.00)), 92.00)
WtW_MGO  = parse_us_any(st.session_state.get("WtW_MGO" , _get(DEFAULTS,"WtW_MGO" ,93.93)), 93.93)
WtW_BIO  = parse_us_any(st.session_state.get("WtW_BIO" , _get(DEFAULTS,"WtW_BIO" ,70.00)), 70.00)
WtW_RFNBO= parse_us_any(st.session_state.get("WtW_RFNBO", _get(DEFAULTS,"WtW_RFNBO",20.00)), 20.00)

wtw = {"HSFO": WtW_HSFO, "LFO": WtW_LFO, "MGO": WtW_MGO, "BIO": WtW_BIO, "RFNBO": WtW_RFNBO, "ELEC": 0.0}
LCVs_now = {"HSFO": LCV_HSFO, "LFO": LCV_LFO, "MGO": LCV_MGO, "BIO": LCV_BIO, "RFNBO": LCV_RFNBO}

# Energies by area
energies_extra_voy = _masses_to_energies(totals_mass["extra_voy"], LCVs_now)
energies_eu_berth  = _masses_to_energies(totals_mass["eu_berth"],  LCVs_now)
energies_intra_voy = _masses_to_energies(totals_mass["intra_voy"], LCVs_now)

# Global in-scope (allocator across *all* extra voyages + EU berth) + add 100% intra
scoped_extra = scoped_energies_extra_eu(energies_extra_voy, energies_eu_berth, OPS_MJ, wtw)
scoped_energies = dict(scoped_extra)
for f in ["HSFO","LFO","MGO","BIO","RFNBO"]:
    scoped_energies[f] = scoped_energies.get(f,0.0) + energies_intra_voy.get(f,0.0)

energies_fuel_full = {f: energies_extra_voy.get(f,0.0) + energies_eu_berth.get(f,0.0) + energies_intra_voy.get(f,0.0)
                      for f in ["HSFO","LFO","MGO","BIO","RFNBO"]}
energies_full = {**energies_fuel_full, "ELEC": OPS_MJ}

E_total_MJ = sum(energies_full.values())
E_scope_MJ = sum(scoped_energies.values())

# Attained GHG of final mix (global, correct)
num_phys = sum(scoped_energies.get(k,0.0) * wtw.get(k,0.0) for k in wtw.keys())
den_phys = E_scope_MJ
g_base = (num_phys / den_phys) if den_phys > 0 else 0.0
E_rfnbo_scope = scoped_energies.get("RFNBO", 0.0)

def attained_intensity_for_year(y: int) -> float:
    if den_phys <= 0: return 0.0
    r = 2.0 if y <= 2033 else 1.0
    den_rwd = den_phys + (r - 1.0) * E_rfnbo_scope
    return num_phys / den_rwd if den_rwd > 0 else 0.0

# Live factor for price linking (not zero anymore).
# Use r=2 (preview effect through 2033) to match early-years credit/penalty conversion shown in UI.
if den_phys > 0:
    den_preview = den_phys + E_rfnbo_scope  # r=2
    g_preview = num_phys / den_preview if den_preview > 0 else 0.0
else:
    g_preview = 0.0

if g_preview <= 0:
    # Fallback to baseline intensity if no energy yet (prevents zeros in derived prices)
    g_preview = BASELINE_2020_GFI

tco2e_per_vlsfo_t = (g_preview * 41_000.0) / 1_000_000.0  # > 0 by construction now
# Update the *visible* derived fields in the already-rendered sidebar card
with st.sidebar:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.text_input("Credit price €/VLSFO-eq t (derived)", value=us2(credit_per_tco2e * tco2e_per_vlsfo_t), disabled=True)
    st.text_input("Penalty price €/tCO₂e (derived)", value=us2((penalty_price_eur_per_vlsfo_t / tco2e_per_vlsfo_t) if tco2e_per_vlsfo_t>0 else 0.0), disabled=True)
    st.markdown("</div>", unsafe_allow_html=True)

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
# Per-segment compact stacks (All vs In-scope), with USD cost
# ──────────────────────────────────────────────────────────────────────────────
def _segment_energy_mj(seg: Dict[str, Any]) -> Dict[str,float]:
    return {
        "HSFO": compute_energy_MJ(seg.get("HSFO_t",0.0), LCV_HSFO),
        "LFO":  compute_energy_MJ(seg.get("LFO_t", 0.0), LCV_LFO),
        "MGO":  compute_energy_MJ(seg.get("MGO_t", 0.0), LCV_MGO),
        "BIO":  compute_energy_MJ(seg.get("BIO_t", 0.0), LCV_BIO),
        "RFNBO":compute_energy_MJ(seg.get("RFNBO_t",0.0), LCV_RFNBO),
    }

def _segment_scope_mj(seg: Dict[str,Any], energies_all: Dict[str,float]) -> Dict[str,float]:
    t = seg.get("type", SEG_TYPES[0])
    if t == "Intra-EU voyage" or t == "EU at-berth (port stay)":
        return dict(energies_all)  # 100%
    else:
        # Cross-border voyage — simple 50% view for per-segment chart (global pool still governs compliance)
        return {k: 0.5*energies_all[k] for k in energies_all.keys()}

def _mini_stack(title: str, energies_all: Dict[str,float], energies_scope: Dict[str,float], extra_note: str = ""):
    cats = ["All", "In-scope"]
    fuels = ["RFNBO","BIO","HSFO","LFO","MGO"]
    fig = go.Figure()
    for f in fuels:
        fig.add_trace(go.Bar(x=cats, y=[energies_all.get(f,0.0), energies_scope.get(f,0.0)], name=f))
    fig.update_layout(barmode="stack", height=240, margin=dict(l=30,r=10,t=40,b=30),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
                      title=dict(text=title, x=0.02, y=0.95, font=dict(size=13)))
    st.plotly_chart(fig, use_container_width=True)
    if extra_note:
        st.caption(extra_note)

# Compute 2025 net compliance cost (excluding BIO premium) to apportion as “indicative share”
years = LIMITS_DF["Year"].tolist()
limit_series = LIMITS_DF["Limit_gCO2e_per_MJ"].tolist()
step_years = [2025,2030,2035,2040,2045,2050]

# Build per-year series with banking/pooling, penalties/credits (kept from your logic)
cb_raw_t, carry_in_list, cb_eff_t = [], [], []
pool_applied, bank_applied = [], []
final_balance_t, penalties_eur, credits_eur, g_att_list = [], [], [], []
penalties_usd, credits_usd = [], []
carry = 0.0
fixed_multiplier_by_step = {}

eur_usd_fx = parse_us_any(st.session_state.get("eur_usd_fx", _get(DEFAULTS,"eur_usd_fx",1.0)), 1.0)

for y in years:
    g_target = float(LIMITS_DF.loc[LIMITS_DF["Year"]==y, "Limit_gCO2e_per_MJ"].iloc[0])
    g_att = attained_intensity_for_year(y)
    g_att_list.append(g_att)

    CB_g = (g_target - g_att) * E_scope_MJ
    CB_t_raw = CB_g / 1e6
    cb_raw_t.append(CB_t_raw)

    cb_eff = CB_t_raw + carry
    carry_in_list.append(carry)
    cb_eff_t.append(cb_eff)

    # Pooling
    if y >= int(st.session_state.get("pooling_start_year", _get(DEFAULTS,"pooling_start_year",YEARS[0]))):
        pooling_tco2e_input = parse_us_any(st.session_state.get("POOL_T", _get(DEFAULTS,"pooling_tco2e",0.0)), 0.0)
        if pooling_tco2e_input >= 0:
            pool_use = pooling_tco2e_input
        else:
            provide_abs = abs(pooling_tco2e_input)
            pre_surplus = max(cb_eff, 0.0)
            pool_use = -min(provide_abs, pre_surplus)
    else:
        pool_use = 0.0

    # Banking
    if y >= int(st.session_state.get("banking_start_year", _get(DEFAULTS,"banking_start_year",YEARS[0]))):
        requested_bank = max(parse_us_any(st.session_state.get("BANK_T", _get(DEFAULTS,"banking_tco2e",0.0)),0.0), 0.0)
        pre_surplus = max(cb_eff, 0.0)
        bank_use = min(requested_bank, pre_surplus)
    else:
        bank_use = 0.0

    # Safety clamp
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

    carry = bank_use

    # Step multiplier (constant within step; seeded)
    if final_bal < 0:
        step_idx = _step_of_year(y)
        if step_idx not in fixed_multiplier_by_step:
            seed = max(int(st.session_state.get("Consecutive deficit years (seed)", _get(DEFAULTS,"consecutive_deficit_years",1))), 1)
            fixed_multiplier_by_step[step_idx] = 1.0 + (seed - 1) * 0.10
        mult = fixed_multiplier_by_step[step_idx]
    else:
        mult = 1.0

    # €
    penalty_price_eur_per_vlsfo_t = parse_us_any(st.session_state.get("penalty_per_vlsfo_t_str", _get(DEFAULTS,"penalty_price_eur_per_vlsfo_t",2400.0)), 2400.0)
    credit_per_tco2e = parse_us_any(st.session_state.get("credit_per_tco2e_str", _get(DEFAULTS,"credit_per_tco2e",200.0)), 200.0)
    factor_preview = (g_preview * 41_000.0) / 1_000_000.0  # use preview for €/tCO2e<->€/VLSFO within UI year
    if final_bal > 0:
        credit_val = euros_from_tco2e(final_bal, g_att, credit_per_tco2e * factor_preview)
        penalty_val = 0.0
    elif final_bal < 0:
        penalty_val = euros_from_tco2e(-final_bal, g_att, penalty_price_eur_per_vlsfo_t) * mult
        credit_val = 0.0
    else:
        credit_val = penalty_val = 0.0

    pool_applied.append(pool_use); bank_applied.append(bank_use)
    final_balance_t.append(final_bal)
    penalties_eur.append(penalty_val); credits_eur.append(credit_val)
    penalties_usd.append(penalty_val * eur_usd_fx)
    credits_usd.append(credit_val * eur_usd_fx)

# BIO premium & pooling cost series
bio_mass_total_t_base = (totals_mass["intra_voy"]["BIO"] + totals_mass["extra_voy"]["BIO"] + totals_mass["eu_berth"]["BIO"])
bio_premium_usd_per_t = parse_us_any(st.session_state.get("bio_premium_usd_per_t", _get(DEFAULTS,"bio_premium_usd_per_t",0.0)), 0.0)
bio_premium_cost_usd_col = [bio_mass_total_t_base * bio_premium_usd_per_t] * len(years)
pooling_price_eur_per_tco2e = parse_us_any(st.session_state.get("pooling_price_eur_per_tco2e", _get(DEFAULTS,"pooling_price_eur_per_tco2e",200.0)), 200.0)
pooling_cost_usd_col = [pool_applied[i] * pooling_price_eur_per_tco2e * eur_usd_fx for i in range(len(years))]
total_cost_usd_col = [penalties_usd[i] + bio_premium_cost_usd_col[i] + pooling_cost_usd_col[i] for i in range(len(years))]

# Per-segment mini dashboards
st.markdown("### Per-segment energy (All vs In-scope)")
if not st.session_state["abs_segments"]:
    st.info("No segments yet. Add segments from the left sidebar.")
else:
    # For indicative cost share we use YEAR 2025 net compliance (penalty − credit + pooling), excluding BIO premium.
    year0_idx = 0
    net_compliance_usd_y0 = (penalties_usd[year0_idx] - credits_usd[year0_idx]) + pooling_cost_usd_col[year0_idx]
    total_scope_energy_all_segments = max(E_scope_MJ, 1e-9)

    # Also show an extra “OPS electricity” tile
    ops_seg = {"type":"OPS", "HSFO_t":0.0,"LFO_t":0.0,"MGO_t":0.0,"BIO_t":0.0,"RFNBO_t":0.0}
    segments_plus_ops = st.session_state["abs_segments"] + ([ops_seg] if OPS_MJ>0 else [])

    for i, seg in enumerate(segments_plus_ops):
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            if seg.get("type") == "OPS":
                energies_all = {"HSFO":0.0,"LFO":0.0,"MGO":0.0,"BIO":0.0,"RFNBO":0.0}
                energies_scope = dict(energies_all)
                # Display OPS as separate small bar (single stack)
                fig_ops = go.Figure()
                fig_ops.add_trace(go.Bar(x=["All","In-scope"], y=[OPS_MJ, OPS_MJ], name="ELEC (OPS)"))
                fig_ops.update_layout(barmode="stack", height=240, margin=dict(l=30,r=10,t=40,b=30),
                                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
                                      title=dict(text="OPS electricity", x=0.02, y=0.95, font=dict(size=13)))
                st.plotly_chart(fig_ops, use_container_width=True)
                bio_premium_usd = 0.0
                scope_mj_seg = OPS_MJ
            else:
                energies_all = _segment_energy_mj(seg)
                energies_scope = _segment_scope_mj(seg, energies_all)
                _mini_stack(f"Segment {i+1}: {seg.get('type','')}", energies_all, energies_scope)

                # Costs
                bio_t_seg = float(seg.get("BIO_t",0.0))
                bio_premium_usd = bio_t_seg * bio_premium_usd_per_t
                scope_mj_seg = sum(energies_scope.values())

        # Cost card next to each chart
        with col2:
            st.markdown("##### USD cost (segment)")
            st.write(f"• BIO premium: **${us2(bio_premium_usd)}**")
            # Indicative compliance share (proportional to in-scope MJ)
            share = (scope_mj_seg / total_scope_energy_all_segments) if total_scope_energy_all_segments>0 else 0.0
            comp_share_usd = net_compliance_usd_y0 * share
            label = "• Indicative compliance share (2025):"
            st.write(f"{label} **${us2(comp_share_usd)}**")
            st.caption("Share apportioned by in-scope energy. Global compliance uses pooled allocator; this per-segment view is indicative.")

        with col3:
            # Quick totals
            st.markdown("##### Quick numbers")
            st.write(f"All energy: **{us2(sum(energies_all.values()))} MJ**")
            st.write(f"In-scope: **{us2(scope_mj_seg)} MJ**")

# ──────────────────────────────────────────────────────────────────────────────
# System-level chart — GHG Intensity vs Limit
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('### GHG Intensity vs. FuelEU Limit (2025–2050)')
actual_series = [attained_intensity_for_year(y) for y in years]
limit_text = [f"{limit_series[i]:,.2f}" if years[i] in step_years else "" for i in range(len(years))]
attained_text = [f"{actual_series[i]:,.2f}" if years[i] in step_years else "" for i in range(len(years))]

fig = go.Figure()
fig.add_trace(go.Scatter(x=years, y=limit_series, name="FuelEU Limit (step)",
                         mode="lines+markers+text", line=dict(shape="hv", width=3),
                         text=limit_text, textposition="bottom center", textfont=dict(size=12),
                         hovertemplate="Year=%{x}<br>Limit=%{y:,.2f} gCO₂e/MJ<extra></extra>"))
fig.add_trace(go.Scatter(x=years, y=actual_series, name="Attained GHG (global mix)",
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
# Results table (your banking/pooling + optimizer kept)
# ──────────────────────────────────────────────────────────────────────────────
st.header("Results (merged per-year table)")

# Optimizer helpers (unchanged logic except inputs come from ABS totals)
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
        h_v, l_v, m_v, b_v, r_v, h_b, l_b, m_b, b_b, r_b, OPS_MJ, wtw, year
    )
    if E_scope_x <= 0: return 0.0, 0.0
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
    h_v, l_v, m_v, b_v, r_v = HSFO_voy_t, LFO_voy_t, MGO_voy_t, BIO_voy_t, RFNBO_voy_t
    h_b, l_b, m_b, b_b, r_b = HSFO_berth_t, LFO_berth_t, MGO_berth_t, BIO_berth_t, RFNBO_berth_t
    if fuel == "HSFO": s_v, s_b, LCV_S = h_v, h_b, LCV_HSFO
    elif fuel == "LFO": s_v, s_b, LCV_S = l_v, l_b, LCV_LFO
    else:              s_v, s_b, LCV_S = m_v, m_b, LCV_MGO
    x = max(0.0, float(x_decrease_t)); x = min(x, s_v + s_b)
    bio_increase_t = (x * LCV_S / LCV_BIO) if LCV_BIO > 0 else 0.0
    take_v = min(x, s_v); s_v -= take_v
    rem = x - take_v; s_b = max(0.0, s_b - rem)
    b_b += min(bio_increase_t, float("inf"))
    rem_bio = bio_increase_t - min(bio_increase_t, float("inf"))
    if rem_bio > 0: b_v += rem_bio
    if fuel == "HSFO": h_v, h_b = s_v, s_b
    elif fuel == "LFO": l_v, l_b = s_v, s_b
    else: m_v, m_b = s_v, s_b
    return h_v, l_v, m_v, b_v, r_v, h_b, l_b, m_b, b_b, r_b

# Optimizer scan (same approach)
dec_opt_list, bio_inc_opt_list = [], []
for i in range(len(years)):
    if selected_fuel_for_opt == "HSFO": total_avail, LCV_SEL = HSFO_voy_t + HSFO_berth_t, LCV_HSFO
    elif selected_fuel_for_opt == "LFO": total_avail, LCV_SEL = LFO_voy_t + LFO_berth_t, LCV_LFO
    else:                                 total_avail, LCV_SEL = MGO_voy_t + MGO_berth_t, LCV_MGO
    if total_avail <= 0 or LCV_BIO <= 0: dec_opt_list.append(0.0); bio_inc_opt_list.append(0.0); continue
    steps_coarse, x_max = 60, total_avail
    best_x, best_cost = 0.0, float("inf")
    for s in range(steps_coarse + 1):
        x = x_max * s / steps_coarse
        masses = masses_after_shift_generic(selected_fuel_for_opt, x)
        penalty_usd_x, _ = penalty_usd_with_masses_for_year(i, *masses)
        new_bio_total_t = (masses[3] + masses[8])
        total_cost_x = penalty_usd_x + new_bio_total_t * bio_premium_usd_per_t
        if total_cost_x < best_cost: best_cost, best_x = total_cost_x, x
    delta = x_max / steps_coarse * 2.0
    left, right, steps_fine = max(0.0, best_x - delta), min(x_max, best_x + delta), 80
    for s in range(steps_fine + 1):
        x = left + (right - left) * s / steps_fine
        masses = masses_after_shift_generic(selected_fuel_for_opt, x)
        penalty_usd_x, _ = penalty_usd_with_masses_for_year(i, *masses)
        new_bio_total_t = (masses[3] + masses[8])
        total_cost_x = penalty_usd_x + new_bio_total_t * bio_premium_usd_per_t
        if total_cost_x < best_cost: best_cost, best_x = total_cost_x, x
    dec_opt = best_x
    bio_inc_opt = dec_opt * (LCV_SEL / LCV_BIO) if LCV_BIO > 0 else 0.0
    dec_opt_list.append(dec_opt); bio_inc_opt_list.append(bio_inc_opt)

# Recompute optimized costs
penalties_usd_opt_col, bio_premium_cost_usd_opt_col, total_cost_usd_opt_col = [], [], []
for i in range(len(years)):
    x_opt = dec_opt_list[i]
    if x_opt <= 0.0 or LCV_BIO <= 0.0:
        penalties_usd_opt = penalties_usd[i]
        bio_premium_usd_opt = bio_premium_cost_usd_col[i]
    else:
        masses_opt = masses_after_shift_generic(selected_fuel_for_opt, x_opt)
        penalties_usd_opt, _ = penalty_usd_with_masses_for_year(i, *masses_opt)
        new_bio_total_t_opt = (masses_opt[3] + masses_opt[8])
        bio_premium_usd_opt = new_bio_total_t_opt * bio_premium_usd_per_t
    penalties_usd_opt_col.append(penalties_usd_opt)
    bio_premium_cost_usd_opt_col.append(bio_premium_usd_opt)
    total_cost_usd_opt_col.append(penalties_usd_opt + bio_premium_usd_opt + pooling_cost_usd_col[i])

# Table
decrease_col_name = f"{selected_fuel_for_opt}_decrease(t)_for_Opt_Cost"
df_cost = pd.DataFrame({
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
})
df_fmt = df_cost.copy()
for col in df_fmt.columns:
    if col != "Year": df_fmt[col] = df_fmt[col].apply(us2)

st.dataframe(df_fmt, use_container_width=True)
st.download_button("Download per-year results (CSV)", data=df_fmt.to_csv(index=False), file_name="fueleu_results_2025_2050.csv", mime="text/csv")
