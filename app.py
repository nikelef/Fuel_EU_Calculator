# app.py — FuelEU Maritime Calculator with ABS-style Segments
# Period: 2025–2050 • EUR-only • WtW basis
# Includes: ABS prioritized allocation for Extra-EU voyages, segment builder (add segments one-by-one),
# optimizer (fossil→BIO), banking & pooling, price-linking, per-segment stacks w/ arrows, final combined stack,
# fixed derived prices, OPS only under EU berth, compact/clean sidebar formatting.

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
# Segment model (ABS-style)
# ──────────────────────────────────────────────────────────────────────────────
SEG_INTRA_VOY = "Intra-EU voyage"      # 100% scope
SEG_IE_VOY   = "Intra→Extra voyage"    # Extra-EU voyage → pooled 50%, prioritized fill
SEG_EI_VOY   = "Extra→Intra voyage"    # Extra-EU voyage → pooled 50%, prioritized fill
SEG_EU_BERTH = "EU at-berth"           # 100% scope + OPS electricity (100%)
SEG_TYPES = [SEG_INTRA_VOY, SEG_IE_VOY, SEG_EI_VOY, SEG_EU_BERTH]

FUEL_KEYS = ["HSFO", "LFO", "MGO", "BIO", "RFNBO"]
ALL_KEYS_WITH_ELEC = FUEL_KEYS + ["ELEC"]

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
# Formatting helpers (US format)
# ──────────────────────────────────────────────────────────────────────────────
def us2(x: float) -> str:
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return x

def us0(x: float) -> str:
    try:
        return f"{float(x):,.0f}"
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

def float_text_input(label: str, default_val: float, key: str, min_value: float = 0.0,
                     label_visibility: str = "visible") -> float:
    if key not in st.session_state:
        st.session_state[key] = us2(default_val)
    def _normalize():
        val = parse_us(st.session_state[key], default=default_val, min_value=min_value)
        st.session_state[key] = us2(val)
    st.text_input(label, value=st.session_state[key], key=key, on_change=_normalize,
                  label_visibility=label_visibility)
    return parse_us(st.session_state[key], default=default_val, min_value=min_value)

def float_text_input_signed(label: str, default_val: float, key: str) -> float:
    if key not in st.session_state:
        st.session_state[key] = us2(default_val)
    def _normalize():
        val = parse_us_any(st.session_state[key], default=default_val)
        st.session_state[key] = us2(val)
    st.text_input(label, value=st.session_state[key], key=key, on_change=_normalize,
                  label_visibility="visible")
    return parse_us_any(st.session_state[key], default=default_val)

def _ss_or_def(key: str, default_val: float) -> float:
    if key in st.session_state:
        try:
            return parse_us_any(st.session_state[key], default=default_val)
        except Exception:
            return float(default_val)
    return float(_get(DEFAULTS, key, default_val))

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
# ABS-style prioritized allocator for Extra-EU voyages
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
             - then take VOYAGE part but only up to the spare pool after reserving berth fossils.
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
# Segment utilities
# ──────────────────────────────────────────────────────────────────────────────
def _segment_energies_MJ(seg: Dict[str, Any],
                         LCV_HSFO, LCV_LFO, LCV_MGO, LCV_BIO, LCV_RFNBO) -> Tuple[Dict[str,float], Dict[str,float], float]:
    """
    Returns: (voyage_fuels_MJ, berth_fuels_MJ, elec_MJ) for a single segment.
    - Intra-EU voyage: voyage_fuels (100% scope later), berth=0, elec=0
    - IE/EI voyage: voyage_fuels (pooled later), berth=0, elec=0
    - EU at-berth: berth_fuels (100% scope later), voyage=0, elec from OPS
    """
    m = seg.get("masses_t", {})  # dict with HSFO/LFO/MGO/BIO/RFNBO in tons
    toMJ = lambda tons, lcv: compute_energy_MJ(float(tons), float(lcv))
    voy = {k: 0.0 for k in FUEL_KEYS}
    brt = {k: 0.0 for k in FUEL_KEYS}
    elec = 0.0

    if seg["type"] in (SEG_INTRA_VOY, SEG_IE_VOY, SEG_EI_VOY):
        voy["HSFO"]  = toMJ(m.get("HSFO",0), LCV_HSFO)
        voy["LFO"]   = toMJ(m.get("LFO",0),  LCV_LFO)
        voy["MGO"]   = toMJ(m.get("MGO",0),  LCV_MGO)
        voy["BIO"]   = toMJ(m.get("BIO",0),  LCV_BIO)
        voy["RFNBO"] = toMJ(m.get("RFNBO",0),LCV_RFNBO)
    if seg["type"] == SEG_EU_BERTH:
        brt["HSFO"]  = toMJ(m.get("HSFO",0), LCV_HSFO)
        brt["LFO"]   = toMJ(m.get("LFO",0),  LCV_LFO)
        brt["MGO"]   = toMJ(m.get("MGO",0),  LCV_MGO)
        brt["BIO"]   = toMJ(m.get("BIO",0),  LCV_BIO)
        brt["RFNBO"] = toMJ(m.get("RFNBO",0),LCV_RFNBO)
        elec = float(seg.get("OPS_kWh", 0.0)) * 3.6

    return voy, brt, elec

def _gather_segments_or_fallback(voyage_type: str,
                                 legacy_values: Dict[str, float]) -> Tuple[List[Dict[str,Any]], bool]:
    """
    Returns (segments, used_fallback).
    Fallback creates segment(s) using legacy fields so the app works even without the ABS UI.
    """
    if "segments" in st.session_state and st.session_state["segments"]:
        return st.session_state["segments"], False

    segs = []
    if "Extra-EU" in voyage_type:
        segs.append({"type": SEG_IE_VOY,
                     "masses_t": {
                         "HSFO": legacy_values["HSFO_voy_t"],
                         "LFO":  legacy_values["LFO_voy_t"],
                         "MGO":  legacy_values["MGO_voy_t"],
                         "BIO":  legacy_values["BIO_voy_t"],
                         "RFNBO":legacy_values["RFNBO_voy_t"],
                     }})
        if any([legacy_values["HSFO_berth_t"], legacy_values["LFO_berth_t"],
                legacy_values["MGO_berth_t"], legacy_values["BIO_berth_t"], legacy_values["RFNBO_berth_t"]]):
            segs.append({"type": SEG_EU_BERTH,
                         "masses_t": {
                             "HSFO": legacy_values["HSFO_berth_t"],
                             "LFO":  legacy_values["LFO_berth_t"],
                             "MGO":  legacy_values["MGO_berth_t"],
                             "BIO":  legacy_values["BIO_berth_t"],
                             "RFNBO":legacy_values["RFNBO_berth_t"],
                         },
                         "OPS_kWh": legacy_values.get("OPS_kWh", 0.0)})
    else:
        segs.append({"type": SEG_INTRA_VOY,
                     "masses_t": {
                         "HSFO": legacy_values["HSFO_voy_t"],
                         "LFO":  legacy_values["LFO_voy_t"],
                         "MGO":  legacy_values["MGO_voy_t"],
                         "BIO":  legacy_values["BIO_voy_t"],
                         "RFNBO":legacy_values["RFNBO_voy_t"],
                     }})
    return segs, True

def compute_combined_scope(segments: List[Dict[str,Any]],
                           LCVs: Dict[str,float],
                           wtw: Dict[str,float]) -> Dict[str, Dict[str,float]]:
    """
    Returns dict with:
      {
        "total_all": {fuel->MJ, "ELEC":MJ},
        "intra_all": {fuel->MJ},           # all MJ from intra-EU voyages
        "berth_all": {fuel->MJ},           # all MJ from EU berth
        "extra_voy_all": {fuel->MJ},       # all MJ from IE/EI voyages
        "elec_MJ": float,
        "in_scope_final": {fuel->MJ, "ELEC":MJ},  # combined in-scope for compliance
        "in_scope_segments": [ {fuel->MJ,"ELEC":MJ} per segment ]  # per-segment in-scope (for stacks)
      }
    Combined in-scope = (100% intra-EU) + scoped_energies_extra_eu( sum(extra voyages), sum(berth), sum(OPS) ).
    """
    LCV_HSFO, LCV_LFO, LCV_MGO, LCV_BIO, LCV_RFNBO = LCVs["HSFO"], LCVs["LFO"], LCVs["MGO"], LCVs["BIO"], LCVs["RFNBO"]

    zero = {k:0.0 for k in FUEL_KEYS}
    total_all = {k:0.0 for k in FUEL_KEYS}
    intra_all = {k:0.0 for k in FUEL_KEYS}
    berth_all = {k:0.0 for k in FUEL_KEYS}
    extra_voy_all = {k:0.0 for k in FUEL_KEYS}
    elec_total = 0.0

    in_scope_segments = []

    for seg in segments:
        voy, brt, elec = _segment_energies_MJ(seg, LCV_HSFO, LCV_LFO, LCV_MGO, LCV_BIO, LCV_RFNBO)

        # accumulate totals
        for k in FUEL_KEYS:
            total_all[k] += voy[k] + brt[k]
        elec_total += elec

        # per-segment in-scope (for stacks)
        if seg["type"] == SEG_INTRA_VOY:
            # 100% in-scope
            in_scope_segments.append({**voy, "ELEC": 0.0})
            for k in FUEL_KEYS: intra_all[k] += voy[k]
        elif seg["type"] in (SEG_IE_VOY, SEG_EI_VOY):
            # segment-level extra-EU prioritized allocation (no berth, no elec locally)
            scoped_seg = scoped_energies_extra_eu(voy, zero, 0.0, wtw)
            in_scope_segments.append(scoped_seg)
            for k in FUEL_KEYS: extra_voy_all[k] += voy[k]
        else:  # EU berth
            # 100% in-scope + 100% ELEC
            scoped_seg = {**brt, "ELEC": elec}
            in_scope_segments.append(scoped_seg)
            for k in FUEL_KEYS: berth_all[k] += brt[k]

    # final combined in-scope (global pool over EU-berth + 50% of all Extra-EU voyages)
    scoped_extra_global = scoped_energies_extra_eu(extra_voy_all, berth_all, elec_total, wtw)

    # add 100% of intra-EU voyages
    in_scope_final = {k: scoped_extra_global.get(k,0.0) + intra_all.get(k,0.0) for k in FUEL_KEYS}
    in_scope_final["ELEC"] = scoped_extra_global.get("ELEC", 0.0)

    total_all_with_elec = {**total_all, "ELEC": elec_total}

    return {
        "total_all": total_all_with_elec,
        "intra_all": intra_all,
        "berth_all": berth_all,
        "extra_voy_all": extra_voy_all,
        "elec_MJ": elec_total,
        "in_scope_final": in_scope_final,
        "in_scope_segments": in_scope_segments,
    }

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit setup & CSS (tidy sidebar, compact inputs)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="FuelEU Maritime Calculator", layout="wide")

st.title("FuelEU Maritime — GHG Intensity & Cost — TMS DRY — ENVIRONMENTAL")
st.caption("Period: 2025–2050 • Limits derived from 2020 baseline 91.16 gCO₂e/MJ • WtW basis • Prices in EUR")

st.markdown(
    """
    <style>
    /* Compact metric styling */
    [data-testid="stMetricLabel"] { font-size: 0.95rem !important; font-weight: 800 !important; color: #111827 !important; }
    [data-testid="stMetricValue"] { font-size: 0.90rem !important; font-weight: 700 !important; line-height: 1.05 !important; }
    [data-testid="stMetric"] { padding: .20rem .30rem !important; }

    /* Sidebar spacing & tidy inputs */
    section[data-testid="stSidebar"] div.block-container{ padding-top: .6rem !important; padding-bottom: .6rem !important; }
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{ gap: .45rem !important; }
    section[data-testid="stSidebar"] [data-testid="column"]{ padding-left:.20rem; padding-right:.20rem; }
    section[data-testid="stSidebar"] label{ font-size: .95rem; margin-bottom: .15rem; font-weight: 600; }
    section[data-testid="stSidebar"] input[type="text"], section[data-testid="stSidebar"] input[type="number"]{
        padding: .28rem .50rem; height: 1.9rem; min-height: 1.9rem; border-radius: 6px;
    }
    .section-title{ font-weight:800; font-size:1.0rem; margin:.3rem 0 .2rem 0; }
    .muted-note{ font-size:.86rem; color:#6b7280; margin:-.05rem 0 .35rem 0; }

    /* Dataframe tighter cells */
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
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — inputs (ABS segments, LCV, WtW, prices, optimizer, banking/pooling)
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Exchange Rate
    st.markdown('<div class="section-title">Exchange rate</div>', unsafe_allow_html=True)
    eur_usd_fx = float_text_input("1 Euro = … USD", _get(DEFAULTS, "eur_usd_fx", 1.00),
                                  key="eur_usd_fx", min_value=0.0)

    # Scope radio (legacy compatibility)
    st.markdown('<div class="section-title">Default/scenario scope (for fallback only)</div>', unsafe_allow_html=True)
    scope_options = ["Intra-EU (100%)", "Extra-EU (50%)"]
    saved_scope = _get(DEFAULTS, "voyage_type", scope_options[0])
    try:
        idx = scope_options.index(saved_scope)
    except ValueError:
        idx = 0
    voyage_type = st.radio("Voyage scope (legacy fallback)", scope_options, index=idx, horizontal=True)

    # Legacy masses (used only if no segments exist)
    st.markdown('<div class="section-title">Legacy inputs (used only if no segments exist)</div>', unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    with m1:
        HSFO_voy_t = float_text_input("Voyage HSFO [t]", _get(DEFAULTS, "HSFO_voy_t", _get(DEFAULTS, "HSFO_t", 5_000.0)),
                                      key="HSFO_voy_t", min_value=0.0)
    with m2:
        LFO_voy_t  = float_text_input("Voyage LFO [t]" , _get(DEFAULTS, "LFO_voy_t" , _get(DEFAULTS, "LFO_t" , 0.0)),
                                      key="LFO_voy_t", min_value=0.0)
    m3, m4 = st.columns(2)
    with m3:
        MGO_voy_t  = float_text_input("Voyage MGO [t]" , _get(DEFAULTS, "MGO_voy_t" , _get(DEFAULTS, "MGO_t" , 0.0)),
                                      key="MGO_voy_t", min_value=0.0)
    with m4:
        BIO_voy_t  = float_text_input("Voyage BIO [t]" , _get(DEFAULTS, "BIO_voy_t" , _get(DEFAULTS, "BIO_t" , 0.0)),
                                      key="BIO_voy_t", min_value=0.0)
    m5, _ = st.columns(2)
    with m5:
        RFNBO_voy_t = float_text_input("Voyage RFNBO [t]", _get(DEFAULTS, "RFNBO_voy_t", _get(DEFAULTS, "RFNBO_t", 0.0)),
                                       key="RFNBO_voy_t", min_value=0.0)
    # Legacy berth masses (used in fallback if no segments exist)
    bm1, bm2 = st.columns(2)
    with bm1:
        HSFO_berth_t = float_text_input("Berth HSFO [t]", _get(DEFAULTS, "HSFO_berth_t", 0.0),
                                        key="HSFO_berth_t", min_value=0.0)
    with bm2:
        LFO_berth_t  = float_text_input("Berth LFO [t]" , _get(DEFAULTS, "LFO_berth_t" , 0.0),
                                        key="LFO_berth_t", min_value=0.0)
    bm3, bm4 = st.columns(2)
    with bm3:
        MGO_berth_t  = float_text_input("Berth MGO [t]" , _get(DEFAULTS, "MGO_berth_t" , 0.0),
                                        key="MGO_berth_t", min_value=0.0)
    with bm4:
        BIO_berth_t  = float_text_input("Berth BIO [t]" , _get(DEFAULTS, "BIO_berth_t" , 0.0),
                                        key="BIO_berth_t", min_value=0.0)
    bm5, _ = st.columns(2)
    with bm5:
        RFNBO_berth_t = float_text_input("Berth RFNBO [t]", _get(DEFAULTS, "RFNBO_berth_t", 0.0),
                                         key="RFNBO_berth_t", min_value=0.0)

    st.divider()

    # ABS-style segments
    st.markdown('<div class="section-title">Segments (ABS style)</div>', unsafe_allow_html=True)

    # Load persisted segments once
    if "segments" not in st.session_state and "segments" in DEFAULTS:
        st.session_state["segments"] = DEFAULTS["segments"]

    if "segments" not in st.session_state:
        st.session_state["segments"] = []

    # Add segment UI
    add_cols = st.columns([1.5, 1, 0.8])
    with add_cols[0]:
        new_seg_type = st.selectbox("Add segment type", SEG_TYPES, key="__new_seg_type")
    with add_cols[1]:
        if st.button("➕ Add segment"):
            st.session_state["segments"].append({
                "type": new_seg_type,
                "masses_t": {"HSFO":0.0,"LFO":0.0,"MGO":0.0,"BIO":0.0,"RFNBO":0.0},
                **({"OPS_kWh":0.0} if new_seg_type == SEG_EU_BERTH else {})
            })
    with add_cols[2]:
        if st.button("🗑 Clear all"):
            st.session_state["segments"] = []

    # Render existing segments
    for i, seg in enumerate(st.session_state["segments"]):
        with st.expander(f"Segment {i+1}: {seg['type']}", expanded=False):
            # Type selector (allows changing after creation)
            seg["type"] = st.selectbox(f"Type (Segment {i+1})", SEG_TYPES,
                                       index=SEG_TYPES.index(seg["type"]),
                                       key=f"seg_type_{i}")
            # Masses per fuel
            grid = st.columns(5)
            for j, f in enumerate(FUEL_KEYS):
                with grid[j]:
                    seg["masses_t"][f] = float_text_input(f"{f} [t]", seg["masses_t"].get(f,0.0),
                                                          key=f"seg{i}_{f}", min_value=0.0)
            # OPS only for EU berth
            if seg["type"] == SEG_EU_BERTH:
                kwh = float_text_input("Electricity delivered (kWh)", seg.get("OPS_kWh", 0.0),
                                       key=f"OPS_kWh_{i}", min_value=0.0)
                seg["OPS_kWh"] = kwh
                st.text_input("Electricity delivered (MJ) (derived)", value=us2(kwh*3.6), disabled=True)

            # Remove button
            rm_cols = st.columns([0.2, 0.8])
            with rm_cols[0]:
                if st.button("Remove", key=f"rm_{i}"):
                    st.session_state["segments"].pop(i)
                    st.experimental_rerun()

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

    # WtW
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

    # Build a preview factor from current segments/LCVs/WtWs for price linking
    LCVs = {"HSFO": LCV_HSFO, "LFO": LCV_LFO, "MGO": LCV_MGO, "BIO": LCV_BIO, "RFNBO": LCV_RFNBO}
    wtw = {"HSFO": WtW_HSFO, "LFO": WtW_LFO, "MGO": WtW_MGO, "BIO": WtW_BIO, "RFNBO": WtW_RFNBO, "ELEC": 0.0}

    # Build segments or fallback (using legacy values)
    legacy_vals = dict(
        HSFO_voy_t=HSFO_voy_t, LFO_voy_t=LFO_voy_t, MGO_voy_t=MGO_voy_t, BIO_voy_t=BIO_voy_t, RFNBO_voy_t=RFNBO_voy_t,
        HSFO_berth_t=HSFO_berth_t, LFO_berth_t=LFO_berth_t, MGO_berth_t=MGO_berth_t, BIO_berth_t=BIO_berth_t, RFNBO_berth_t=RFNBO_berth_t,
        OPS_kWh=0.0,
    )
    segments, _used_fallback = _gather_segments_or_fallback(voyage_type, legacy_vals)

    # Compute combined to derive factor
    combo_preview = compute_combined_scope(segments, LCVs, wtw)
    total_scope_prev = sum(combo_preview["in_scope_final"].values())
    num_phys_prev = sum((combo_preview["in_scope_final"].get(k,0.0) * wtw.get(k,0.0)) for k in wtw.keys())
    E_rfnbo_scope_prev = combo_preview["in_scope_final"].get("RFNBO", 0.0)
    if total_scope_prev > 0:
        # use r=2 reward (<=2033) for preview factor
        den_prev = total_scope_prev + (2.0 - 1.0) * E_rfnbo_scope_prev
        g_preview = (num_phys_prev / den_prev) if den_prev > 0 else 0.0
        factor_vlsfo_per_tco2e_prev = (g_preview * 41_000.0) / 1_000_000.0 if g_preview > 0 else 0.0
    else:
        factor_vlsfo_per_tco2e_prev = 0.0
    st.session_state["factor_vlsfo_per_tco2e"] = factor_vlsfo_per_tco2e_prev

    st.divider()

    # Compliance Market — Credits (user edits €/tCO2e; derived €/VLSFO-eq t)
    st.markdown('<div class="section-title">Compliance Market — Credits</div>', unsafe_allow_html=True)
    credit_per_tco2e = float_text_input("Credit price €/tCO₂e",
                                        _get(DEFAULTS, "credit_per_tco2e", 200.0),
                                        key="credit_per_tco2e_str", min_value=0.0)
    credit_price_eur_per_vlsfo_t = credit_per_tco2e * factor_vlsfo_per_tco2e_prev if factor_vlsfo_per_tco2e_prev > 0 else 0.0
    st.text_input("Credit price €/VLSFO-eq t (derived)", value=us2(credit_price_eur_per_vlsfo_t), disabled=True)

    # Compliance Market — Penalties (user edits €/VLSFO-eq t; derived €/tCO2e)
    st.markdown('<div class="section-title">Compliance Market — Penalties</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted-note">Regulated default. Change only if regulation changes.</div>', unsafe_allow_html=True)
    penalty_price_eur_per_vlsfo_t = float_text_input("Penalty price €/VLSFO-eq t",
                                                     _get(DEFAULTS, "penalty_price_eur_per_vlsfo_t", 2_400.0),
                                                     key="penalty_per_vlsfo_t_str", min_value=0.0)
    penalty_price_eur_per_tco2e_prev = (penalty_price_eur_per_vlsfo_t / factor_vlsfo_per_tco2e_prev) if factor_vlsfo_per_tco2e_prev > 0 else 0.0
    st.text_input("Penalty price €/tCO₂e (derived)", value=us2(penalty_price_eur_per_tco2e_prev), disabled=True)

    # BIO premium (USD/t)
    st.markdown('<div class="section-title">Premium BIO vs HSFO [USD/ton]</div>', unsafe_allow_html=True)
    bio_premium_usd_per_t = float_text_input("", _get(DEFAULTS, "bio_premium_usd_per_t", 0.0),
                                             key="bio_premium_usd_per_t", min_value=0.0,
                                             label_visibility="collapsed")

    st.divider()

    # Other + Optimizer selection
    st.markdown('<div class="section-title">Other</div>', unsafe_allow_html=True)
    consecutive_deficit_years_seed = int(
        st.number_input("Consecutive deficit years (seed)", min_value=1,
                        value=int(_get(DEFAULTS, "consecutive_deficit_years", 1)), step=1)
    )
    opt_fuels = ["HSFO", "LFO", "MGO"]
    _saved_opt_fuel = _get(DEFAULTS, "opt_reduce_fuel", "HSFO")
    try:
        _idx = opt_fuels.index(_saved_opt_fuel)
    except ValueError:
        _idx = 0
    selected_fuel_for_opt = st.selectbox("Fuel to reduce (for optimization)", opt_fuels, index=_idx)

    st.divider()

    # Banking & Pooling
    st.markdown('<div class="section-title">Banking & Pooling (tCO₂e)</div>', unsafe_allow_html=True)
    pooling_price_eur_per_tco2e = float_text_input(
        "Pooling price €/tCO₂e",
        _get(DEFAULTS, "pooling_price_eur_per_tco2e", 200.0),
        key="pooling_price_eur_per_tco2e",
        min_value=0.0
    )
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

    # Save defaults (including segments)
    if st.button("💾 Save current inputs as defaults"):
        defaults_to_save = {
            "voyage_type": voyage_type,
            "eur_usd_fx": eur_usd_fx,
            "bio_premium_usd_per_t": bio_premium_usd_per_t,
            "HSFO_voy_t": HSFO_voy_t, "LFO_voy_t": LFO_voy_t, "MGO_voy_t": MGO_voy_t, "BIO_voy_t": BIO_voy_t, "RFNBO_voy_t": RFNBO_voy_t,
            "HSFO_berth_t": HSFO_berth_t, "LFO_berth_t": LFO_berth_t, "MGO_berth_t": MGO_berth_t, "BIO_berth_t": BIO_berth_t, "RFNBO_berth_t": RFNBO_berth_t,
            "LCV_HSFO": LCV_HSFO, "LCV_LFO": LCV_LFO, "LCV_MGO": LCV_MGO, "LCV_BIO": LCV_BIO, "LCV_RFNBO": LCV_RFNBO,
            "WtW_HSFO": WtW_HSFO, "WtW_LFO": WtW_LFO, "WtW_MGO": WtW_MGO, "WtW_BIO": WtW_BIO, "WtW_RFNBO": WtW_RFNBO,
            "credit_per_tco2e": credit_per_tco2e,
            "penalty_price_eur_per_vlsfo_t": penalty_price_eur_per_vlsfo_t,
            "consecutive_deficit_years": consecutive_deficit_years_seed,
            "banking_tco2e": banking_tco2e_input,
            "pooling_tco2e": pooling_tco2e_input,
            "pooling_start_year": int(pooling_start_year),
            "banking_start_year": int(banking_start_year),
            "opt_reduce_fuel": selected_fuel_for_opt,
            "pooling_price_eur_per_tco2e": pooling_price_eur_per_tco2e,
            "segments": st.session_state["segments"],
        }
        try:
            with open(DEFAULTS_PATH, "w", encoding="utf-8") as f:
                json.dump(defaults_to_save, f, indent=2)
            st.success("Defaults saved. They will be used next time the app starts.")
        except Exception as e:
            st.error(f"Could not save defaults: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# Derived energies & intensity (IN-SCOPE for compliance) — COMBINED
# ──────────────────────────────────────────────────────────────────────────────
# Recompute from live values (outside sidebar)
LCVs = {"HSFO": LCV_HSFO, "LFO": LCV_LFO, "MGO": LCV_MGO, "BIO": LCV_BIO, "RFNBO": LCV_RFNBO}
wtw = {"HSFO": WtW_HSFO, "LFO": WtW_LFO, "MGO": WtW_MGO, "BIO": WtW_BIO, "RFNBO": WtW_RFNBO, "ELEC": 0.0}
segments, _ = _gather_segments_or_fallback(voyage_type, legacy_vals)
combo = compute_combined_scope(segments, LCVs, wtw)

energies_full = combo["total_all"]                 # MJ incl. ELEC
scoped_combined = combo["in_scope_final"]          # combined in-scope (global pool for Extra-EU)
scoped_per_segment = combo["in_scope_segments"]    # per-segment in-scope (for stacks)

E_total_MJ = sum(energies_full.values())
E_scope_MJ = sum(scoped_combined.values())

# physical (no RFNBO reward)
num_phys = sum(scoped_combined.get(k,0.0) * wtw.get(k,0.0) for k in wtw.keys())
den_phys = E_scope_MJ
g_base = (num_phys / den_phys) if den_phys > 0 else 0.0

E_rfnbo_scope = scoped_combined.get("RFNBO", 0.0)
def attained_intensity_for_year(y: int) -> float:
    if den_phys <= 0:
        return 0.0
    r = 2.0 if y <= 2033 else 1.0
    den_rwd = den_phys + (r - 1.0) * E_rfnbo_scope
    return num_phys / den_rwd if den_rwd > 0 else 0.0

# Update live factor (for any later uses)
g_preview_live = attained_intensity_for_year(2025)  # r=2 period
factor_vlsfo_per_tco2e_live = (g_preview_live * 41_000.0) / 1_000_000.0 if g_preview_live > 0 else 0.0
st.session_state["factor_vlsfo_per_tco2e"] = factor_vlsfo_per_tco2e_live

# ──────────────────────────────────────────────────────────────────────────────
# Top breakdown (smaller numbers)
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("Energy breakdown (MJ)")
cA, cB, cC, cD, cE, cF, cG, cH = st.columns(8)
with cA: st.metric("Total energy (all)", f"{us0(E_total_MJ)} MJ")
with cB: st.metric("In-scope energy", f"{us0(E_scope_MJ)} MJ")
with cC: st.metric("Fossil — all", f"{us0(energies_full['HSFO'] + energies_full['LFO'] + energies_full['MGO'])} MJ")
with cD: st.metric("BIO — all", f"{us0(energies_full['BIO'])} MJ")
with cE: st.metric("RFNBO — all", f"{us0(energies_full['RFNBO'])} MJ")
with cF: st.metric("Fossil — in scope", f"{us0(scoped_combined.get('HSFO',0)+scoped_combined.get('LFO',0)+scoped_combined.get('MGO',0))} MJ")
with cG: st.metric("BIO — in scope", f"{us0(scoped_combined.get('BIO',0))} MJ")
with cH: st.metric("RFNBO — in scope", f"{us0(scoped_combined.get('RFNBO',0))} MJ")

# ──────────────────────────────────────────────────────────────────────────────
# Stacks — per segment + final combined (ascending WtW, arrows, hover MJ)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<h2 style="margin:0 0 .25rem 0;">Energy composition by segment</h2>', unsafe_allow_html=True)

COLORS = {
    "ELEC":  "#FACC15",
    "RFNBO": "#86EFAC",
    "BIO":   "#065F46",
    "MGO":   "#93C5FD",
    "LFO":   "#2563EB",
    "HSFO":  "#1E3A8A",
}
fuels_sorted = sorted(["RFNBO","BIO","HSFO","LFO","MGO"], key=lambda f: wtw.get(f, float("inf")))
stack_layers = [("ELEC", "ELEC (OPS)")] + [(f, f) for f in fuels_sorted]

def stack_two_bars(title: str, left_dict: Dict[str,float], right_dict: Dict[str,float]):
    cats = ["All energy", "In-scope energy"]
    fig = go.Figure()
    for key, label in stack_layers:
        y_left = left_dict.get(key, 0.0)
        y_right = right_dict.get(key, 0.0)
        if (y_left == 0.0 and y_right == 0.0):
            continue
        fig.add_trace(
            go.Bar(
                x=cats,
                y=[y_left, y_right],
                name=label,
                marker_color=COLORS.get(key, None),
                hovertemplate=f"{label}<br>%{{x}}<br>%{{y:,.2f}} MJ<extra></extra>",
            )
        )
    total_all = sum(left_dict.values())
    total_scope = sum(right_dict.values())
    fig.add_annotation(x=cats[0], y=total_all,  text=f"{us2(total_all)} MJ",  showarrow=False, yshift=10, font=dict(size=12))
    fig.add_annotation(x=cats[1], y=total_scope, text=f"{us2(total_scope)} MJ", showarrow=False, yshift=10, font=dict(size=12))

    # arrows + % per layer
    cum_left = 0.0
    cum_right = 0.0
    for key, label in stack_layers:
        layer_left = float(left_dict.get(key, 0.0))
        layer_right = float(right_dict.get(key, 0.0))
        if layer_left <= 0.0 and layer_right <= 0.0:
            cum_left += layer_left; cum_right += layer_right
            continue
        y_center_left = cum_left + (layer_left / 2.0)
        y_center_right = cum_right + (layer_right / 2.0)

        fig.add_trace(
            go.Scatter(x=cats, y=[y_center_left, y_center_right], mode="lines",
                       line=dict(dash="dot", width=2), hoverinfo="skip", showlegend=False)
        )
        fig.add_annotation(x=cats[1], y=y_center_right, ax=cats[0], ay=y_center_left,
                           xref="x", yref="y", axref="x", ayref="y", text="", showarrow=True,
                           arrowhead=3, arrowsize=1.2, arrowwidth=2, arrowcolor="rgba(0,0,0,0.65)")
        pct = (layer_right / layer_left) * 100.0 if layer_left > 0 else 100.0
        pct = max(min(pct, 100.0), 0.0)
        y_mid = 0.5 * (y_center_left + y_center_right)
        fig.add_annotation(xref="paper", yref="y", x=0.5, y=y_mid, text=f"{pct:.0f}%", showarrow=False,
                           font=dict(size=11, color="#374151"),
                           bgcolor="rgba(255,255,255,0.65)", bordercolor="rgba(0,0,0,0)", borderpad=1)
        cum_left += layer_left; cum_right += layer_right

    fig.update_layout(
        title=title, barmode="stack", xaxis_title="", yaxis_title="Energy [MJ]", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        margin=dict(l=40, r=20, t=50, b=20), bargap=0.35,
    )
    return fig

# Segment stacks
for i, seg in enumerate(segments):
    voy, brt, elec = _segment_energies_MJ(seg, LCV_HSFO, LCV_LFO, LCV_MGO, LCV_BIO, LCV_RFNBO)
    all_seg = {k: voy[k] + brt[k] for k in FUEL_KEYS}
    all_seg["ELEC"] = elec
    in_scope_seg = scoped_per_segment[i]
    title = f"Segment {i+1}: {seg['type']}"
    fig_seg = stack_two_bars(title, all_seg, in_scope_seg)
    st.plotly_chart(fig_seg, use_container_width=True)

st.caption("Arrows show the in-scope take from each layer; labels show the % retained in-scope per fuel. OPS appears only on EU-berth and is always 100% in-scope.")

# Final combined stack
st.markdown('<h2 style="margin:0 0 .25rem 0;">Final combined stack (global prioritized allocation for Extra-EU voyages)</h2>', unsafe_allow_html=True)
combined_all = energies_full
combined_scope = scoped_combined
fig_comb = stack_two_bars("All segments — Combined", combined_all, combined_scope)
st.plotly_chart(fig_comb, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# GHG Intensity Plot (uses COMBINED attained intensity)
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
fig.add_trace(go.Scatter(x=years, y=actual_series, name="Attained GHG (combined)",
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
# Optimizer support — aggregate tons across segments
# ──────────────────────────────────────────────────────────────────────────────
def aggregate_tons_from_segments(segments: List[Dict[str,Any]]) -> Dict[str, Dict[str, float]]:
    intra_voy_t = {k:0.0 for k in FUEL_KEYS}
    extra_voy_t = {k:0.0 for k in FUEL_KEYS}
    berth_t     = {k:0.0 for k in FUEL_KEYS}
    total_bio_t = 0.0
    for seg in segments:
        m = seg.get("masses_t", {})
        if seg["type"] == SEG_INTRA_VOY:
            for k in FUEL_KEYS: intra_voy_t[k] += float(m.get(k,0.0))
        elif seg["type"] in (SEG_IE_VOY, SEG_EI_VOY):
            for k in FUEL_KEYS: extra_voy_t[k] += float(m.get(k,0.0))
        else:
            for k in FUEL_KEYS: berth_t[k] += float(m.get(k,0.0))
        total_bio_t += float(m.get("BIO", 0.0))
    return {"intra_voy_t": intra_voy_t, "extra_voy_t": extra_voy_t, "berth_t": berth_t, "bio_total_t": total_bio_t}

agg_tons = aggregate_tons_from_segments(segments)

def combined_scope_from_totals(intra_voy_t: Dict[str,float],
                               extra_voy_t: Dict[str,float],
                               berth_t: Dict[str,float],
                               elec_MJ: float,
                               LCVs: Dict[str,float],
                               wtw: Dict[str,float]) -> Tuple[Dict[str,float], float, float, float]:
    """Return (in_scope_final, E_scope_MJ, num_phys, E_rfnbo_scope) from aggregated tons."""
    # energies
    toMJ = lambda tons, lcv: compute_energy_MJ(float(tons), float(lcv))
    # voyage (intra)
    intra_MJ = {k: toMJ(intra_voy_t.get(k,0.0), LCVs[k]) for k in FUEL_KEYS}
    # extra voyages
    extra_MJ = {k: toMJ(extra_voy_t.get(k,0.0), LCVs[k]) for k in FUEL_KEYS}
    # berth
    berth_MJ = {k: toMJ(berth_t.get(k,0.0), LCVs[k]) for k in FUEL_KEYS}

    # Scoped for extra (global pool with berth + elec)
    scoped_extra_global = scoped_energies_extra_eu(extra_MJ, berth_MJ, elec_MJ, wtw)
    # add intra 100%
    in_scope_final = {k: scoped_extra_global.get(k,0.0) + intra_MJ.get(k,0.0) for k in FUEL_KEYS}
    in_scope_final["ELEC"] = scoped_extra_global.get("ELEC", 0.0)

    E_scope = sum(in_scope_final.values())
    num_phys = sum(in_scope_final.get(k,0.0) * wtw.get(k,0.0) for k in wtw.keys())
    E_rfnbo = in_scope_final.get("RFNBO", 0.0)
    return in_scope_final, E_scope, num_phys, E_rfnbo

# Candidate penalty recompute using totals
def penalty_usd_with_totals_for_year(year_idx: int,
                                     intra_voy_t: Dict[str,float],
                                     extra_voy_t: Dict[str,float],
                                     berth_t: Dict[str,float]) -> Tuple[float, float]:
    year = years[year_idx]
    g_target = limit_series[year_idx]
    in_scope_x, E_scope_x, num_phys_x, E_rfnbo_x = combined_scope_from_totals(
        intra_voy_t, extra_voy_t, berth_t, combo["elec_MJ"], LCVs, wtw
    )
    if E_scope_x <= 0:
        return 0.0, 0.0
    r = 2.0 if year <= 2033 else 1.0
    den_rwd_x = E_scope_x + (r - 1.0) * E_rfnbo_x
    g_att_x = (num_phys_x / den_rwd_x) if den_rwd_x > 0 else 0.0

    CB_g_x = (g_target - g_att_x) * E_scope_x
    CB_t_raw_x = CB_g_x / 1e6
    cb_eff_x = CB_t_raw_x + carry_in_list[year_idx]

    # pooling/banking (same caps)
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

def masses_after_shift_generic_over_totals(fuel: str, x_decrease_t: float,
                                           intra_voy_t: Dict[str,float],
                                           extra_voy_t: Dict[str,float],
                                           berth_t: Dict[str,float]) -> Tuple[Dict[str,float], Dict[str,float], Dict[str,float], float]:
    """
    Reduce `fuel` mass by x (t) across voyages first (extra+intra), then berth; add BIO mass scaled by LCV ratio to keep MJ constant.
    BIO added to berth first, then voyage. Returns new (intra_voy_t, extra_voy_t, berth_t, bio_added_t)
    """
    x = max(0.0, float(x_decrease_t))

    # Available
    avail_voy = intra_voy_t.get(fuel,0.0) + extra_voy_t.get(fuel,0.0)
    avail_berth = berth_t.get(fuel,0.0)
    x = min(x, avail_voy + avail_berth)

    # Decrease on voyages first
    take_v = min(x, avail_voy)
    rem = x - take_v

    # remove from extra_voy first (arbitrary split: remove proportionally to current shares could be added)
    ex = extra_voy_t.get(fuel,0.0)
    inr = intra_voy_t.get(fuel,0.0)
    if ex + inr > 0:
        # proportional removal on voyages
        if ex > 0:
            rem_ex = min(take_v * (ex/(ex+inr)), ex)
        else:
            rem_ex = 0.0
        rem_in = take_v - rem_ex
        extra_voy_t[fuel] = max(0.0, ex - rem_ex)
        intra_voy_t[fuel] = max(0.0, inr - rem_in)
    # remove remainder from berth
    if rem > 0:
        berth_t[fuel] = max(0.0, berth_t.get(fuel,0.0) - rem)

    # BIO to add (energy neutrality): x * LCV_fuel / LCV_BIO
    LCV_SEL = LCVs[fuel]
    bio_added_t = (x * LCV_SEL / LCVs["BIO"]) if LCVs["BIO"] > 0 else 0.0

    # Add BIO to berth first, then voyage (we put to berth entirely; if desired, you can split)
    berth_t["BIO"] = berth_t.get("BIO",0.0) + bio_added_t

    return intra_voy_t, extra_voy_t, berth_t, bio_added_t

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

# per-step fixed multiplier store
fixed_multiplier_by_step = {}  # step_idx -> locked penalty multiplier for that step

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

    # Penalty multiplier — constant within each regulatory step
    if final_bal < 0:
        step_idx = _step_of_year(year)
        if step_idx not in fixed_multiplier_by_step:
            start_count = max(int(consecutive_deficit_years_seed), 1)
            fixed_multiplier_by_step[step_idx] = 1.0 + (start_count - 1) * 0.10
        multiplier_y = fixed_multiplier_by_step[step_idx]
    else:
        multiplier_y = 1.0

    # € → USD values
    if final_bal > 0:
        # credits use €/tCO2e input → convert to €/VLSFO-eq t via factor
        credit_val = euros_from_tco2e(final_bal, g_att,
                                      (credit_per_tco2e * st.session_state["factor_vlsfo_per_tco2e"]
                                       if st.session_state["factor_vlsfo_per_tco2e"]>0 else 0.0))
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
bio_mass_total_t_base = agg_tons["bio_total_t"]
bio_premium_cost_usd_base = bio_mass_total_t_base * bio_premium_usd_per_t
bio_premium_cost_usd_col = [bio_premium_cost_usd_base] * len(years)

# Pooling cost column (USD)
pooling_cost_usd_col = [
    pool_applied[i] * pooling_price_eur_per_tco2e * eur_usd_fx
    for i in range(len(years))
]

# Updated Total cost includes pooling
total_cost_usd_col = [
    penalties_usd[i] + bio_premium_cost_usd_col[i] + pooling_cost_usd_col[i]
    for i in range(len(years))
]

# ──────────────────────────────────────────────────────────────────────────────
# Optimizer (fossil → BIO; berth-first BIO add; voyage-first fossil cut)
# ──────────────────────────────────────────────────────────────────────────────
dec_opt_list, bio_inc_opt_list = [], []

# Prepare base aggregated tons
base_intra_voy_t = agg_tons["intra_voy_t"].copy()
base_extra_voy_t = agg_tons["extra_voy_t"].copy()
base_berth_t     = agg_tons["berth_t"].copy()

total_avail = base_intra_voy_t.get(selected_fuel_for_opt,0.0) + base_extra_voy_t.get(selected_fuel_for_opt,0.0) + base_berth_t.get(selected_fuel_for_opt,0.0)

for i in range(len(years)):
    if total_avail <= 0 or LCVs["BIO"] <= 0:
        dec_opt_list.append(0.0)
        bio_inc_opt_list.append(0.0)
        continue

    steps_coarse = 60
    x_max = total_avail
    best_x, best_cost = 0.0, float("inf")

    # Coarse scan
    for s in range(steps_coarse + 1):
        x = x_max * s / steps_coarse
        intra_t = base_intra_voy_t.copy()
        extra_t = base_extra_voy_t.copy()
        berth_t = base_berth_t.copy()
        intra_t, extra_t, berth_t, bio_added_t = masses_after_shift_generic_over_totals(
            selected_fuel_for_opt, x, intra_t, extra_t, berth_t
        )
        penalty_usd_x, _ = penalty_usd_with_totals_for_year(i, intra_t, extra_t, berth_t)
        new_bio_total_t = (intra_t.get("BIO",0.0) + extra_t.get("BIO",0.0) + berth_t.get("BIO",0.0))
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
        intra_t = base_intra_voy_t.copy()
        extra_t = base_extra_voy_t.copy()
        berth_t = base_berth_t.copy()
        intra_t, extra_t, berth_t, bio_added_t = masses_after_shift_generic_over_totals(
            selected_fuel_for_opt, x, intra_t, extra_t, berth_t
        )
        penalty_usd_x, _ = penalty_usd_with_totals_for_year(i, intra_t, extra_t, berth_t)
        new_bio_total_t = (intra_t.get("BIO",0.0) + extra_t.get("BIO",0.0) + berth_t.get("BIO",0.0))
        total_cost_x = penalty_usd_x + new_bio_total_t * bio_premium_usd_per_t
        if total_cost_x < best_cost:
            best_cost, best_x = total_cost_x, x

    dec_opt = best_x
    bio_inc_opt = dec_opt * (LCVs[selected_fuel_for_opt] / LCVs["BIO"]) if LCVs["BIO"] > 0 else 0.0
    dec_opt_list.append(dec_opt)
    bio_inc_opt_list.append(bio_inc_opt)

# Optimized-cost recomputation per year (pooling cost unchanged)
penalty_usd_opt_col = []
bio_premium_cost_usd_opt_col = []
total_cost_usd_opt_col = []

for i in range(len(years)):
    x_opt = dec_opt_list[i]
    if x_opt <= 0.0 or LCVs["BIO"] <= 0.0:
        penalty_usd_opt = penalties_usd[i]
        bio_premium_usd_opt = bio_premium_cost_usd_col[i]
    else:
        # Apply optimal decrease to totals
        intra_t = base_intra_voy_t.copy()
        extra_t = base_extra_voy_t.copy()
        berth_t = base_berth_t.copy()
        intra_t, extra_t, berth_t, bio_added_t = masses_after_shift_generic_over_totals(
            selected_fuel_for_opt, x_opt, intra_t, extra_t, berth_t
        )
        penalty_usd_opt, _ = penalty_usd_with_totals_for_year(i, intra_t, extra_t, berth_t)
        new_bio_total_t_opt = (intra_t.get("BIO",0.0) + extra_t.get("BIO",0.0) + berth_t.get("BIO",0.0))
        bio_premium_usd_opt = new_bio_total_t_opt * bio_premium_usd_per_t

    penalty_usd_opt_col.append(penalty_usd_opt)
    bio_premium_cost_usd_opt_col.append(bio_premium_usd_opt)
    total_cost_usd_opt_col.append(
        penalty_usd_opt + bio_premium_usd_opt + pooling_cost_usd_col[i]
    )

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

        "Pooling_Cost_USD": pooling_cost_usd_col,

        "Penalty_USD": penalties_usd,
        "Credit_USD": credits_usd,
        "BIO Premium Cost_USD": bio_premium_cost_usd_col,
        "Total_Cost_USD": total_cost_usd_col,

        decrease_col_name: dec_opt_list,
        "BIO_Increase(t)_For_Opt_Cost": bio_inc_opt_list,

        "Penalty_USD_Opt": penalty_usd_opt_col,
        "BIO Premium Cost_USD_Opt": bio_premium_cost_usd_opt_col,
        "Total_Cost_USD_Opt": total_cost_usd_opt_col,
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
