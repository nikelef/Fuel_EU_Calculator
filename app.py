# app.py — FuelEU Maritime Calculator (with voyage segments, prioritized per cross-border segments, pooled final from segment sums)
# Updates per request (2025-10-23):
#   1) Per-segment toggle “Apply prioritized allocation” now applies to ALL fuels (RFNBO, BIO, HSFO, LFO, MGO)
#      by ascending WtW for the two cross-border types: "EU→non-EU voyage" and "non-EU→EU voyage".
#      For these segments:
#         • Toggle ON  → fill the 50% in-scope POOL (half of total segment energy) by WtW priority (lowest first),
#                        taking from each fuel up to its available energy until the pool is full.
#         • Toggle OFF → simple 50% of each fuel (classic).
#      Intra-EU = 100%. EU at-berth = 100% + OPS electricity (MJ) at 100% scope.
#   2) The “Combined energy (All segments)”:
#         • The "All energy" and "In-scope energy" stacks are computed by SUMMING the corresponding per-segment
#           All and In-scope energies (including OPS only from EU-berth segments).
#         • The Combined "In-scope energy" is then used to compute the FINAL attained GHG intensity (with RFNBO
#           reward per year) and all compliance results & penalties. No separate global pooling is applied.
#   3) All original functionality kept: banking, pooling, optimizer (note: optimizer still evaluates with a pooled
#      allocator approximation for candidate mixes; main results/plot use the combined per-segment in-scope sums).
#
# Currency change (this version):
#   • All prices, costs, and outputs are in EUR. Removed USD and FX.
#   • Renamed *_USD columns/variables to *_EUR. Added bio_premium_eur_per_t input.

from __future__ import annotations
import json, os
from typing import Dict, Any, Tuple, List

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
    Convert a tCO2e amount into EUR using an equivalent VLSFO-tonne price and the attained intensity.
    """
    if balance_tco2e_positive <= 0 or price_eur_per_vlsfo_t <= 0 or g_attained <= 0:
        return 0.0
    tco2e_per_vlsfot = (g_attained * 41_000.0) / 1_000_000.0  # tCO2e per VLSFO-eq tonne at attained intensity
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
# Allocators
# ──────────────────────────────────────────────────────────────────────────────
def scoped_energies_extra_eu(energies_fuel_voyage: Dict[str, float],
                             energies_fuel_berth: Dict[str, float],
                             elec_MJ: float,
                             wtw: Dict[str, float]) -> Dict[str, float]:
    """
    (Kept for optimizer evaluation)  pooled allocator with berth-100% guarantee:
      • Pool = 100% berth + 50% of total voyage (fuels only). ELEC always 100%.
      • Fill by WtW priority: renewables first (berth→voy up to spare after reserving berth fossils),
        then 100% fossil berth, then 50% fossil voyage.
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

    ren_sorted = sorted(["RFNBO","BIO"], key=lambda f: wtw.get(f, float("inf")))
    for f in ren_sorted:
        take_b = min(g(energies_fuel_berth, f), remaining)
        if take_b > 0: scoped[f] += take_b; remaining -= take_b
        if remaining <= 0: return scoped
        spare_for_voy_ren = max(0.0, remaining - berth_fossil_total)
        take_v = min(g(energies_fuel_voyage, f), spare_for_voy_ren)
        if take_v > 0: scoped[f] += take_v; remaining -= take_v
        if remaining <= 0: return scoped

    for f in foss_sorted:
        take = min(g(energies_fuel_berth, f), remaining)
        if take > 0: scoped[f] += take; remaining -= take
        if remaining <= 0: return scoped

    for f in foss_sorted:
        half_v = 0.5 * g(energies_fuel_voyage, f)
        if half_v <= 0 or remaining <= 0: continue
        take = min(half_v, remaining)
        scoped[f] += take; remaining -= take
        if remaining <= 0: return scoped

    return scoped

def prioritized_half_scope_all_fuels(energies_voy: Dict[str, float],
                                     wtw: Dict[str, float]) -> Dict[str, float]:
    """
    New per-segment allocator for cross-border segments when toggle is ON:
      • POOL = 50% of TOTAL segment energy (fuels only).
      • Fill the pool by ascending WtW across ALL fuels (RFNBO, BIO, HSFO, LFO, MGO),
        taking up to each fuel's available energy until pool is full.
      • No ELEC in voyage segments.
    """
    pool = 0.5 * sum(energies_voy.values())
    result = {k: 0.0 for k in energies_voy.keys()}
    order = sorted(energies_voy.keys(), key=lambda f: wtw.get(f, float("inf")))
    for f in order:
        if pool <= 0: break
        take = min(energies_voy.get(f, 0.0), pool)
        if take > 0:
            result[f] = take
            pool -= take
    return result

# ──────────────────────────────────────────────────────────────────────────────
# Segments
# ──────────────────────────────────────────────────────────────────────────────
SEG_TYPES = [
    "Intra-EU voyage",
    "EU→non-EU voyage",
    "non-EU→EU voyage",
    "EU at-berth (port stay)"
]

def _default_segment() -> Dict[str, Any]:
    return {
        "type": SEG_TYPES[0],
        "HSFO_t": 0.0, "LFO_t": 0.0, "MGO_t": 0.0, "BIO_t": 0.0, "RFNBO_t": 0.0,
        "OPS_kWh": 0.0,
        "prio_on": True  # default ON for cross-border; harmless otherwise
    }

def _ensure_segments_state():
    if "abs_segments" not in st.session_state:
        if "abs_segments" in DEFAULTS and isinstance(DEFAULTS["abs_segments"], list):
            st.session_state["abs_segments"] = DEFAULTS["abs_segments"]
        else:
            st.session_state["abs_segments"] = []

def _segments_totals_masses_and_ops() -> Tuple[Dict[str, Dict[str, float]], float]:
    res = {
        "intra_voy": {k: 0.0 for k in ["HSFO","LFO","MGO","BIO","RFNBO"]},
        "extra_voy": {k: 0.0 for k in ["HSFO","LFO","MGO","BIO","RFNBO"]},
        "eu_berth":  {k: 0.0 for k in ["HSFO","LFO","MGO","BIO","RFNBO"]},
    }
    ops_kwh_total = 0.0
    for seg in st.session_state.get("abs_segments", []):
        t = seg.get("type", SEG_TYPES[0])
        bucket = "intra_voy" if t == "Intra-EU voyage" else ("eu_berth" if t == "EU at-berth (port stay)" else "extra_voy")
        for f in ["HSFO","LFO","MGO","BIO","RFNBO"]:
            res[bucket][f] += float(seg.get(f + "_t", 0.0)) or 0.0
        if t == "EU at-berth (port stay)":
            ops_kwh_total += float(seg.get("OPS_kWh", 0.0)) or 0.0
    return res, ops_kwh_total

def _masses_to_energies(masses: Dict[str, float], LCVs: Dict[str, float]) -> Dict[str, float]:
    return {f: compute_energy_MJ(masses.get(f, 0.0), LCVs.get(f, 0.0)) for f in ["HSFO","LFO","MGO","BIO","RFNBO"]}

# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="FuelEU Maritime — Voyage Segments", layout="wide")
st.title("FuelEU Maritime — Voyage Segments — GHG Intensity & Cost")
st.caption("2025–2050 • Limits from 2020 baseline 91.16 gCO₂e/MJ • WtW • Prices in EUR")

# Sidebar CSS (compact), top metric smaller value text
st.markdown("""
<style>
section[data-testid="stSidebar"] div.block-container{ padding-top:.6rem; padding-bottom:.6rem; }
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{ gap:.6rem; }
section[data-testid="stSidebar"] label{ font-size:.95rem; margin-bottom:.2rem; font-weight:600; }
section[data-testid="stSidebar"] input[type="text"], section[data-testid="stSidebar"] input[type="number"]{ height:2.0rem; min-height:2.0rem; padding:.32rem .55rem; }
.card{ padding:.65rem .75rem; border:1px solid #e5e7eb; border-radius:.6rem; background:#fbfbfb; }
.card h4{ margin:.15rem 0 .4rem 0; font-size:1.0rem; font-weight:800; }
.card .help{ font-size:.86rem; color:#6b7280; margin-top:.1rem; }
hr{ border:none; border-top:1px solid #e5e7eb; margin:.4rem 0; }
[data-testid="stMetricLabel"] { font-size: .95rem !important; font-weight: 800 !important; }
[data-testid="stMetricValue"] { font-size: .80rem !important; font-weight: 700 !important; line-height: 1.05 !important; }
[data-testid="stDataFrame"] div[role="columnheader"],[data-testid="stDataFrame"] div[role="gridcell"]{ padding:2px 6px !important; }
[data-testid="stDataFrame"] { font-size: 0.85rem !important; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    _ensure_segments_state()

    # 1) Segments builder
    st.markdown('<div class="card"><h4>Voyage segments</h4><div class="help">Add voyages and EU at-berth stays one by one. OPS appears only inside EU at-berth. Cross-border segments have a toggle for prioritized allocation (applies to all fuels by ascending WtW).</div>', unsafe_allow_html=True)
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
            # Toggle appears only for cross-border voyages
            if seg["type"] in ("EU→non-EU voyage", "non-EU→EU voyage"):
                seg["prio_on"] = st.checkbox("Apply prioritized allocation", value=bool(seg.get("prio_on", True)), key=f"seg_prio_{i}")
            cA, cB = st.columns(2)
            with cA:
                seg["HSFO_t"]  = float_text_input("HSFO [t]" , seg.get("HSFO_t", 0.0), key=f"seg_hsfo_{i}",  min_value=0.0)
                seg["MGO_t"]   = float_text_input("MGO [t]"  , seg.get("MGO_t",  0.0), key=f"seg_mgo_{i}",   min_value=0.0)
                seg["RFNBO_t"] = float_text_input("RFNBO [t]", seg.get("RFNBO_t",0.0), key=f"seg_rfn_{i}",   min_value=0.0)
            with cB:
                seg["LFO_t"]   = float_text_input("LFO [t]"  , seg.get("LFO_t",  0.0), key=f"seg_lfo_{i}",   min_value=0.0)
                seg["BIO_t"]   = float_text_input("BIO [t]"  , seg.get("BIO_t",  0.0), key=f"seg_bio_{i}",   min_value=0.0)
            # OPS appears only for EU at-berth
            if seg["type"] == "EU at-berth (port stay)":
                seg["OPS_kWh"] = float_text_input("EU OPS electricity (kWh)", seg.get("OPS_kWh", 0.0), key=f"seg_ops_{i}", min_value=0.0)
                st.text_input("Electricity (MJ) (derived)", value=us2(seg["OPS_kWh"]*3.6), disabled=True)
            if st.button("Remove this segment", key=f"seg_remove_{i}"):
                to_remove.append(i)
    if to_remove:
        st.session_state["abs_segments"] = [s for j, s in enumerate(st.session_state["abs_segments"]) if j not in to_remove]
    st.markdown("</div>", unsafe_allow_html=True)

    # 2) Fuel properties
    st.markdown('<div class="card"><h4>Fuel properties</h4>', unsafe_allow_html=True)

    # — LCVs first (MJ/t) —
    st.markdown("**Lower Heating Values (LCV)** [MJ/t]")
    lcv_c1, lcv_c2, lcv_c3 = st.columns(3)
    with lcv_c1:
        LCV_HSFO  = float_text_input("HSFO LCV [MJ/t]" , _get(DEFAULTS, "LCV_HSFO" , 40_200.0), key="LCV_HSFO",  min_value=0.0)
    with lcv_c2:
        LCV_LFO   = float_text_input("LFO LCV [MJ/t]"  , _get(DEFAULTS, "LCV_LFO"  , 42_700.0), key="LCV_LFO",   min_value=0.0)
    with lcv_c3:
        LCV_MGO   = float_text_input("MGO LCV [MJ/t]"  , _get(DEFAULTS, "LCV_MGO"  , 42_700.0), key="LCV_MGO",   min_value=0.0)
    lcv_c4, lcv_c5 = st.columns(2)
    with lcv_c4:
        LCV_BIO   = float_text_input("BIO LCV [MJ/t]"  , _get(DEFAULTS, "LCV_BIO"  , 38_000.0), key="LCV_BIO",   min_value=0.0)
    with lcv_c5:
       LCV_RFNBO = float_text_input("RFNBO LCV [MJ/t]", _get(DEFAULTS, "LCV_RFNBO", 30_000.0), key="LCV_RFNBO", min_value=0.0)

    st.markdown("<hr style='margin:0.35rem 0;'/>", unsafe_allow_html=True)

    # — WtW after (gCO₂e/MJ) —
    st.markdown("**Well-to-Wake (WtW) intensities** [gCO₂e/MJ]")
    wtw_c1, wtw_c2, wtw_c3 = st.columns(3)
    with wtw_c1:
        WtW_HSFO  = float_text_input("HSFO WtW [g/MJ]" , _get(DEFAULTS, "WtW_HSFO" , 92.78),  key="WtW_HSFO",  min_value=0.0)
    with wtw_c2:
        WtW_LFO   = float_text_input("LFO WtW [g/MJ]"  , _get(DEFAULTS, "WtW_LFO"  , 92.00),  key="WtW_LFO",   min_value=0.0)
    with wtw_c3:
       WtW_MGO   = float_text_input("MGO WtW [g/MJ]"  , _get(DEFAULTS, "WtW_MGO"  , 93.93),  key="WtW_MGO",   min_value=0.0)
    wtw_c4, wtw_c5 = st.columns(2)
    with wtw_c4:
       WtW_BIO   = float_text_input("BIO WtW [g/MJ]"  , _get(DEFAULTS, "WtW_BIO"  , 70.00),  key="WtW_BIO",   min_value=0.0)
    with wtw_c5:
       WtW_RFNBO = float_text_input("RFNBO WtW [g/MJ]", _get(DEFAULTS, "WtW_RFNBO", 20.00),  key="WtW_RFNBO", min_value=0.0)
    st.markdown("</div>", unsafe_allow_html=True)

    # 4) Other + Optimizer
    st.markdown('<div class="card"><h4>Other settings</h4>', unsafe_allow_html=True)
    consecutive_deficit_years_seed = int(st.number_input("Consecutive deficit years (seed)", min_value=1, value=int(_get(DEFAULTS, "consecutive_deficit_years", 1)), step=1))
    opt_fuels = ["HSFO", "LFO", "MGO"]
    try:
        _idx = opt_fuels.index(_get(DEFAULTS, "opt_reduce_fuel", "HSFO"))
    except ValueError:
        _idx = 0
    selected_fuel_for_opt = st.selectbox("Fuel to reduce (for optimization)", opt_fuels, index=_idx)
    st.markdown("</div>", unsafe_allow_html=True)

    # 3) Market prices  (ALL in EUR)
    st.markdown('<div class="card"><h4>Market prices</h4>', unsafe_allow_html=True)
    credit_per_tco2e = float_text_input(
        "Credit price €/tCO₂e",
        _get(DEFAULTS, "credit_per_tco2e", 200.0),
        key="credit_per_tco2e_str", min_value=0.0
    )
    penalty_price_eur_per_vlsfo_t = float_text_input(
        "Penalty price €/VLSFO-eq t",
        _get(DEFAULTS, "penalty_price_eur_per_vlsfo_t", 2_400.0),
        key="penalty_per_vlsfo_t_str", min_value=0.0
    )
    bio_premium_label = f"Premium BIO vs {selected_fuel_for_opt} [EUR/ton]"
    bio_premium_eur_per_t = float_text_input(
        bio_premium_label,
        _get(DEFAULTS, "bio_premium_eur_per_t", _get(DEFAULTS, "bio_premium_usd_per_t", 0.0)),
        key="bio_premium_eur_per_t", min_value=0.0
    )
    st.markdown("</div>", unsafe_allow_html=True)



    # 5) Banking & Pooling
    st.markdown('<div class="card"><h4>Banking & Pooling (tCO₂e)</h4>', unsafe_allow_html=True)
    pooling_price_eur_per_tco2e = float_text_input("Pooling price €/tCO₂e", _get(DEFAULTS, "pooling_price_eur_per_tco2e", 200.0), key="pooling_price_eur_per_tco2e", min_value=0.0)
    pooling_tco2e_input = float_text_input_signed("Pooling [tCO₂e]: + uptake, − provide", _get(DEFAULTS, "pooling_tco2e", 0.0), key="POOL_T")
    pooling_start_year = st.selectbox("Pooling starts from year", YEARS, index=YEARS.index(int(_get(DEFAULTS, "pooling_start_year", YEARS[0]))))
    banking_tco2e_input = float_text_input("Banking to next year [tCO₂e]", _get(DEFAULTS, "banking_tco2e", 0.0), key="BANK_T", min_value=0.0)
    banking_start_year = st.selectbox("Banking starts from year", YEARS, index=YEARS.index(int(_get(DEFAULTS, "banking_start_year", YEARS[0]))))
    st.markdown("</div>", unsafe_allow_html=True)

    # 6) Save
    if st.button("💾 Save current inputs as defaults"):
        defaults_to_save = {
            # Prices/settings (EUR-only)
            "credit_per_tco2e": credit_per_tco2e,
            "penalty_price_eur_per_vlsfo_t": penalty_price_eur_per_vlsfo_t,
            "bio_premium_eur_per_t": bio_premium_eur_per_t,
            "pooling_price_eur_per_tco2e": pooling_price_eur_per_tco2e,
            "banking_tco2e": banking_tco2e_input,
            "pooling_tco2e": pooling_tco2e_input,
            "pooling_start_year": int(pooling_start_year),
            "banking_start_year": int(banking_start_year),
            "consecutive_deficit_years": consecutive_deficit_years_seed,
            "opt_reduce_fuel": selected_fuel_for_opt,
            # Fuel props
            "LCV_HSFO": LCV_HSFO, "LCV_LFO": LCV_LFO, "LCV_MGO": LCV_MGO, "LCV_BIO": LCV_BIO, "LCV_RFNBO": LCV_RFNBO,
            "WtW_HSFO": WtW_HSFO, "WtW_LFO": WtW_LFO, "WtW_MGO": WtW_MGO, "WtW_BIO": WtW_BIO, "WtW_RFNBO": WtW_RFNBO,
            # Segments
            "abs_segments": st.session_state.get("abs_segments", []),
        }
        try:
            with open(DEFAULTS_PATH, "w", encoding="utf-8") as f:
                json.dump(defaults_to_save, f, indent=2)
            st.success("Defaults saved.")
        except Exception as e:
            st.error(f"Could not save defaults: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# Live LCV/WtW dicts (used throughout)
# ──────────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────────
# Bucket totals (kept for optimizer & CSV endpoints)
# ──────────────────────────────────────────────────────────────────────────────
totals_mass, ops_kwh_total = _segments_totals_masses_and_ops()
ELEC_MJ_input = ops_kwh_total * 3.6

# Energies by buckets (for optimizer use; kept)
energies_extra_voy = _masses_to_energies(totals_mass["extra_voy"], LCVs_now)
energies_eu_berth  = _masses_to_energies(totals_mass["eu_berth"],  LCVs_now)
energies_intra_voy = _masses_to_energies(totals_mass["intra_voy"], LCVs_now)

# ──────────────────────────────────────────────────────────────────────────────
# Per-segment rendering + build combined sums (ALL and IN-SCOPE from segments)
# ──────────────────────────────────────────────────────────────────────────────
COLORS = {  # original palette
    "ELEC":  "#FACC15",
    "RFNBO": "#86EFAC",
    "BIO":   "#065F46",
    "MGO":   "#93C5FD",
    "LFO":   "#2563EB",
    "HSFO":  "#1E3A8A",
}
FUELS = ["RFNBO","BIO","HSFO","LFO","MGO"]

def _segment_energy_mj(seg: Dict[str, Any]) -> Dict[str,float]:
    return {
        "HSFO": compute_energy_MJ(seg.get("HSFO_t",0.0), LCV_HSFO),
        "LFO":  compute_energy_MJ(seg.get("LFO_t", 0.0), LCV_LFO),
        "MGO":  compute_energy_MJ(seg.get("MGO_t", 0.0), LCV_MGO),
        "BIO":  compute_energy_MJ(seg.get("BIO_t", 0.0), LCV_BIO),
        "RFNBO":compute_energy_MJ(seg.get("RFNBO_t",0.0), LCV_RFNBO),
    }

def _segment_scope_with_toggle(seg: Dict[str,Any], energies_all: Dict[str,float]) -> Tuple[Dict[str,float], float]:
    """
    Returns (in_scope_fuel_MJ_dict, elec_MJ_segment).
    • Intra-EU: 100%.
    • EU-berth: 100% + ELEC (kWh→MJ).
    • Cross-border: toggle OFF → 50% each fuel; toggle ON → prioritized_half_scope_all_fuels().
    """
    t = seg.get("type", SEG_TYPES[0])
    if t == "Intra-EU voyage":
        return dict(energies_all), 0.0
    if t == "EU at-berth (port stay)":
        return dict(energies_all), float(seg.get("OPS_kWh",0.0))*3.6
    # Cross-border
    prio_on = bool(seg.get("prio_on", True))
    if prio_on:
        scoped = prioritized_half_scope_all_fuels(energies_all, wtw)
        return scoped, 0.0
    else:
        return {k: 0.5*energies_all[k] for k in energies_all.keys()}, 0.0

def _stack_with_arrows(title: str, left_vals: Dict[str,float], right_vals: Dict[str,float], show_elec: bool):
    categories = ["All", "In-scope"]
    fuels_sorted = sorted(FUELS, key=lambda f: wtw.get(f, float("inf")))
    stack_layers = ([("ELEC","ELEC (OPS)")] if show_elec else []) + [(f, f) for f in fuels_sorted]

    fig = go.Figure()
    for key, label in stack_layers:
        fig.add_trace(
            go.Bar(
                x=categories,
                y=[left_vals.get(key, 0.0), right_vals.get(key, 0.0)],
                name=label,
                marker_color=COLORS.get(key, None),
                hovertemplate=f"{label}<br>%{{x}}<br>%{{y:,.2f}} MJ<extra></extra>",
            )
        )
    total_all = sum(left_vals.get(k,0.0) for k,_ in stack_layers)
    total_scope = sum(right_vals.get(k,0.0) for k,_ in stack_layers)
    fig.add_annotation(x=categories[0], y=total_all,  text=f"{us2(total_all)} MJ", showarrow=False, yshift=10, font=dict(size=12))
    fig.add_annotation(x=categories[1], y=total_scope, text=f"{us2(total_scope)} MJ", showarrow=False, yshift=10, font=dict(size=12))

    # arrows + % retained
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
        fig.add_trace(go.Scatter(x=categories, y=[y_center_left, y_center_right], mode="lines",
                                 line=dict(dash="dot", width=2), hoverinfo="skip", showlegend=False))
        fig.add_annotation(x=categories[1], y=y_center_right, ax=categories[0], ay=y_center_left,
                           xref="x", yref="y", axref="x", ayref="y", text="", showarrow=True,
                           arrowhead=3, arrowsize=1.2, arrowwidth=2, arrowcolor="rgba(0,0,0,0.65)")
        pct = (layer_right / layer_left * 100.0) if layer_left > 0 else 100.0
        pct = max(min(pct, 100.0), 0.0)
        y_mid = 0.5 * (y_center_left + y_center_right)
        fig.add_annotation(xref="paper", yref="y", x=0.5, y=y_mid, text=f"{pct:.0f}%", showarrow=False,
                           font=dict(size=11, color="#374151"),
                           bgcolor="rgba(255,255,255,0.65)", bordercolor="rgba(0,0,0,0)", borderpad=1)
        cum_left += layer_left; cum_right += layer_right

    fig.update_layout(
        title=dict(text=title, x=0.02, y=0.95, font=dict(size=13)),
        barmode="stack", xaxis_title="", yaxis_title="Energy [MJ]", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        margin=dict(l=40, r=20, t=50, b=20), bargap=0.35, height=260,
    )
    st.plotly_chart(fig, use_container_width=True)

# Build combined sums while rendering per-segment stacks
combined_all = {"ELEC":0.0, "RFNBO":0.0, "BIO":0.0, "HSFO":0.0, "LFO":0.0, "MGO":0.0}
combined_scope = {"ELEC":0.0, "RFNBO":0.0, "BIO":0.0, "HSFO":0.0, "LFO":0.0, "MGO":0.0}

st.markdown("### Per-segment energy (All vs In-scope)")
if not st.session_state["abs_segments"]:
    st.info("No segments yet. Add segments from the left sidebar.")
else:
    for i, seg in enumerate(st.session_state["abs_segments"]):
        energies_all = _segment_energy_mj(seg)
        energies_scope, elec_mj_seg = _segment_scope_with_toggle(seg, energies_all)
        left_vals = dict(energies_all)
        right_vals = dict(energies_scope)
        if seg["type"] == "EU at-berth (port stay)":
            left_vals["ELEC"] = elec_mj_seg
            right_vals["ELEC"] = elec_mj_seg
            show_elec = True
        else:
            show_elec = False
        _stack_with_arrows(f"Segment {i+1}: {seg.get('type','')}", left_vals, right_vals, show_elec)

        # accumulate to combined sums
        combined_all["ELEC"]  += left_vals.get("ELEC", 0.0)
        combined_scope["ELEC"]+= right_vals.get("ELEC", 0.0)
        for f in FUELS:
            combined_all[f]   += left_vals.get(f, 0.0)
            combined_scope[f] += right_vals.get(f, 0.0)

# ──────────────────────────────────────────────────────────────────────────────
# Combined (from segment sums) → metrics, derived prices, stacks, intensity
# ──────────────────────────────────────────────────────────────────────────────
E_total_MJ = sum(combined_all.values())
E_scope_MJ = sum(combined_scope.values())

# Attained GHG of combined in-scope mix
num_phys = sum(combined_scope.get(k,0.0) * wtw.get(k,0.0) for k in ["HSFO","LFO","MGO","BIO","RFNBO","ELEC"])
den_phys = E_scope_MJ
E_rfnbo_scope = combined_scope.get("RFNBO", 0.0)
def attained_intensity_for_year(y: int) -> float:
    if den_phys <= 0: return 0.0
    r = 2.0 if y <= 2033 else 1.0
    den_rwd = den_phys + (r - 1.0) * E_rfnbo_scope
    return num_phys / den_rwd if den_rwd > 0 else 0.0

# Derived price factor (non-zero): use preview r=2
if den_phys > 0:
    den_preview = den_phys + E_rfnbo_scope
    g_preview = num_phys / den_preview if den_preview > 0 else 0.0
else:
    g_preview = 0.0
if g_preview <= 0:
    g_preview = BASELINE_2020_GFI
tco2e_per_vlsfo_t = (g_preview * 41_000.0) / 1_000_000.0

# Headline metrics (smaller numbers)
st.subheader("Energy breakdown (MJ)")
cA, cB, cC, cD, cE, cF, cG, cH = st.columns(8)
with cA: st.metric("Total energy (all)", f"{us2(E_total_MJ)} MJ")
with cB: st.metric("In-scope energy", f"{us2(E_scope_MJ)} MJ")
with cC: st.metric("Fossil — all", f"{us2(combined_all['HSFO'] + combined_all['LFO'] + combined_all['MGO'])} MJ")
with cD: st.metric("BIO — all", f"{us2(combined_all['BIO'])} MJ")
with cE: st.metric("RFNBO — all", f"{us2(combined_all['RFNBO'])} MJ")
with cF: st.metric("Fossil — in scope", f"{us2(combined_scope['HSFO']+combined_scope['LFO']+combined_scope['MGO'])} MJ")
with cG: st.metric("BIO — in scope", f"{us2(combined_scope['BIO'])} MJ")
with cH: st.metric("RFNBO — in scope", f"{us2(combined_scope['RFNBO'])} MJ")

# Derived prices card (EUR)
with st.sidebar:
    st.markdown('<div class="card"><h4>Derived prices</h4>', unsafe_allow_html=True)
    st.text_input("Credit price €/VLSFO-eq t (at current mix)", value=us2(credit_per_tco2e * tco2e_per_vlsfo_t), disabled=True)
    st.text_input("Penalty price €/tCO₂e (at current mix)", value=us2((penalty_price_eur_per_vlsfo_t / tco2e_per_vlsfo_t) if tco2e_per_vlsfo_t>0 else 0.0), disabled=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Combined stacks from segment sums
st.markdown("### Combined energy (All segments)")
categories = ["All energy", "In-scope energy"]
fuels_sorted_global = sorted(FUELS, key=lambda f: wtw.get(f, float("inf")))
stack_layers_global = [("ELEC", "ELEC (OPS)")] + [(f, f) for f in fuels_sorted_global]

left_vals = {
    "ELEC":  combined_all.get("ELEC", 0.0),
    "RFNBO": combined_all.get("RFNBO", 0.0),
    "BIO":   combined_all.get("BIO",   0.0),
    "HSFO":  combined_all.get("HSFO",  0.0),
    "LFO":   combined_all.get("LFO",   0.0),
    "MGO":   combined_all.get("MGO",   0.0),
}
right_vals = {
    "ELEC":  combined_scope.get("ELEC",  0.0),
    "RFNBO": combined_scope.get("RFNBO", 0.0),
    "BIO":   combined_scope.get("BIO",   0.0),
    "HSFO":  combined_scope.get("HSFO",  0.0),
    "LFO":   combined_scope.get("LFO",   0.0),
    "MGO":   combined_scope.get("MGO",   0.0),
}

fig_stacks = go.Figure()
for key, label in stack_layers_global:
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
for key, label in stack_layers_global:
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
    pct = (layer_right / layer_left * 100.0) if layer_left > 0 else 100.0
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
st.caption("Combined right bar = sum of per-segment in-scope energies (with cross-border prioritized allocation toggle as selected; OPS from EU-berth only).")

# ──────────────────────────────────────────────────────────────────────────────
# GHG Intensity vs. FuelEU Limit (uses combined in-scope)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('### GHG Intensity vs FuelEU Limit (2025–2050)')
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
fig.add_trace(go.Scatter(x=years, y=actual_series, name="Attained GHG (combined in-scope)",
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
# Results — Banking/Pooling + optimizer (kept), using combined in-scope intensity
# ──────────────────────────────────────────────────────────────────────────────
st.header("Results (merged per-year table)")

cb_raw_t, carry_in_list, cb_eff_t = [], [], []
pool_applied, bank_applied = [], []
final_balance_t, penalties_eur, credits_eur, g_att_list = [], [], [], []
carry = 0.0
fixed_multiplier_by_step = {}

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

# Pooling (uptake only fills a deficit; provide only from surplus)
    if year >= int(st.session_state.get("pooling_start_year", _get(DEFAULTS,"pooling_start_year",YEARS[0]))):
       pooling_tco2e_val = parse_us_any(st.session_state.get("POOL_T", _get(DEFAULTS,"pooling_tco2e",0.0)), 0.0)
       if pooling_tco2e_val >= 0:
           # uptake: cap by current deficit (negative cb_eff)
           pre_deficit = max(-cb_eff, 0.0)
           pool_use = min(pooling_tco2e_val, pre_deficit)
       else:
           # provide: cap by current surplus (positive cb_eff)
           provide_abs = abs(pooling_tco2e_val)
           pre_surplus = max(cb_eff, 0.0)
           pool_use = -min(provide_abs, pre_surplus)
    else:
        pool_use = 0.0

    # Banking
    if year >= int(st.session_state.get("banking_start_year", _get(DEFAULTS,"banking_start_year",YEARS[0]))):
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

    # Constant within step multiplier
    if final_bal < 0:
        step_idx = _step_of_year(year)
        if step_idx not in fixed_multiplier_by_step:
            seed = max(int(st.session_state.get("consecutive_deficit_years", _get(DEFAULTS,"consecutive_deficit_years",1))), 1)
            fixed_multiplier_by_step[step_idx] = 1.0 + (seed - 1) * 0.10
        mult = fixed_multiplier_by_step[step_idx]
    else:
        mult = 1.0

    # EUR (no FX)
    penalty_vlsfo = parse_us_any(st.session_state.get("penalty_per_vlsfo_t_str", _get(DEFAULTS,"penalty_price_eur_per_vlsfo_t",2400.0)), 2400.0)
    credit_per_tco2e_val = parse_us_any(st.session_state.get("credit_per_tco2e_str", _get(DEFAULTS,"credit_per_tco2e",200.0)), 200.0)
    if final_bal > 0:
        credit_val = final_bal * credit_per_tco2e_val
        penalty_val = 0.0
    elif final_bal < 0:
        penalty_val = euros_from_tco2e(-final_bal, g_att, penalty_vlsfo) * mult
        credit_val = 0.0
    else:
        credit_val = penalty_val = 0.0

    pool_applied.append(pool_use); bank_applied.append(bank_use)
    final_balance_t.append(final_bal)
    penalties_eur.append(penalty_val); credits_eur.append(credit_val)

# BIO premium & pooling cost series (EUR)
bio_mass_total_t_base = (totals_mass["intra_voy"]["BIO"] + totals_mass["extra_voy"]["BIO"] + totals_mass["eu_berth"]["BIO"])
bio_premium_eur_per_t_val = parse_us_any(st.session_state.get("bio_premium_eur_per_t", _get(DEFAULTS,"bio_premium_eur_per_t", _get(DEFAULTS,"bio_premium_usd_per_t",0.0))), 0.0)
bio_premium_cost_eur_col = [bio_mass_total_t_base * bio_premium_eur_per_t_val] * len(YEARS)
pooling_price_eur_per_tco2e_val = parse_us_any(st.session_state.get("pooling_price_eur_per_tco2e", _get(DEFAULTS,"pooling_price_eur_per_tco2e",200.0)), 200.0)
pooling_cost_eur_col = [pool_applied[i] * pooling_price_eur_per_tco2e_val for i in range(len(YEARS))]
net_total_cost_eur_col = [penalties_eur[i] - credits_eur[i] + bio_premium_cost_eur_col[i] + pooling_cost_eur_col[i] for i in range(len(YEARS))]

# ──────────────────────────────────────────────────────────────────────────────
# Optimizer utilities (kept; pooled allocator approximation)
# ──────────────────────────────────────────────────────────────────────────────
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
ELEC_MJ = ELEC_MJ_input

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

# ---- OPTIMIZER CHANGE 1: expand return values to include final balance and pooling used
def penalty_eur_with_masses_for_year(year_idx: int,
                                     h_v, l_v, m_v, b_v, r_v,
                                     h_b, l_b, m_b, b_b, r_b) -> Tuple[float, float, float, float]:
    year = YEARS[year_idx]
    g_target = LIMITS_DF["Limit_gCO2e_per_MJ"].iloc[year_idx]
    E_scope_x, num_phys_x, E_rfnbo_scope_x = scoped_and_intensity_from_masses(
        h_v, l_v, m_v, b_v, r_v, h_b, l_b, m_b, b_b, r_b, ELEC_MJ, wtw, year
    )
    if E_scope_x <= 0: 
        return 0.0, 0.0, 0.0, 0.0
    r = 2.0 if year <= 2033 else 1.0
    den_rwd_x = E_scope_x + (r - 1.0) * E_rfnbo_scope_x
    g_att_x = (num_phys_x / den_rwd_x) if den_rwd_x > 0 else 0.0

    CB_g_x = (g_target - g_att_x) * E_scope_x
    CB_t_raw_x = CB_g_x / 1e6
    cb_eff_x = CB_t_raw_x + carry_in_list[year_idx]

    if YEARS[year_idx] >= int(pooling_start_year):
        if pooling_tco2e_input >= 0:
        # uptake: cap by current deficit (negative cb_eff_x)
           pre_deficit_x = max(-cb_eff_x, 0.0)
           pool_use_x = min(pooling_tco2e_input, pre_deficit_x)
        else:
        # provide: cap by current surplus (positive cb_eff_x)
           provide_abs = abs(pooling_tco2e_input)
           pre_surplus_x = max(cb_eff_x, 0.0)
           pool_use_x = -min(provide_abs, pre_surplus_x)
    else:
        pool_use_x = 0.0

    if YEARS[year_idx] >= int(banking_start_year):
        pre_surplus = max(cb_eff_x, 0.0)
        requested_bank = max(banking_tco2e_input, 0.0)
        bank_use_x = min(requested_bank, pre_surplus)
    else:
        bank_use_x = 0.0

    final_bal_x = cb_eff_x + pool_use_x - bank_use_x
    if final_bal_x < 0:
        needed = -final_bal_x
        trim_bank = min(needed, bank_use_x); bank_use_x -= trim_bank; needed -= trim_bank
        if needed > 0 and pool_use_x < 0:
            pool_use_x += needed
        final_bal_x = cb_eff_x + pool_use_x - bank_use_x

    if final_bal_x < 0:
        step_idx = _step_of_year(year)
        start_count = max(int(consecutive_deficit_years_seed), 1)
        step_mult = 1.0 + (start_count - 1) * 0.10
        penalty_eur_x = euros_from_tco2e(-final_bal_x, g_att_x, penalty_price_eur_per_vlsfo_t) * step_mult
    else:
        penalty_eur_x = 0.0
    # return: penalty, attained, final balance (tCO2e), pooling applied (tCO2e)
    return penalty_eur_x, g_att_x, final_bal_x, pool_use_x

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
    add_b = min(bio_increase_t, float("inf")); b_b += add_b
    rem_bio = bio_increase_t - add_b
    if rem_bio > 0: b_v += rem_bio
    if fuel == "HSFO": h_v, h_b = s_v, s_b
    elif fuel == "LFO": l_v, l_b = s_v, s_b
    else: m_v, m_b = s_v, s_b
    return h_v, l_v, m_v, b_v, r_v, h_b, l_b, m_b, b_b, r_b

# Optimizer search (coarse + fine)
# ---- OPTIMIZER CHANGE 2: include credits and pooling cost in candidate evaluation
credit_per_tco2e_val_opt = parse_us_any(st.session_state.get("credit_per_tco2e_str", _get(DEFAULTS,"credit_per_tco2e",200.0)), 200.0)

# Optimizer search — finer: dense grid + golden-section, and full cost (penalty − credits + pooling + bio premium)
credit_per_tco2e_val_opt = parse_us_any(
    st.session_state.get("credit_per_tco2e_str", _get(DEFAULTS, "credit_per_tco2e", 200.0)), 200.0
)
penalty_vlsfo_opt = parse_us_any(
    st.session_state.get("penalty_per_vlsfo_t_str", _get(DEFAULTS, "penalty_price_eur_per_vlsfo_t", 2400.0)), 2400.0
)

dec_opt_list, bio_inc_opt_list = [], []
for i in range(len(YEARS)):
    if selected_fuel_for_opt == "HSFO":
        total_avail, LCV_SEL = HSFO_voy_t + HSFO_berth_t, LCV_HSFO
    elif selected_fuel_for_opt == "LFO":
        total_avail, LCV_SEL = LFO_voy_t + LFO_berth_t, LCV_LFO
    else:
        total_avail, LCV_SEL = MGO_voy_t + MGO_berth_t, LCV_MGO

    x_max = total_avail
    if x_max <= 0 or LCV_BIO <= 0:
        dec_opt_list.append(0.0)
        bio_inc_opt_list.append(0.0)
        continue

    def _total_cost_for_x(x: float) -> float:
        # 1) masses after shifting selected fossil → BIO (energy-equivalent)
        masses = masses_after_shift_generic(selected_fuel_for_opt, x)

        # 2) compute scope, numerators (same as helper), then g_att_x
        E_scope_x, num_phys_x, E_rfnbo_scope_x = scoped_and_intensity_from_masses(
            *masses, ELEC_MJ, wtw, YEARS[i]
        )
        if E_scope_x <= 0:
            return 0.0  # no energy → no cost

        r = 2.0 if YEARS[i] <= 2033 else 1.0
        den_rwd_x = E_scope_x + (r - 1.0) * E_rfnbo_scope_x
        g_att_x = (num_phys_x / den_rwd_x) if den_rwd_x > 0 else 0.0

        # 3) compliance balance for this candidate
        g_target = LIMITS_DF["Limit_gCO2e_per_MJ"].iloc[i]
        CB_g_x = (g_target - g_att_x) * E_scope_x
        CB_t_raw_x = CB_g_x / 1e6
        cb_eff_x = CB_t_raw_x + carry_in_list[i]

        # 4) pooling (cap: uptake by deficit, provide by surplus)
        if YEARS[i] >= int(pooling_start_year):
            if pooling_tco2e_input >= 0:
                pre_deficit_x = max(-cb_eff_x, 0.0)
                pool_use_x = min(pooling_tco2e_input, pre_deficit_x)
            else:
                provide_abs = abs(pooling_tco2e_input)
                pre_surplus_x = max(cb_eff_x, 0.0)
                pool_use_x = -min(provide_abs, pre_surplus_x)
        else:
            pool_use_x = 0.0

        # 5) banking (cap by current surplus)
        if YEARS[i] >= int(banking_start_year):
            pre_surplus = max(cb_eff_x, 0.0)
            requested_bank = max(banking_tco2e_input, 0.0)
            bank_use_x = min(requested_bank, pre_surplus)
        else:
            bank_use_x = 0.0

        # 6) safety clamp (cannot end negative if bank/provide were over-applied)
        final_bal_x = cb_eff_x + pool_use_x - bank_use_x
        if final_bal_x < 0:
            needed = -final_bal_x
            trim_bank = min(needed, bank_use_x)
            bank_use_x -= trim_bank
            needed -= trim_bank
            if needed > 0 and pool_use_x < 0:
                pool_use_x += needed
            final_bal_x = cb_eff_x + pool_use_x - bank_use_x

        # 7) penalty / credit
        if final_bal_x < 0:
            step_idx = _step_of_year(YEARS[i])
            start_count = max(int(consecutive_deficit_years_seed), 1)
            step_mult = 1.0 + (start_count - 1) * 0.10
            penalty_eur_x = euros_from_tco2e(-final_bal_x, g_att_x, penalty_vlsfo_opt) * step_mult
            credits_eur_x = 0.0
        else:
            penalty_eur_x = 0.0
            credits_eur_x = final_bal_x * credit_per_tco2e_val_opt

        # 8) pooling & bio premium costs
        pooling_cost_x = pool_use_x * pooling_price_eur_per_tco2e_val
        new_bio_total_t = (masses[3] + masses[8])  # b_v + b_b

        # 9) total cost objective (match Net_Total_Cost logic)
        return penalty_eur_x - credits_eur_x + new_bio_total_t * bio_premium_eur_per_t_val + pooling_cost_x

    # A) dense coarse scan to bracket minimum
    steps_coarse = 200
    best_x, best_cost = 0.0, float("inf")
    for s in range(steps_coarse + 1):
        x = x_max * s / steps_coarse
        c = _total_cost_for_x(x)
        if c < best_cost:
            best_cost, best_x = c, x

    # bracket around best coarse point (±3 bins)
    bin_w = x_max / steps_coarse
    a = max(0.0, best_x - 3 * bin_w)
    b = min(x_max, best_x + 3 * bin_w)

    # B) golden-section refinement on [a, b]
    phi = (5 ** 0.5 - 1) / 2.0  # ≈0.618
    tol = max(x_max * 1e-5, 1e-4)  # tonnes

    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc = _total_cost_for_x(c)
    fd = _total_cost_for_x(d)

    it, max_iter = 0, 120
    while (b - a) > tol and it < max_iter:
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = _total_cost_for_x(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = _total_cost_for_x(d)
        it += 1

    dec_opt = (a + b) / 2.0
    bio_inc_opt = dec_opt * (LCV_SEL / LCV_BIO) if LCV_BIO > 0 else 0.0
    dec_opt_list.append(dec_opt)
    bio_inc_opt_list.append(bio_inc_opt)





# Recompute optimized cost columns (EUR)
# ---- OPTIMIZER CHANGE 3: subtract credits; pooling cost from candidate, not base
penalties_eur_opt_col, bio_premium_cost_eur_opt_col, total_cost_eur_opt_col = [], [], []
for i in range(len(YEARS)):
    x_opt = dec_opt_list[i]
    if x_opt <= 0.0 or LCV_BIO <= 0.0:
        penalties_eur_opt = penalties_eur[i]
        bio_premium_eur_opt = bio_premium_cost_eur_col[i]
        credits_eur_opt = credits_eur[i]
        pooling_cost_eur_opt = pooling_cost_eur_col[i]
        penalties_eur_opt_col.append(penalties_eur_opt)
        bio_premium_cost_eur_opt_col.append(bio_premium_eur_opt)
        total_cost_eur_opt_col.append(penalties_eur_opt - credits_eur_opt + bio_premium_eur_opt + pooling_cost_eur_opt)
    else:
        masses_opt = masses_after_shift_generic(selected_fuel_for_opt, x_opt)
        penalties_eur_opt, _, final_bal_x, pool_use_x = penalty_eur_with_masses_for_year(i, *masses_opt)
        new_bio_total_t_opt = (masses_opt[3] + masses_opt[8])
        bio_premium_eur_opt = new_bio_total_t_opt * bio_premium_eur_per_t_val
        credits_eur_opt = max(final_bal_x, 0.0) * credit_per_tco2e_val_opt
        pooling_cost_eur_opt = pool_use_x * pooling_price_eur_per_tco2e_val

        penalties_eur_opt_col.append(penalties_eur_opt)
        bio_premium_cost_eur_opt_col.append(bio_premium_eur_opt)
        total_cost_eur_opt_col.append(penalties_eur_opt - credits_eur_opt + bio_premium_eur_opt + pooling_cost_eur_opt)

# ──────────────────────────────────────────────────────────────────────────────
# Table
# ──────────────────────────────────────────────────────────────────────────────
decrease_col_name = f"{selected_fuel_for_opt}_decrease(t)_for_Opt_Cost"
emissions_tco2e = num_phys / 1e6  # physical emissions for the in-scope mix (no RFNBO reward)

df_cost = pd.DataFrame({
    "Year": YEARS,
    "Reduction_%": LIMITS_DF["Reduction_%"].tolist(),
    "Limit_gCO2e_per_MJ": LIMITS_DF["Limit_gCO2e_per_MJ"].tolist(),
    "Actual_gCO2e_per_MJ": [attained_intensity_for_year(y) for y in YEARS],
    "Emissions_tCO2e": [emissions_tco2e]*len(YEARS),

    "Compliance_Balance_tCO2e": cb_raw_t,
    "CarryIn_Banked_tCO2e": carry_in_list,
    "Effective_Balance_tCO2e": cb_eff_t,
    "Banked_to_Next_Year_tCO2e": bank_applied,
    "Pooling_tCO2e_Applied": pool_applied,
    "Final_Balance_tCO2e": final_balance_t,

    "Pooling_Cost_EUR": pooling_cost_eur_col,
    "Penalty_EUR": penalties_eur,
    "Credit_EUR": credits_eur,
    "BIO Premium Cost_EUR": bio_premium_cost_eur_col,
    "Net_Total_Cost_EUR": net_total_cost_eur_col,

    decrease_col_name: dec_opt_list,
    "BIO_Increase(t)_For_Opt_Cost": bio_inc_opt_list,
    "Total_Cost_EUR_Opt": total_cost_eur_opt_col,
})

df_fmt = df_cost.copy()
for col in df_fmt.columns:
    if col != "Year": df_fmt[col] = df_fmt[col].apply(us2)
st.dataframe(df_fmt, use_container_width=True)
st.download_button("fueleu_voyage_segments_2025_2050_eur.csv", data=df_cost.to_csv(index=False), file_name="fueleu_results_2025_2050_eur.csv", mime="text/csv")
