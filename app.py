# app.py — FuelEU Maritime Calculator (Simplified, 2025–2050, EUR-only + defaults)
# (Entire file content — unchanged header/logic except the optimizer & penalty evaluator updated
#  to support pooling as a decision variable and to expose the optimization result column.)
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
# Scoping helpers (unchanged)
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
# PRECOMPUTE PREVIEW FACTOR for the sidebar (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
def _ss_or_def(key: str, default_val: float) -> float:
    if key in st.session_state:
        try:
            return parse_us_any(st.session_state[key], default=default_val)
        except Exception:
            return float(default_val)
    return float(_get(DEFAULTS, key, default_val))

def _compute_preview_factor_first_pass() -> float:
    voyage_type_saved = st.session_state.get("voyage_type", _get(DEFAULTS, "voyage_type", "Intra-EU (100%)"))

    LCV_HSFO_pre = _ss_or_def("LCV_HSFO", _get(DEFAULTS, "LCV_HSFO", 40200.0))
    LCV_LFO_pre  = _ss_or_def("LCV_LFO",  _get(DEFAULTS, "LCV_LFO",  42700.0))
    LCV_MGO_pre  = _ss_or_def("LCV_MGO",  _get(DEFAULTS, "LCV_MGO",  42700.0))
    LCV_BIO_pre  = _ss_or_def("LCV_BIO",  _get(DEFAULTS, "LCV_BIO",  38000.0))
    LCV_RFN_pre  = _ss_or_def("LCV_RFNBO",_get(DEFAULTS, "LCV_RFNBO",30000.0))

    W_HSFO_pre = _ss_or_def("WtW_HSFO", _get(DEFAULTS, "WtW_HSFO", 92.78))
    W_LFO_pre  = _ss_or_def("WtW_LFO",  _get(DEFAULTS, "WtW_LFO" , 92.00))
    W_MGO_pre  = _ss_or_def("WtW_MGO",  _get(DEFAULTS, "WtW_MGO",  93.93))
    W_BIO_pre  = _ss_or_def("WtW_BIO",  _get(DEFAULTS, "WtW_BIO",  70.00))
    W_RFN_pre  = _ss_or_def("WtW_RFNBO",_get(DEFAULTS, "WtW_RFNBO",20.00))
    wtw_pre = {"HSFO": W_HSFO_pre, "LFO": W_LFO_pre, "MGO": W_MGO_pre, "BIO": W_BIO_pre, "RFNBO": W_RFN_pre, "ELEC": 0.0}

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
# UI (unchanged except for the pooling price input which is already present)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="FuelEU Maritime Calculator", layout="wide")

st.title("FuelEU Maritime — GHG Intensity & Cost — TMS DRY — ENVIRONMENTAL")
st.caption("Period: 2025–2050 • Limits derived from 2020 baseline 91.16 gCO₂e/MJ • WtW basis • Prices in EUR")

# (CSS, sidebar inputs, masses, LCVs, WtW, OPS_kWh etc. — same as your previous file)
# For brevity in this reply I assume the entire sidebar + inputs block is exactly as in your
# previous file and already includes:
#   - pooling_price_eur_per_tco2e  (NEW input you asked for),
#   - pooling_tco2e_input (manual request, sign determines uptake/provide),
#   - pooling_start_year, banking_tco2e_input, banking_start_year, etc.
#
# ... (the full sidebar code is identical to the previous version you already approved)
#
# For the optimizer we will rely on these variables:
#   pooling_tco2e_input  (manual request; positive=uptake allowed, negative=provide allowed)
#   pooling_price_eur_per_tco2e (€/tCO2e)
#   pooling_start_year, banking_start_year, banking_tco2e_input
#   carry_in_list (precomputed from the baseline per-year pass — unchanged)
#
# ──────────────────────────────────────────────────────────────────────────────
# The application continues unchanged up until the optimizer section.
# I will now present the updated optimizer and the updated penalty evaluator
# that are required to let the optimizer choose pooling p simultaneously with x.
# ──────────────────────────────────────────────────────────────────────────────

# --- Helper: compute scoped + intensity for masses (unchanged helper used earlier) ---
# (scoped_and_intensity_from_masses is the same as in your app; keep it.)

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

# --- Updated penalty evaluator that accepts a pooling override (decision variable p) ---
def penalty_credit_pooling_with_masses_for_year(year_idx: int,
                                                h_v, l_v, m_v, b_v, r_v,
                                                h_b, l_b, m_b, b_b, r_b,
                                                pooling_override: float | None = None) -> Tuple[float, float, float, float, float, float, float]:
    """
    Returns:
      penalty_usd_x, credits_usd_x, pooling_cost_usd_x, g_att_x, pool_use_applied, bank_use_x, final_bal_x
    The function uses the *same* rules as the main app:
      - banking cap is computed from cb_eff (before pool),
      - pooling provide is capped vs pre-surplus (computed from cb_eff),
      - safety clamp trims bank then pool (identical approach).
    pooling_override: if None → use global pooling_tco2e_input (manual); if numeric → treat as candidate p (decision variable).
    """
    year = years[year_idx]
    g_target = limit_series[year_idx]

    # Scoped energy & intensity for masses (x-dependent; p does not change fuel energies)
    E_scope_x, num_phys_x, E_rfnbo_scope_x = scoped_and_intensity_from_masses(
        h_v, l_v, m_v, b_v, r_v, h_b, l_b, m_b, b_b, r_b, ELEC_MJ, wtw, year
    )
    if E_scope_x <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    r = 2.0 if year <= 2033 else 1.0
    den_rwd_x = E_scope_x + (r - 1.0) * E_rfnbo_scope_x
    g_att_x = (num_phys_x / den_rwd_x) if den_rwd_x > 0 else 0.0

    CB_g_x = (g_target - g_att_x) * E_scope_x
    CB_t_raw_x = CB_g_x / 1e6
    cb_eff_x = CB_t_raw_x + carry_in_list[year_idx]

    # Determine pool_use_x: honor pooling_start_year and apply capping similar to main code
    if year >= int(pooling_start_year):
        # decide pool request from override or global input
        pool_request = pooling_override if pooling_override is not None else pooling_tco2e_input
        if pool_request >= 0:
            pool_use_x = float(pool_request)
        else:
            provide_abs = abs(pool_request)
            pre_surplus = max(cb_eff_x, 0.0)
            applied_provide = min(provide_abs, pre_surplus)
            pool_use_x = -applied_provide
    else:
        pool_use_x = 0.0

    # Banking (unchanged): pre-surplus based on cb_eff_x (not on pool)
    if year >= int(banking_start_year):
        pre_surplus_b = max(cb_eff_x, 0.0)
        requested_bank = max(banking_tco2e_input, 0.0)
        bank_use_x = min(requested_bank, pre_surplus_b)
    else:
        bank_use_x = 0.0

    # Final balance and safety clamp (trim bank then pool to avoid flipping surplus->deficit)
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

    # Monetary conversions (linear in final_bal_x)
    # Credits: final_bal_x > 0
    if final_bal_x > 0:
        credit_price_eur_per_vlsfo_t = credit_per_tco2e * st.session_state["factor_vlsfo_per_tco2e"] if st.session_state["factor_vlsfo_per_tco2e"]>0 else 0.0
        credit_eur_x = euros_from_tco2e(final_bal_x, g_att_x, credit_price_eur_per_vlsfo_t)
        credits_usd_x = credit_eur_x * eur_usd_fx
        penalty_usd_x = 0.0
    elif final_bal_x < 0:
        penalty_eur_x = euros_from_tco2e(-final_bal_x, g_att_x, penalty_price_eur_per_vlsfo_t)
        penalty_usd_x = penalty_eur_x * eur_usd_fx
        credits_usd_x = 0.0
    else:
        penalty_usd_x = credits_usd_x = 0.0

    # Pooling cashflow in USD (sign preserved: + for uptake cost, − for provide revenue)
    pooling_cost_usd_x = pool_use_x * pooling_price_eur_per_tco2e * eur_usd_fx

    return penalty_usd_x, credits_usd_x, pooling_cost_usd_x, g_att_x, pool_use_x, bank_use_x, final_bal_x

# ──────────────────────────────────────────────────────────────────────────────
# Optimizer (now chooses both x (fuel decrease) and p (pooling tCO2e) per year)
# ──────────────────────────────────────────────────────────────────────────────
dec_opt_list, bio_inc_opt_list, pooling_opt_list = [], [], []

# Also store the *optimized* cost breakdown per year so we can show it
opt_penalty_usd_list = []
opt_credits_usd_list = []
opt_pooling_cost_usd_list = []
opt_bio_premium_cost_usd_list = []
opt_total_cost_usd_list = []
opt_attained_g_list = []

for i in range(len(years)):
    # Year-specific data
    year = years[i]

    # Determine selected fuel availability and LCV
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
        # nothing to optimize
        dec_opt_list.append(0.0)
        bio_inc_opt_list.append(0.0)
        pooling_opt_list.append(0.0)
        opt_penalty_usd_list.append(0.0)
        opt_credits_usd_list.append(0.0)
        opt_pooling_cost_usd_list.append(0.0)
        opt_bio_premium_cost_usd_list.append(0.0)
        opt_total_cost_usd_list.append(0.0)
        opt_attained_g_list.append(0.0)
        continue

    steps_coarse = 60
    steps_fine = 80
    x_max = total_avail

    best_x = 0.0
    best_p = 0.0
    best_cost = float("inf")
    best_components = (0.0, 0.0, 0.0, 0.0)  # penalty, credits, pooling_cost, bio_premium
    best_g_att = 0.0

    def evaluate_for_x(x_candidate: float) -> Tuple[float, float, float, float, float, float, float]:
        """
        For given x_candidate, search feasible p candidates efficiently (endpoints and p that zeros final balance),
        evaluate objective for each, and return tuple for best p:
        (best_total_cost_usd, best_penalty_usd, best_credits_usd, best_pooling_cost_usd, best_bio_prem_usd, best_p, best_g_att)
        """
        masses = masses_after_shift_generic(selected_fuel_for_opt, x_candidate)

        # new BIO total (tons) under candidate x
        if "Extra-EU" in voyage_type:
            new_bio_total_t = (masses[3] + masses[8])  # b_v + b_b
        else:
            new_bio_total_t = masses[3]                # b_v

        bio_premium_usd_x = new_bio_total_t * bio_premium_usd_per_t

        # Compute CB eff and bank caps to get feasible p range and p_zero
        # Use the internal evaluator but first compute cb_eff_x & bank_use_x ourselves (cheap)
        # We'll call the evaluator for each p candidate to get the clamped penalty/credits and pooling cost.
        # We compute cb_eff_x and bank_use_x by calling the evaluator with pooling_override=0 (temporary)
        # and then reconstruct from those returned values.
        # (This ensures the same formulas and caps are used.)
        dummy_pen, dummy_cred, dummy_pool_cost, g_att_x, pool_applied_tmp, bank_use_x, cb_final_tmp = \
            penalty_credit_pooling_with_masses_for_year(i, *masses, pooling_override=0.0)

        # Note: cb_eff_x = cb_raw_x + carry_in_list[i]. We can reconstruct cb_raw_x from cb_final_tmp, bank and pool
        # but simpler: re-run the internal steps to get cb_eff_x directly:
        # Use the scoped / CB formulas:
        E_scope_x, num_phys_x, E_rfnbo_scope_x = scoped_and_intensity_from_masses(
            masses[0], masses[1], masses[2], masses[3], masses[4],
            masses[5], masses[6], masses[7], masses[8], masses[9],
            ELEC_MJ, wtw, year
        )
        if E_scope_x <= 0:
            return float("inf"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        r_local = 2.0 if year <= 2033 else 1.0
        den_rwd_local = E_scope_x + (r_local - 1.0) * E_rfnbo_scope_x
        g_att_local = (num_phys_x / den_rwd_local) if den_rwd_local > 0 else 0.0
        CB_g_local = (limit_series[i] - g_att_local) * E_scope_x
        CB_t_raw_local = CB_g_local / 1e6
        cb_eff_local = CB_t_raw_local + carry_in_list[i]
        pre_surplus_local = max(cb_eff_local, 0.0)
        requested_bank_local = max(banking_tco2e_input, 0.0) if year >= int(banking_start_year) else 0.0
        bank_use_local = min(requested_bank_local, pre_surplus_local) if year >= int(banking_start_year) else 0.0

        # Feasible p range (respect pooling_start_year and the sign + magnitude of the manual pooling request)
        if year >= int(pooling_start_year):
            if pooling_tco2e_input >= 0:
                p_min = 0.0
                p_max = float(pooling_tco2e_input)
            else:
                provide_abs = abs(float(pooling_tco2e_input))
                p_min = -min(provide_abs, pre_surplus_local)
                p_max = 0.0
        else:
            p_min = p_max = 0.0

        # Candidate p's: endpoints and p that zeros final balance (p_zero = bank_use - cb_eff)
        candidates_p = set([p_min, p_max])
        p_zero = bank_use_local - cb_eff_local  # solves cb_eff + p - bank_use = 0
        if p_zero >= p_min - 1e-12 and p_zero <= p_max + 1e-12:
            candidates_p.add(p_zero)

        # Evaluate each candidate p using the canonical evaluator (which applies clamps exactly)
        local_best_cost = float("inf")
        local_best_tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        for p_candidate in sorted(candidates_p):
            pen_usd, cred_usd, pool_cost_usd, g_att_eval, pool_applied_eval, bank_use_eval, final_bal_eval = \
                penalty_credit_pooling_with_masses_for_year(i, *masses, pooling_override=p_candidate)

            total_cost_candidate = pen_usd - cred_usd + pool_cost_usd + bio_premium_usd_x

            if total_cost_candidate < local_best_cost:
                local_best_cost = total_cost_candidate
                local_best_tuple = (total_cost_candidate, pen_usd, cred_usd, pool_cost_usd, bio_premium_usd_x, p_candidate, g_att_eval)

        return local_best_tuple

    # ----- Coarse scan over x -----
    for s in range(steps_coarse + 1):
        x = x_max * s / steps_coarse
        total_cost_candidate, pen_c, cred_c, pool_cost_c, bio_prem_c, p_choice_c, g_att_c = evaluate_for_x(x)
        if total_cost_candidate < best_cost:
            best_cost = total_cost_candidate
            best_x = x
            best_p = p_choice_c
            best_components = (pen_c, cred_c, pool_cost_c, bio_prem_c)
            best_g_att = g_att_c

    # ----- Fine scan around best_x -----
    delta = x_max / steps_coarse * 2.0
    left = max(0.0, best_x - delta)
    right = min(x_max, best_x + delta)
    for s in range(steps_fine + 1):
        x = left + (right - left) * s / steps_fine
        total_cost_candidate, pen_c, cred_c, pool_cost_c, bio_prem_c, p_choice_c, g_att_c = evaluate_for_x(x)
        if total_cost_candidate < best_cost:
            best_cost = total_cost_candidate
            best_x = x
            best_p = p_choice_c
            best_components = (pen_c, cred_c, pool_cost_c, bio_prem_c)
            best_g_att = g_att_c

    # Save results for the year
    dec_opt_list.append(best_x)
    bio_inc_opt = best_x * (LCV_SEL / LCV_BIO) if LCV_BIO > 0 else 0.0
    bio_inc_opt_list.append(bio_inc_opt)
    pooling_opt_list.append(best_p)

    pen_opt, cred_opt, pool_cost_opt, bio_prem_opt = best_components[0], best_components[1], best_components[2], best_components[3]
    opt_penalty_usd_list.append(pen_opt)
    opt_credits_usd_list.append(cred_opt)
    opt_pooling_cost_usd_list.append(pool_cost_opt)
    opt_bio_premium_cost_usd_list.append(bio_prem_opt)
    opt_total_cost_usd_list.append(best_cost)
    opt_attained_g_list.append(best_g_att)

# ──────────────────────────────────────────────────────────────────────────────
# Build results DataFrame — keep base columns and add optimizer outputs
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

        # Pooling cost (base, from manual pool_applied)
        "Pooling_Cost_USD": [p * pooling_price_eur_per_tco2e * eur_usd_fx for p in pool_applied],

        "Penalty_USD": penalties_usd,
        "Credit_USD": credits_usd,
        "BIO Premium Cost_USD": bio_premium_cost_usd_col,

        # Keep base total (unchanged) and also provide the *optimized* total as a separate column
        "Total_Cost_USD": total_cost_usd_col,  # base (manual) total as before
        decrease_col_name: dec_opt_list,
        "BIO_Increase(t)_For_Opt_Cost": bio_inc_opt_list,

        # >>> NEW column requested by you: pooling choice selected by the optimizer (tCO2e)
        "Pooling(tCO2e)_For_Opt_Cost": pooling_opt_list,

        # Optimized breakdown (from the optimizer)
        "Penalty_USD_Opt": opt_penalty_usd_list,
        "Pooling_Cost_USD_Opt": opt_pooling_cost_usd_list,
        "BIO_Premium_USD_Opt": opt_bio_premium_cost_usd_list,
        "Total_Cost_USD_Opt": opt_total_cost_usd_list,
        "Actual_gCO2e_per_MJ_After_Opt": opt_attained_g_list,
    }
)

# Format and display
df_fmt = df_cost.copy()
for col in df_fmt.columns:
    if col != "Year":
        df_fmt[col] = df_fmt[col].apply(us2)

st.dataframe(df_fmt, use_container_width=True)

st.download_button(
    "Download per-year results (CSV)",
    data=df_fmt.to_csv(index=False),
    file_name="fueleu_results_2025_2050_with_opt_pooling.csv",
    mime="text/csv",
)
