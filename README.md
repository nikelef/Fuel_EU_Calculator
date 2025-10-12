# FuelEU Maritime — GHG Intensity & Cost Calculator (2025–2050)

Single‑vessel Streamlit application to compute **attained GHG intensity** (WtW), compare it to the **FuelEU Maritime limit** for 2025–2050 (derived from the **2020 baseline 91.16 gCO₂e/MJ**), and estimate **compliance costs or credits in EUR**. The app supports **Intra‑EU** and **Extra‑EU** scoping, **RFNBO reward (×2 in the denominator through 2033)**, and operational levers including **Banking** and **Pooling** with rigorous caps and a **consecutive‑deficit penalty multiplier**.

> **Status:** Production‑ready for internal use. EUR‑only. Numbers are **WtW**; electricity at‑berth (EU OPS) is assumed **0 gCO₂e/MJ**.

---

## Key Features

* **Scope logic**

  * **Intra‑EU:** 100% of **all fuels** + **100%** of EU **OPS electricity** (WtW = 0).
  * **Extra‑EU:** EU OPS **electricity 100%**; fuels split into:

    * **At‑berth fuels (EU ports):** **100%** scope
    * **Voyage fuels:** **50%** scope

* **Extra‑EU allocator (WtW‑prioritized, pool‑then‑fill)**

  1. Build one in‑scope **fuel pool** = 100% **at‑berth** fuels + **50%** of **total voyage** fuels (electricity always 100% in‑scope, handled separately).
  2. **Fill** that pool (without changing its total) by WtW priority:

     * **Renewables**: **RFNBO vs BIO** — **lower WtW first**; take **at‑berth 100% first**, then **voyage** up to the **spare** pool (leaving room so **all fossil at‑berth** still fit 100%).
     * **Fossil at‑berth** (HSFO, LFO, MGO): **100%** in ascending WtW.
     * **Fossil voyage** (HSFO, LFO, MGO): **50%** per fuel in ascending WtW (**partial on the last** if needed).

* **RFNBO reward (compliance only)**

  * Through **31 Dec 2033**, **RFNBO** energy counts **×2 in the denominator** (physical emissions unchanged), i.e. `den_rwd = den_phys + RFNBO_energy`.

* **Banking & Pooling (independent)**

  * **Pooling [tCO₂e]**: `+` uptake applies as entered (can overshoot); `−` provide is **capped** to **pre‑adjustment surplus** (never flips a surplus to deficit).
  * **Banking to next year [tCO₂e]**: **capped** to **pre‑adjustment surplus**; creates next‑year **carry‑in** equal to the **final banked** amount.
  * **Start‑year selectors** for both Pooling and Banking.

* **Consecutive‑deficit multiplier (automatic)**

  * **+10%** penalty **per additional consecutive deficit year**. The UI “**seed**” pre‑loads the run length when the **first deficit** appears.

* **Linked market prices**

  * **Credits**: user edits **€/tCO₂e**, app derives **€/VLSFO‑eq t**.
  * **Penalties**: user edits **€/VLSFO‑eq t**, app derives **€/tCO₂e**.
  * Conversion uses the **attained intensity** preview:
    [ tCO₂e/VLSFO_eq_t = (g_{attained} [g/MJ] * 41,000 [MJ/t]) / 10^6 ]

* **UI/formatting**

  * **US number format** (`1,234.56`) everywhere; **no steppers** except for the **seed**.
  * Clear **Intra‑EU vs Extra‑EU** mass sections and **EU OPS electricity in kWh** (internally → MJ).
  * Compact data table with **wrapped, centered headers** and **centered values**.
  * Visuals: **Energy composition (all vs in‑scope)** with connectors & % labels; **GHG Intensity vs Limit** with step labels and dashed “Attained”.

---

## Reduction Steps & Limits (from 2020 baseline 91.16 gCO₂e/MJ)

| Years     | Reduction % | Limit formula         |
| --------- | ----------- | --------------------- |
| 2025–2029 | 2.0%        | `91.16 * (1 − 0.02)`  |
| 2030–2034 | 6.0%        | `91.16 * (1 − 0.06)`  |
| 2035–2039 | 14.5%       | `91.16 * (1 − 0.145)` |
| 2040–2044 | 31.0%       | `91.16 * (1 − 0.31)`  |
| 2045–2049 | 62.0%       | `91.16 * (1 − 0.62)`  |
| 2050      | 80.0%       | `91.16 * (1 − 0.80)`  |

The app computes and displays the **step series** and overlays the **attained** line for 2025–2050.

---

## Inputs

* **Voyage scope**: `Intra‑EU (100%)` or `Extra‑EU (50%)`.
* **Masses [t]**

  * *Intra‑EU*: **total** (voyage + at‑berth) for HSFO, LFO, MGO, BIO, RFNBO.
  * *Extra‑EU*: **voyage (excluding at‑berth)** + **at‑berth (EU ports)**, both for HSFO, LFO, MGO, BIO, RFNBO.
* **LCVs [MJ/ton]**: HSFO, LFO, MGO, BIO, RFNBO.
* **WtW intensities [gCO₂e/MJ]**: HSFO, LFO, MGO, BIO, RFNBO (electricity fixed at **0**).
* **EU OPS electricity [kWh]** (converted to MJ with `× 3.6`).
* **Compliance Market — Credits**: edit **€/tCO₂e** (derived **€/VLSFO‑eq t** is read‑only).
* **Compliance Market — Penalties**: edit **€/VLSFO‑eq t** (derived **€/tCO₂e** is read‑only).
* **Consecutive deficit years (seed)**: integer ≥ 1.
* **Banking & Pooling [tCO₂e]**

  * **Pooling**: `+ uptake`, `− provide` (capped vs pre‑surplus); **start year** selector.
  * **Banking**: non‑negative; **capped vs pre‑surplus**; **start year** selector; creates **carry‑in**.
* **Save defaults**: writes inputs to **`.fueleu_defaults.json`** (see Persistence).

> **Units**: Mass = **ton**; LCV = **MJ/ton**; Intensity = **gCO₂e/MJ**; Electricity input = **kWh**.

---

## Outputs & Visuals

* **Top metrics**: Total energy (all), In‑scope energy, Fossil/BIO/RFNBO (all and in‑scope).
* **Energy composition (MJ)**: two **stacked columns** (All vs In‑scope), ordered **ELEC at bottom**, then fuels by **ascending WtW** on both sides; dashed connectors with **% labels** for each layer; totals annotated.
* **GHG Intensity vs FuelEU Limit (2025–2050)**: step line for limits, dashed line for attained; **step labels** shown; units **gCO₂e/MJ**.
* **Results table (per year)** with CSV export, columns:

  * `Year`, `Reduction_%`, `Limit_gCO2e_per_MJ`, `Actual_gCO2e_per_MJ`, `Emissions_tCO2e` (physical),
  * `Compliance_Balance_tCO2e` (raw), `CarryIn_Banked_tCO2e`, `Effective_Balance_tCO2e`,
  * `Banked_to_Next_Year_tCO2e`, `Pooling_tCO2e_Applied`, `Final_Balance_tCO2e_for_€`,
  * `Penalty_EUR`, `Credit_EUR`, `Net_EUR`.

All numbers use **US formatting** with **2 decimals**.

---

## Computation Details

### Energies & Intensities

* Per‑fuel energy (MJ): `mass_t × LCV_MJ_per_t`.
* **In‑scope energies**: direct (Intra‑EU) or via the **Extra‑EU allocator** (see above). Electricity always 100% in‑scope.
* **Physical attained intensity** `g_base`: `num_phys / den_phys`, where `num_phys = Σ(E_fuel_in_scope × WtW_fuel)` and `den_phys = Σ(E_fuel_in_scope)`.
* **Compliance attained intensity for year y**:

  * Let `r = 2` **if y ≤ 2033**, else `r = 1`. Denominator becomes `den_rwd = den_phys + (r − 1) × E_RFNBO_in_scope`.
  * `g_attained(y) = num_phys / den_rwd` (if `den_rwd > 0`).

### Credits/Penalties (EUR)

* Convert **positive** tCO₂e (surplus for credits / deficit for penalties) using **attained** intensity:

  * `tCO₂e per VLSFO‑eq t = (g_attained [g/MJ] × 41,000 [MJ/t]) / 10^6`.
  * `EUR = (tCO₂e / tCO₂e_per_VLSFO_eq_t) × price_eur_per_VLSFO_eq_t`.
* **Penalty multiplier**: if `Final_Balance_tCO₂e_for_€ < 0`, apply `1 + 0.10 × (run_len − 1)` to **penalties** only. Run length increases with **consecutive** deficit years; the UI **seed** pre‑loads the initial run.

### Banking & Pooling Order (per year)

1. **Carry‑in** (previous year’s banked amount).
2. Compute **effective balance** `cb_eff = CB_raw + carry_in`.
3. Apply **Pooling** (with independence vs `cb_eff`, capping `provide` to pre‑surplus).
4. Apply **Banking** (capped vs pre‑surplus); set **carry‑out** = banked amount.
5. **Safety clamp**: prevent flips from surplus → deficit due to over‑banking or over‑providing.
6. Convert **Final Balance** to **€** (credits or penalties), then add **multiplier** if in deficit.

Informational notes surface when **capping** or the **safety trim** occurred.

---

## Persistence

* Inputs can be saved to **`.fueleu_defaults.json`** via the “Save current inputs as defaults” button.
* On load, the app reads from this file; if `OPS_kWh` is missing but legacy `OPS_MWh` exists, it converts `MWh → kWh` automatically.
* Do **not** commit sensitive pricing to public repos.

---

## Installation & Run

```bash
# 1) Python 3.10+ recommended
python -V

# 2) Clone repo and enter it
git clone <your-repo-url>.git
cd <your-repo-folder>

# 3) (Optional) create a virtual environment
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate

# 4) Install dependencies
pip install -r requirements.txt

# 5) Run the app
streamlit run app.py
```

Open the local URL shown by Streamlit (typically `http://localhost:8501`).

---

## Repository Structure

```
.
├── app.py                    # The Streamlit app (this project)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── .fueleu_defaults.json     # (optional) persisted defaults, created at runtime
```

> You may also include a `.streamlit/config.toml` if you wish to customize Streamlit theme.

---

## Deployment

* **Streamlit Community Cloud**: push to GitHub, then deploy the repo. Ensure Python version and `requirements.txt` are set. The app is **offline** (no external API calls) and stores only a small JSON defaults file.
* **Self‑hosting**: run behind an internal reverse proxy; persist the working directory so `.fueleu_defaults.json` survives restarts.

---

## Assumptions & Notes

* Electricity (EU OPS) is **0 gCO₂e/MJ** in compliance accounting.
* WtW values, LCVs, masses, and prices are **user‑provided**; verify against current regulations and supplier certificates.
* This tool is **single‑vessel** and **EUR‑only**.
* **RFNBO reward** affects **compliance denominator** only; physical emissions remain based on **den_phys**.

---

## Changelog

* **2025‑10‑08**

  * Added **independent** Pooling/Banking with **caps vs pre‑surplus** and **start‑year** selectors.
  * One‑way **price linking** (Credits: €/tCO₂e → €/VLSFO‑eq t; Penalties: €/VLSFO‑eq t → €/tCO₂e).
  * Introduced **consecutive‑deficit multiplier** (+10% per year) with UI **seed**.
  * Refined visuals (compact margins, dashed “Attained”, wrapped/centered table headers and values).

---

## License

Specify your project’s license (e.g., MIT). If omitted, default repository terms apply.

---

## Disclaimer

This tool reflects the logic implemented in `app.py` and is provided for **operational planning**. It is **not legal advice**. Always consult the official regulation text and your compliance team before making commercial decisions.
