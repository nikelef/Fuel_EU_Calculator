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

| Years     | Reduction % | Limit formula |
| --------- | ----------- | ------------- |
| 2025–2029 | 2.0%        |               |
