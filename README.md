# FuelEU Maritime — GHG Intensity & Cost (First Cut)

A minimal Streamlit app to help you **visualize FuelEU Maritime limits (2025–2050)**, compute
your **mix GHG intensity (WtW, gCO₂e/MJ)** from user‑entered fuel quantities and LCVs,
and estimate **FuelEU penalties/credits** and **BIO premium economics**.

> **Assumptions**
>
> * Baseline (2020): **91.16 gCO₂e/MJ**.
> * Reduction steps (limit plateaus): **2% (2025–2029), 6% (2030–2034), 14.5% (2035–2039), 31% (2040–2044), 62% (2045–2049), 80% (2050)**.
> * Penalty (Annex IV, simplified):  
>   `Penalty(€) = max(0, −CB) / (GHG_actual * 41,000) * 2,400`,  
>   where `CB (gCO₂e) = (Target − Actual) * Energy_MJ_in_scope`.
> * Voyage scope: **100%** of energy for **Intra‑EU** voyages; **50%** for **Extra‑EU** voyages.

## Run locally

```bash
# 1) Create env (optional)
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)

# 2) Install deps
pip install -r requirements.txt

# 3) Launch
streamlit run app.py
```

## Deploy to Streamlit Cloud

1. Push this folder to a GitHub repo.
2. In Streamlit Cloud, select your repo and set **Main file** to `app.py`.
3. Set **Python version** to 3.10+.

## Inputs

* Voyage scope (Intra‑EU 100% / Extra‑EU 50%).
* Masses `[t]`, WtW `[gCO₂e/MJ]`, LCV `[MJ/ton]` for **HSFO, LFO, MGO, BIO**.
* **Fuel to be replaced** by BIO, **Premium** `USD/ton = price(BIO) − price(selected fuel)`,
  and **Base price** of the selected fuel (used to compute an **energy‑equivalent** cost delta).
* Optional: **Credit price** (€/VLSFO‑equiv t), **EUR→USD FX**, and **consecutive deficit years** (n).

## Outputs

* **Step plot**: FuelEU **limit** vs. your **mix intensity**, 2025–2050. Title shows **total energy considered (MJ)**.
* **Tables**:
  * GHG intensity (mix) per year (with delta to limit),
  * Emissions (tCO₂e) for the considered scope,
  * FuelEU **penalty (€)** / **credit (€)** and **net (€)** per year.
* **Premium economics**: USD cost delta for using BIO vs. the replaced fuel on an **energy‑equivalent** basis.

## Notes

* This is a **first cut**. It does **not** yet model RFNBO multipliers, OPS penalties, pooling/banking/borrowing
  strategies, or verifier‑specific factors. Those can be added next.
* All numbers are user‑provided; defaults are illustrative only.