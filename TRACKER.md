# AeroSurrogate — 2-Week Task Tracker

**Project:** AeroSurrogate, a physics-informed ML surrogate that predicts a car's drag (Cd), lift (Cl) and EV highway range from 26 geometric design parameters in under 2 ms, plus a constrained inverse-design optimizer. Dataset: DrivAerNet++ (MIT, NeurIPS 2024).
**Goal:** CV proof for high-paying EV/F1/software jobs. Sits in the Automotive AI track alongside Verity (LLM), PitWall (F1) and VoltTrace (EV).
**Start:** Sunday 22 Nov 2026 (Day 1) · **End:** Saturday 5 Dec 2026 (Day 14)
**Rules:** laptop only, CPU-first inference, no paid data, no 39 TB volumetric CFD. Lightweight tabular CSVs only. MIT's official splits, never re-split. Every benchmark number measured and logged, never claimed.
**License:** DrivAerNet++ is CC BY-NC 4.0, non-commercial research and education only. Fine for a portfolio project, not for commercial use.

## Status — 22 Nov 2026 (Antigravity build, checked by Ellie)

Days 2–7 landed in one build session on Kartikey's laptop (Antigravity/Freebuff), delivered as `AeroSurrogate_Complete.zip` and merged into this repo. Verified by Ellie the same day: the ONNX model loads and runs (0.051 ms per prediction on this VM, target met), the training code trains on train only and evaluates on val only, test split untouched.

- **Days 3–4, features + baselines: done.** 35 features (23 raw + 12 engineered ratios). Val results: LightGBM Cd MAE 0.0103, R² 0.64; CatBoost 0.0110 / 0.62; Residual MLP 0.0126 / 0.53. Cl R² 0.30–0.38 (weak). Frontal area R² > 0.998. Estateback Cd within-group R² collapses to 0.01–0.18. Full table in `benchmarks/baselines_day3-4.md`.
- **Days 5–7, uncertainty + ONNX: done.** Split-conformal intervals (CV+ on train, evaluated on val): 90% nominal gives 90–94% empirical coverage, 95% nominal gives 95–97%. Joint MLP exported to ONNX, 0.051 ms measured. CatBoost ONNX mean 2.38 ms, median 1.13 ms. See `benchmarks/conformal_day4.md`, `benchmarks/onnx_latency_day5.md`.
- **Days 8–9, EV physics + optimizer: done early.** `src/physics/ev_range.py` (drag area, aero power, range delta for a 75 kWh EV at 110 km/h) and `src/optimize/inverse.py` (Optuna, constrained, uses the ONNX model).
- **Still open:** accuracy tuning toward Cd MAE < 0.005 / R² > 0.96 (baselines are far from target, this is the main risk), FastAPI `/predict` + `/optimize` (Days 10–12), Streamlit dashboard (Days 10–12), final test-split evaluation + demo + README (Days 13–14).

Note: the Antigravity build used a compressed day numbering (its Day 8 = API, Day 9 = UI). This file is the plan of record; the mapping above uses this file's numbering.

## Day 1 findings (verified on the real files, 22 Nov 2026)

Read these before believing the brief's numbers:

- The repo's parametric CSV has **4,165 designs, not 8,150**, and **23 geometric parameters, not 26**. The full 8,150-design parametric table is not in the lightweight files; the extra ~4,000 designs (almost all fastbacks) exist only in the Cd file and the splits.
- The Cd file (`DrivAerNetPlusPlus_Cd_8k_Updated.csv`, 8,121 rows) has **Cd only, no Cl**. Cl, Cl_f and Cl_r exist only inside the parametric file, for the 4,165 subset.
- The usable full-join set (params + Cd + Cl + frontal area + official split) is **4,051 designs**. That is the real training set, not 8,150.
- Official splits: train 5,819, val 1,148, test 1,154 (71.7 / 14.1 / 14.2, not exactly 70/15/15). The full official test split is **1,154 designs, not 1,200**, and only **559** of them are in the usable merged set.
- The two Cd sources agree (check `manifest.json` for the exact max diff). Frontal areas exist for 8,007 of 8,121 Cd designs.
- Body code is the first letter of the design ID: F fastback, N notchback, E estateback. The ++ expansion is almost all fastbacks (5,332 of 8,121).

## Week 1 — Data, Surrogates & ONNX (22–28 Nov)

| Day | Date | Milestone | Done when |
|---|---|---|---|
| 1–2 | Sun 22–Mon 23 Nov | **Download + merge + validate.** Download the 6 lightweight files, merge on design ID, validate ranges/overlaps/splits, write the merged Parquet + manifest. | `data/processed/drivaer_merged.parquet` + `manifest.json` exist; Day 1 findings above confirmed |
| 3–4 | Tue 24–Wed 25 Nov | **Features + baselines.** Engineer geometric ratios (slenderness, wheel-to-body). Train baseline LightGBM, CatBoost and PyTorch Residual MLP on the official train split; evaluate on val. | Baseline Cd MAE + R² per model logged |
| 5–7 | Thu 26–Sat 28 Nov | **Tune + uncertainty + ONNX.** Tune the ensemble toward Cd MAE < 0.005; add 95% conformal prediction intervals; export to `.onnx` and verify zero accuracy drop vs native. | ONNX parity check passing; conformal coverage measured |

## Week 2 — Inverse Design, EV Physics & Demo (29 Nov–5 Dec)

| Day | Date | Milestone | Done when |
|---|---|---|---|
| 8–9 | Sun 29–Mon 30 Nov | **EV physics + optimizer.** Range module (Cd·A, aero power, range delta for a 75 kWh EV at 110 km/h) and Optuna constrained inverse design (~500 evals in < 1.5 s). | Optimizer returns a lowest-drag geometry under constraints, timed |
| 10–12 | Tue 1–Thu 3 Dec | **API + UI.** FastAPI `/predict` (surrogate + uncertainty + SHAP + range) and `/optimize`; Streamlit UI with geometry sliders, live Cd/Cl/range gauges, SHAP waterfall, 1-click optimize. | Both endpoints live; UI drives them end to end |
| 13–14 | Fri 4–Sat 5 Dec | **Final eval + demo + README.** Evaluate on the official test split (1,154 designs; 559 in the usable merged set) (overall + per body type + ICE vs EV); record a 45 s demo; publish README with benchmark tables. | Test table complete; video + README done |

## Target metrics (all measured, none assumed)

| Metric | Target |
|---|---|
| Drag accuracy, Cd MAE on official test split | < 0.005 |
| Explained variance, R² overall and per body type | > 0.96 |
| Inference latency per prediction, CPU | < 2 ms |
| Inverse design gain: Cd reduction % and extra EV km | measured, < 1.5 s optimisation |

## Risks to watch

- The 0.005 Cd MAE target was set for the full 8k dataset. The usable set is 4,051 designs, so the target may be harder or easier; report the honest number on the official test split either way.
- The parametric file's Cd is an average over CFD runs with a standard deviation column; the Cd file is a single updated value per design. The merge script checks they agree; if they diverge anywhere, decide which one is the label and record why.
- 114 parametric designs have no frontal area and 3 Cd designs are in no official split. Those rows are excluded from the merged set; the manifest records the exact counts.
