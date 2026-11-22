# AeroSurrogate

A physics-informed ML surrogate that predicts a car's aerodynamic drag (Cd), lift (Cl) and EV highway range from geometric design parameters in under 2 ms, replacing hours of CFD, plus a constrained inverse-design optimizer.

Car aerodynamic design normally needs CFD simulation, which takes hours for one design. AeroSurrogate learns the mapping from shape to aerodynamics from high-fidelity CFD data, so engineers can iterate on a shape in milliseconds and ask the inverse question: what shape gives the lowest drag under my constraints?

## What it does

- **Surrogate engine:** LightGBM + CatBoost + PyTorch Residual MLP ensemble (ONNX Runtime on CPU) predicting Cd, Cl and frontal area from parametric car geometry, with calibrated 95% conformal prediction intervals.
- **EV range bridge:** drag area (Cd·A) and aero power turn drag changes into EV highway range delta in km for a reference 75 kWh EV at 110 km/h.
- **Inverse design:** Optuna optimizer that returns the lowest-drag geometry under user constraints in under 1.5 seconds.
- **Dashboard:** FastAPI `/predict` and `/optimize` endpoints behind a UI with geometry sliders, live Cd/Cl/range gauges, a SHAP waterfall and 1-click optimization. *(In progress, see Status.)*

## Measured results

All numbers measured on 22 Nov 2026, laptop CPU only. Validation split is MIT's official val split (555 designs in the usable merged set). The test split (559 designs) is untouched until final evaluation.

| Metric | Target | Measured |
|---|---:|---:|
| Cd MAE, LightGBM baseline, val | < 0.005 | 0.0103 |
| Cd R², LightGBM baseline, val | > 0.96 | 0.64 |
| Cd MAE, CatBoost baseline, val | < 0.005 | 0.0110 |
| Cd MAE, Residual MLP baseline, val | < 0.005 | 0.0126 |
| Cl R², best baseline, val | > 0.96 | 0.38 |
| Frontal area R², best baseline, val | > 0.96 | 0.999 |
| Inference latency, joint MLP ONNX, single prediction | < 2 ms | 0.051 ms |
| Conformal coverage, 90% nominal, Cd | 90% | 90–94% |
| Conformal coverage, 95% nominal, Cd | 95% | 95–97% |

Honest read: the latency target is met and the uncertainty calibration is sound, but the accuracy targets are not. Baselines sit at Cd MAE 0.0103 and R² 0.64, and lift is hard to predict from geometry alone (R² 0.30–0.38). Estateback drag is the weakest subgroup (within-group R² 0.01–0.18). Closing the accuracy gap is the remaining work; see `benchmarks/baselines_day3-4.md` for the full breakdown.

## Dataset

DrivAerNet++ (MIT, NeurIPS 2024), 8,150 car designs with high-fidelity CFD, fastback / notchback / estateback, ICE and EV underbodies. This project uses only the lightweight tabular files, not the 39 TB of volumetric CFD. License: CC BY-NC 4.0 (non-commercial research and education only).

Honest scope note, verified on the real files: the repo's parametric CSV covers 4,165 designs with 23 geometric parameters, and the full join (params + Cd + Cl + frontal area + official split) is 4,051 designs. The remaining ~4,000 designs exist in the Cd file and splits but have no lightweight parametric row. See `TRACKER.md` and `data/processed/manifest.json` for the verified numbers. Data files are not committed; download them with the ingest scripts below.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.ingest.download_drivaer   # 6 lightweight files, ~2 MB
python -m src.ingest.merge_validate     # merged Parquet + manifest
python -m src.models.train_baselines    # LightGBM, CatBoost, Residual MLP on train, eval on val
```

## Repo layout

```
aerosurrogate/
├── data/
│   ├── raw/          # 6 lightweight files (not committed)
│   └── processed/    # merged + featurised Parquet, manifest.json (not committed)
├── src/
│   ├── ingest/       # download + merge/validate
│   ├── features/     # geometric ratios (slenderness, wheel-to-body, ...)
│   ├── models/       # surrogate ensemble, conformal intervals, ONNX export
│   ├── physics/      # EV range bridge (Cd·A, aero power, range delta)
│   └── optimize/     # Optuna constrained inverse design
├── api/              # FastAPI backend (/predict, /optimize) — in progress
├── ui/               # Streamlit dashboard — in progress
├── benchmarks/       # measured results
├── models/           # exported ONNX models (regenerable, not committed)
├── tests/
├── TRACKER.md        # 2-week plan with daily milestones
└── requirements.txt
```

## Status

Two-week build, 22 Nov – 5 Dec 2026. Day 1 (data) and Days 2–7 (features, baselines, conformal intervals, ONNX export, EV physics, inverse optimizer) are done. Remaining: FastAPI + Streamlit dashboard, accuracy tuning toward the Cd targets, and final evaluation on the official test split. See `TRACKER.md`.

## License

Code is MIT licensed. The DrivAerNet++ dataset is CC BY-NC 4.0, for non-commercial research and education only.
