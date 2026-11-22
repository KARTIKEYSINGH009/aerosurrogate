# Antigravity prompt — AeroSurrogate Day 3-4 (features + baselines)

Paste this into Antigravity with the `aerosurrogate` repo open.

---

You are working on AeroSurrogate, a physics-informed ML surrogate that predicts a car's drag (Cd), lift (Cl) and frontal area from geometric design parameters in under 2 ms, replacing hours of CFD. It is a portfolio project for EV/F1/software jobs. Dataset: DrivAerNet++ (MIT, NeurIPS 2024), CC BY-NC 4.0, lightweight tabular files only, no volumetric CFD.

## Current state (Day 1 done, verified on real data)

- `data/processed/drivaer_merged.parquet` and `data/processed/manifest.json` already exist. Do NOT re-download or re-merge unless the files are missing.
- The usable merged set is **4,051 designs**, not 8,150. The parametric CSV has 4,165 rows and **23 geometric parameters**, not 26. The Cd file has Cd only, no Cl.
- Official splits in the merged set: train 2,937 / val 555 / test 559. Full official splits are train 5,819 / val 1,148 / test 1,154. Body type is the first letter of the design ID: F fastback, N notchback, E estateback.
- Full Day 1 findings are in `TRACKER.md`. Read it first.
- `requirements.txt` already lists everything (polars, lightgbm, catboost, torch CPU, shap, optuna, onnx). Use the existing `.venv` if present, otherwise create it and `pip install -r requirements.txt`.

## Your task: Day 3-4, features + baselines

1. **Verify Day 1 state first.** Load `data/processed/drivaer_merged.parquet`, confirm 4,051 rows and the split counts above. If anything disagrees, stop and report the mismatch instead of guessing.
2. **Feature engineering.** Create `src/features/build_features.py`. Engineer interpretable geometric ratios from the 23 parameters, including slenderness (length / height or similar), wheel-to-body ratios, and frontal-area ratios. Keep every feature a pure function of the geometry columns, no target leakage. Document each feature with a one-line comment. Save the featurised train/val/test frames as Parquet in `data/processed/` (e.g. `features.parquet`), and record the feature list in `manifest.json` or a small `features.md`.
3. **Train three baselines, all on the official train split only.** Targets: Cd (primary), Cl, frontal area.
   - LightGBM regressor
   - CatBoost regressor
   - PyTorch Residual MLP (CPU, small, with early stopping on val)
   Keep hyperparameters simple and sensible for a baseline. No tuning yet, that is Day 5-7.
4. **Evaluate on the val split only. Never touch the test split, that is Day 13-14.** Report for each model and each target: MAE and R² overall, plus Cd MAE and R² per body type (fastback / notchback / estateback).
5. **Log results.** Write `benchmarks/baselines_day3-4.md` with a table of the numbers above, the exact commands used, and the time each model took to train. Numbers must be measured, never claimed or estimated.

## Rules

- Laptop only, CPU inference, no GPU, no paid data.
- Use the official split column. Never re-split the data and never train on val or test.
- Every benchmark number must be measured and logged. If a model fails, report the error and what you tried, do not fake a number.
- Keep the code runnable from the repo root with the `.venv`. Add a short `python -m` entry point for training (e.g. `python -m src.models.train_baselines`).
- Do not change Day 1 files (`src/ingest/`, `data/processed/drivaer_merged.parquet`, `manifest.json`) except to append the feature list.

## When done

Reply with a short summary: the feature list, and a table of Cd MAE and R² per model on val, overall and per body type. Flag anything that looks wrong (for example a model much worse than the others) instead of hiding it.
