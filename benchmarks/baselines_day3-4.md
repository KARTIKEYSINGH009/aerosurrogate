# Baseline Surrogate Benchmarks (Day 3-4)

- **Evaluation Date**: 2026-11-22 09:31:40 UTC
- **Dataset**: DrivAerNet++ Merged Parquet (`data/processed/drivaer_merged.parquet`)
- **Splits Used**: Train = 2,937 samples, Val = 555 samples (Official split; test split of 559 samples strictly untouched)
- **Hardware / Compute**: Laptop CPU only (no GPU, single machine)
- **Feature Set**: 35 total features (23 raw geometric CAD deformation parameters + 12 engineered aerodynamic ratios)
- **Execution Command**: `python -m src.models.train_baselines`

---

## 1. Overall Validation Results Across All Targets

| Model | Target | Val MAE | Val R² | Training Time (s) |
| :--- | :--- | :--- | :--- | :--- |
| **LightGBM** | Cd (Drag) | **0.0103** | **0.6398** | 10.52s |
| **CatBoost** | Cd (Drag) | 0.0110 | 0.6207 | 17.83s |
| **PyTorch Residual MLP** | Cd (Drag) | 0.0126 | 0.5341 | 39.78s (stopped @ ep 12) |
| **LightGBM** | Cl (Lift) | **0.0532** | **0.3822** | 9.49s |
| **CatBoost** | Cl (Lift) | 0.0562 | 0.3685 | 11.04s |
| **PyTorch Residual MLP** | Cl (Lift) | 0.0612 | 0.2969 | 38.47s (stopped @ ep 13) |
| **LightGBM** | Frontal Area ($m^2$) | 0.0044 | 0.9988 | 10.97s |
| **CatBoost** | Frontal Area ($m^2$) | **0.0036** | **0.9992** | 8.57s |
| **PyTorch Residual MLP** | Frontal Area ($m^2$) | 0.0041 | 0.9991 | 113.05s (stopped @ ep 72) |

---

## 2. Drag Coefficient (Cd) Breakdown by Body Type

Validation set distribution: Estateback = 215, Fastback = 177, Notchback = 163 (Total = 555).

| Model | Overall MAE | Overall R² | Fastback MAE (R²) | Notchback MAE (R²) | Estateback MAE (R²) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LightGBM** | **0.0103** | **0.6398** | 0.0083 (0.6664) | 0.0076 (0.5991) | 0.0140 (0.1687) |
| **CatBoost** | 0.0110 | 0.6207 | 0.0093 (0.6068) | 0.0086 (0.5430) | 0.0143 (0.1788) |
| **PyTorch Residual MLP** | 0.0126 | 0.5341 | 0.0105 (0.4999) | 0.0102 (0.4255) | 0.0160 (0.0126) |

---

## 3. Findings & Flagged Observations

1. **Flagged Issue - Estateback Within-Subgroup R² Collapse ($R^2 \approx 0.01 - 0.18$)**:
   - While overall Cd $R^2$ is **0.6398** and estateback absolute MAE is reasonably low (0.0140 vs overall 0.0103), the within-subgroup $R^2$ for estateback drops sharply.
   - **Root Cause**: Estateback has a blunt, squared rear-end where wake separation occurs right at the trailing edges regardless of small slant variations. In contrast, notchback ($R^2 \approx 0.54 - 0.60$) and fastback ($R^2 \approx 0.61 - 0.67$) have sensitive separation locations that correlate strongly with trunk and rear-window angles. Future iterations (Day 5-7) may benefit from body-type stratified trees or explicit estateback interaction terms.
2. **Flagged Issue - Lift Coefficient (Cl) Poor Predictability ($R^2 \approx 0.30 - 0.38$)**:
   - Aerodynamic lift across all three models shows modest $R^2$. Cl depends strongly on localized vortex breakdown and pressure suction on the underside diffusor/ground-effect, which is highly non-linear and harder to capture with generic baseline hyper-parameters.
3. **Frontal Area Prediction is Near-Perfect ($R^2 > 0.998$, $\text{MAE} < 0.0045 \text{ m}^2$)**:
   - Frontal area is accurately recovered from the combination of vehicle width, height, and pillar offsets by both tree models and the MLP.
4. **Model Comparison**:
   - **LightGBM** is the best performing baseline for Cd (MAE 0.0103, $R^2$ 0.6398) and trains in ~10 seconds on CPU.
   - **CatBoost** is closely competitive for Cd (MAE 0.0110, $R^2$ 0.6207) and slightly leads on Frontal Area ($R^2$ 0.9992).
   - **PyTorch Residual MLP** trains fast on CPU with early stopping (12-15 epochs for aero targets), achieving $R^2 = 0.5341$ on Cd, but trails tree ensembles on this tabular geometry setting.
