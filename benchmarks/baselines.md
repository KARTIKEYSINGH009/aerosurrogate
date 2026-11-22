# Baseline results

Measured on the locked DrivAerNet++ validation split. Models were fit on `split == "train"` only; the test split was never read.

- **Data**: data/processed/drivaer_merged.parquet
- **Generated (UTC)**: 2026-11-22 09:24:35
- **Train rows**: 2937
- **Val rows**: 555
- **Features**: 27
- **Random seed**: 42
- **lightgbm**: 4.7.0
- **catboost**: 1.2.10
- **torch**: 2.15.0+cpu
- **scikit-learn**: 1.9.1
- **pandas**: 3.1.1

| Model | Target | Val MAE | Val R2 | R2 (estateback) | R2 (fastback) | R2 (notchback) |
| --- | --- | --- | --- | --- | --- | --- |
| lightgbm | average_cd | 0.0104 | 0.6252 | 0.1037 | 0.6837 | 0.5947 |
| lightgbm | average_cl | 0.0536 | 0.3766 | -0.4823 | 0.1052 | 0.3084 |
| catboost | average_cd | 0.0110 | 0.6082 | 0.1317 | 0.5991 | 0.5537 |
| catboost | average_cl | 0.0565 | 0.3525 | -0.4646 | -0.0635 | 0.2735 |
| residual_mlp | average_cd | 0.0111 | 0.5555 | -0.1759 | 0.6729 | 0.6423 |
| residual_mlp | average_cl | 0.0570 | 0.2489 | -0.9970 | 0.2938 | 0.1949 |

Best validation Cd MAE: **lightgbm** (0.0104).
