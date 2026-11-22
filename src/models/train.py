"""Train LightGBM, CatBoost and residual-MLP baselines for Cd and Cl.

Cd and Cl are modelled as two separate regression targets. Models are fitted on
``split == "train"`` only and scored on ``split == "val"`` only; the test split is
never read. There is deliberately no validation-based early stopping and no
hyper-parameter search, so the validation numbers remain an honest hold-out
estimate rather than a tuned score.

Run with: ``python -m src.models.train``
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import catboost
import lightgbm
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.features import GROUP_COLUMN, TARGET_COLUMNS, build_model_matrix

SEED = 42
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
MODEL_NAMES: tuple[str, ...] = ("lightgbm", "catboost", "residual_mlp")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "drivaer_merged.parquet"
BENCHMARK_PATH = PROJECT_ROOT / "benchmarks" / "baselines.md"

# CPU only: the PyTorch baseline never touches a GPU.
DEVICE = torch.device("cpu")


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def fit_lightgbm(x_train: np.ndarray, y_train: np.ndarray) -> lightgbm.LGBMRegressor:
    """Fit a gradient-boosted tree baseline with a fixed, untuned configuration."""
    model = lightgbm.LGBMRegressor(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=SEED,
        deterministic=True,
        force_row_wise=True,
        n_jobs=-1,
        verbose=-1,
    )
    return model.fit(x_train, y_train)


def fit_catboost(x_train: np.ndarray, y_train: np.ndarray) -> catboost.CatBoostRegressor:
    """Fit a CatBoost baseline with a fixed, untuned configuration."""
    model = catboost.CatBoostRegressor(
        iterations=800,
        learning_rate=0.05,
        depth=6,
        loss_function="RMSE",
        random_seed=SEED,
        thread_count=-1,
        verbose=0,
        allow_writing_files=False,
    )
    return model.fit(x_train, y_train)


class ResidualBlock(nn.Module):
    """Pre-norm residual MLP block."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear_in = nn.Linear(hidden_dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(inputs)
        hidden = self.dropout(self.activation(self.linear_in(hidden)))
        hidden = self.linear_out(hidden)
        return inputs + hidden


class ResidualMLP(nn.Module):
    """Small residual MLP mapping standardised design features to a standardised target."""

    def __init__(
        self, input_dim: int, hidden_dim: int, n_blocks: int, dropout: float
    ) -> None:
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.input_layer(inputs)
        hidden = self.blocks(hidden)
        hidden = self.output_norm(hidden)
        return self.output_layer(hidden).squeeze(-1)


@dataclass(frozen=True)
class MLPConfig:
    """Fixed training configuration for the residual MLP baseline."""

    hidden_dim: int = 128
    n_blocks: int = 2
    dropout: float = 0.1
    epochs: int = 300
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = SEED


class ResidualMLPRegressor:
    """sklearn-style wrapper around :class:`ResidualMLP` with fixed-epoch training."""

    def __init__(self, config: MLPConfig | None = None) -> None:
        self.config = config or MLPConfig()
        self._feature_scaler = StandardScaler()
        self._target_scaler = StandardScaler()
        self._network: ResidualMLP | None = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "ResidualMLPRegressor":
        config = self.config
        set_seed(config.seed)

        x_scaled = self._feature_scaler.fit_transform(x_train).astype("float32")
        y_scaled = self._target_scaler.fit_transform(y_train.reshape(-1, 1))
        y_scaled = y_scaled.reshape(-1).astype("float32")

        dataset = TensorDataset(
            torch.from_numpy(x_scaled), torch.from_numpy(y_scaled)
        )
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(config.seed),
        )

        self._network = ResidualMLP(
            input_dim=x_scaled.shape[1],
            hidden_dim=config.hidden_dim,
            n_blocks=config.n_blocks,
            dropout=config.dropout,
        ).to(DEVICE)
        optimiser = torch.optim.AdamW(
            self._network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        loss_fn = nn.MSELoss()

        # Fixed epoch budget: no early stopping on the validation split.
        for _ in range(config.epochs):
            self._network.train()
            for features, targets in loader:
                features = features.to(DEVICE)
                targets = targets.to(DEVICE)
                optimiser.zero_grad()
                loss = loss_fn(self._network(features), targets)
                loss.backward()
                optimiser.step()
        return self

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        if self._network is None:
            raise RuntimeError("ResidualMLPRegressor must be fitted before predicting")
        x_scaled = self._feature_scaler.transform(x_test).astype("float32")
        self._network.eval()
        with torch.no_grad():
            predictions = self._network(torch.from_numpy(x_scaled).to(DEVICE))
        restored = self._target_scaler.inverse_transform(
            predictions.cpu().numpy().reshape(-1, 1)
        )
        return restored.reshape(-1)


def set_seed(seed: int = SEED) -> None:
    """Seed every RNG used by the baselines."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fit_model(
    name: str, x_train: np.ndarray, y_train: np.ndarray
) -> lightgbm.LGBMRegressor | catboost.CatBoostRegressor | ResidualMLPRegressor:
    """Dispatch to the named baseline."""
    if name == "lightgbm":
        return fit_lightgbm(x_train, y_train)
    if name == "catboost":
        return fit_catboost(x_train, y_train)
    if name == "residual_mlp":
        return ResidualMLPRegressor().fit(x_train, y_train)
    raise ValueError(f"Unknown model name: {name!r}")


# --------------------------------------------------------------------------- #
# Data and metrics
# --------------------------------------------------------------------------- #
def load_split(data: pd.DataFrame, split: str) -> pd.DataFrame:
    """Return the rows for one official DrivAerNet++ split, without re-splitting."""
    if "split" not in data.columns:
        raise KeyError("Dataset has no 'split' column; cannot select an official split")
    subset = data.loc[data["split"] == split].reset_index(drop=True)
    if subset.empty:
        raise ValueError(f"No rows found for split == {split!r}")
    return subset


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    """Return (MAE, R^2) for a single target."""
    mae = float(np.mean(np.abs(predicted - actual)))
    r2 = float(r2_score(actual, predicted))
    return mae, r2


def per_body_style_r2(
    groups: pd.Series, actual: np.ndarray, predicted: np.ndarray
) -> dict[str, float]:
    """R^2 computed inside each body style (NaN if a style has fewer than 2 rows)."""
    scores: dict[str, float] = {}
    labels = groups.to_numpy()
    for style in sorted(pd.unique(labels)):
        mask = labels == style
        scores[str(style)] = (
            float(r2_score(actual[mask], predicted[mask])) if mask.sum() > 1 else float("nan")
        )
    return scores


@dataclass
class RunResult:
    """One (model, target) evaluation on the validation split."""

    model: str
    target: str
    mae: float
    r2: float
    per_body_r2: dict[str, float]
    fit_seconds: float

    def body_r2(self, style: str) -> float:
        return self.per_body_r2.get(style, float("nan"))


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _format_metric(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.4f}"


def write_benchmark(
    results: list[RunResult],
    metadata: dict[str, object],
    output_path: Path = BENCHMARK_PATH,
) -> None:
    """Write the measured validation results as a markdown table."""
    body_styles = sorted({style for result in results for style in result.per_body_r2})
    header = ["Model", "Target", "Val MAE", "Val R2"] + [
        f"R2 ({style})" for style in body_styles
    ]
    rows = [
        [result.model, result.target, _format_metric(result.mae), _format_metric(result.r2)]
        + [_format_metric(result.body_r2(style)) for style in body_styles]
        for result in results
    ]

    lines = [
        "# Baseline results",
        "",
        "Measured on the locked DrivAerNet++ validation split. Models were fit on "
        "`split == \"train\"` only; the test split was never read.",
        "",
    ]
    lines += [f"- **{key}**: {value}" for key, value in metadata.items()]
    lines += [
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]

    cd_results = [result for result in results if result.target == "average_cd"]
    if cd_results:
        best = min(cd_results, key=lambda result: result.mae)
        lines += [
            "",
            f"Best validation Cd MAE: **{best.model}** ({_format_metric(best.mae)}).",
        ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_metadata(
    data_path: Path, train: pd.DataFrame, val: pd.DataFrame, feature_count: int
) -> dict[str, object]:
    """Describe the run so every reported number is traceable."""
    try:
        data_label = data_path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        data_label = str(data_path)
    return {
        "Data": data_label,
        "Generated (UTC)": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "Train rows": len(train),
        "Val rows": len(val),
        "Features": feature_count,
        "Random seed": SEED,
        "lightgbm": lightgbm.__version__,
        "catboost": catboost.__version__,
        "torch": torch.__version__,
        "scikit-learn": sklearn.__version__,
        "pandas": pd.__version__,
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Merged parquet path")
    parser.add_argument(
        "--output", type=Path, default=BENCHMARK_PATH, help="Markdown results path"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.data.exists():
        raise FileNotFoundError(
            f"Dataset not found: {args.data}. Expected the merged DrivAerNet++ parquet "
            "from the ingestion step (data/processed/drivaer_merged.parquet)."
        )

    set_seed(SEED)
    data = pd.read_parquet(args.data)
    train = load_split(data, TRAIN_SPLIT)
    val = load_split(data, VAL_SPLIT)

    missing_targets = [name for name in TARGET_COLUMNS if name not in data.columns]
    if missing_targets:
        raise KeyError(f"Missing target columns: {missing_targets}")

    x_train_frame = build_model_matrix(train)
    x_val_frame = build_model_matrix(val)
    x_train = x_train_frame.to_numpy(dtype="float64")
    x_val = x_val_frame.to_numpy(dtype="float64")

    print(
        f"train rows={len(train)} val rows={len(val)} features={x_train.shape[1]}",
        flush=True,
    )

    results: list[RunResult] = []
    for model_name in MODEL_NAMES:
        for target in TARGET_COLUMNS:
            actual = val[target].to_numpy(dtype="float64")
            started = time.perf_counter()
            model = fit_model(model_name, x_train, train[target].to_numpy(dtype="float64"))
            predicted = np.asarray(model.predict(x_val), dtype="float64")
            elapsed = time.perf_counter() - started

            mae, r2 = regression_metrics(actual, predicted)
            per_body = per_body_style_r2(val[GROUP_COLUMN], actual, predicted)
            results.append(
                RunResult(
                    model=model_name,
                    target=target,
                    mae=mae,
                    r2=r2,
                    per_body_r2=per_body,
                    fit_seconds=elapsed,
                )
            )
            print(
                f"{model_name:>12} {target:>10} val_MAE={mae:.4f} val_R2={r2:.4f} "
                f"fit={elapsed:.1f}s",
                flush=True,
            )

    write_benchmark(
        results,
        collect_metadata(args.data, train, val, x_train.shape[1]),
        args.output,
    )

    print("\nCd on validation split:")
    cd_results = [result for result in results if result.target == "average_cd"]
    for result in cd_results:
        print(f"  {result.model:<12} MAE={result.mae:.4f} R2={result.r2:.4f}")
    best = min(cd_results, key=lambda result: result.mae)
    print(f"Best Cd model by val MAE: {best.model}")
    print(f"Results written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
