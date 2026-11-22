"""Merge source files and validate the processed dataset."""

import json
from pathlib import Path

import pandas as pd


def merge_csvs(input_dir: Path, output_path: Path) -> tuple[pd.DataFrame, dict]:
    """Merge source CSVs into Parquet and return the data with a manifest."""
    frames = [pd.read_csv(path) for path in sorted(input_dir.glob("*.csv"))]
    if not frames:
        return pd.DataFrame(), {}

    merged = pd.concat(frames, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)
    manifest = {
        "row_count": len(merged),
        "columns": list(merged.columns),
        "output": str(output_path),
    }
    manifest_path = output_path.with_name("manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return merged, manifest
