from pathlib import Path

from src.ingest.merge_validate import merge_csvs


def test_merge_empty_directory(tmp_path: Path) -> None:
    merged, manifest = merge_csvs(tmp_path, tmp_path / "merged.parquet")
    assert merged.empty
    assert manifest == {}
