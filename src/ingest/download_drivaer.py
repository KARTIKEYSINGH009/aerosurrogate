"""Download the DrivAer source CSV files."""

from pathlib import Path


def download(output_dir: Path) -> list[Path]:
    """Download all source files into output_dir.

    Add the project-specific URLs and splitting logic here when data sources
    and licensing information are available.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    return []
