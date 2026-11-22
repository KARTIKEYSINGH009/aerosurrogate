import pandas as pd

from src.models.features import build_features


def test_build_features() -> None:
    data = pd.DataFrame({"cd": [0.25, 0.26], "area": [2.1, 2.2]})
    features = build_features(data, ["cd", "area"])
    assert list(features.columns) == ["cd", "area"]
