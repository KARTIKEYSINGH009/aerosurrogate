import pandas as pd
import pytest

from src.models.features import (
    FRONT_BUMPER_LENGTH_OVER_CAR_LENGTH,
    GLAZING_LENGTH_OVER_CAR_LENGTH,
    LENGTH_TO_ROOF_HEIGHT,
    RAW_GEOMETRY_COLUMNS,
    WIDTH_TO_ROOF_HEIGHT,
    add_geometric_features,
    build_model_matrix,
    front_bumper_to_length_ratio,
    glazing_to_length_ratio,
    slenderness,
    width_to_height_ratio,
)


def _geometry_frame() -> pd.DataFrame:
    """Two designs with hand-checkable dimensions."""
    return pd.DataFrame(
        {
            "a_car_length": [4.0, 5.0],
            "a_car_width": [2.0, 2.5],
            "a_car_roof_height": [1.6, 2.0],
            "d_winscreen_length": [1.0, 1.2],
            "d_rear_window_length": [0.5, 0.8],
            "h_front_bumper_length": [0.4, 0.7],
        }
    )


def test_slenderness_and_width_to_height() -> None:
    frame = _geometry_frame()

    assert slenderness(frame["a_car_length"], frame["a_car_roof_height"]).tolist() == pytest.approx(
        [4.0 / 1.6, 5.0 / 2.0]
    )
    assert width_to_height_ratio(
        frame["a_car_width"], frame["a_car_roof_height"]
    ).tolist() == pytest.approx([2.0 / 1.6, 2.5 / 2.0])


def test_glazing_and_bumper_ratios_normalise_by_car_length() -> None:
    frame = _geometry_frame()

    assert glazing_to_length_ratio(
        frame["d_winscreen_length"],
        frame["d_rear_window_length"],
        frame["a_car_length"],
    ).tolist() == pytest.approx([(1.0 + 0.5) / 4.0, (1.2 + 0.8) / 5.0])
    assert front_bumper_to_length_ratio(
        frame["h_front_bumper_length"], frame["a_car_length"]
    ).tolist() == pytest.approx([0.4 / 4.0, 0.7 / 5.0])


def test_add_geometric_features_keeps_raw_params_and_is_length_invariant() -> None:
    frame = _geometry_frame()
    stretched = frame.copy()
    stretched["a_car_length"] = frame["a_car_length"] * 2.0

    enriched = add_geometric_features(frame)
    stretched_enriched = add_geometric_features(stretched)

    # Raw parameters are preserved alongside the new columns.
    for column in frame.columns:
        assert enriched[column].tolist() == frame[column].tolist()
    assert LENGTH_TO_ROOF_HEIGHT in enriched.columns
    assert WIDTH_TO_ROOF_HEIGHT in enriched.columns
    assert GLAZING_LENGTH_OVER_CAR_LENGTH in enriched.columns
    assert FRONT_BUMPER_LENGTH_OVER_CAR_LENGTH in enriched.columns

    # Length-normalised ratios halve when only the car length doubles, the
    # width-to-height ratio is untouched, and slenderness doubles with length.
    assert stretched_enriched[LENGTH_TO_ROOF_HEIGHT].tolist() == pytest.approx(
        [2.0 * value for value in enriched[LENGTH_TO_ROOF_HEIGHT]]
    )
    assert stretched_enriched[WIDTH_TO_ROOF_HEIGHT].tolist() == pytest.approx(
        enriched[WIDTH_TO_ROOF_HEIGHT].tolist()
    )
    assert stretched_enriched[FRONT_BUMPER_LENGTH_OVER_CAR_LENGTH].tolist() == pytest.approx(
        [0.5 * value for value in enriched[FRONT_BUMPER_LENGTH_OVER_CAR_LENGTH]]
    )
    assert stretched_enriched[GLAZING_LENGTH_OVER_CAR_LENGTH].tolist() != pytest.approx(
        enriched[GLAZING_LENGTH_OVER_CAR_LENGTH].tolist()
    )


def test_zero_denominator_is_rejected() -> None:
    frame = _geometry_frame()
    frame.loc[1, "a_car_roof_height"] = 0.0

    with pytest.raises(ValueError, match="denominator contains zeros"):
        add_geometric_features(frame)


def test_missing_geometry_column_is_rejected() -> None:
    frame = _geometry_frame().drop(columns=["h_front_bumper_length"])

    with pytest.raises(KeyError, match="h_front_bumper_length"):
        add_geometric_features(frame)


def test_build_model_matrix_returns_all_raw_and_derived_features() -> None:
    design = {column: [1.0, 1.0] for column in RAW_GEOMETRY_COLUMNS}
    design.update(_geometry_frame().to_dict("list"))
    matrix = build_model_matrix(pd.DataFrame(design))

    assert matrix.shape == (2, len(RAW_GEOMETRY_COLUMNS) + 4)
    assert set(RAW_GEOMETRY_COLUMNS).issubset(matrix.columns)
    assert matrix[FRONT_BUMPER_LENGTH_OVER_CAR_LENGTH].tolist() == pytest.approx([0.1, 0.14])
