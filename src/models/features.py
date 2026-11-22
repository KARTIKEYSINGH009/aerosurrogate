"""Feature engineering helpers for the aerodynamic surrogate.

The raw DrivAerNet++ design parameters are dimensionless geometry descriptors, so
they carry no absolute scale information. The derived features below express the
body proportions that aerodynamicists reason about (slenderness, width-to-height,
glazing share of the body length, bumper share of the body length) as ratios, which
are invariant to the parameterisation of the source CSV.

Every function here is pure: it reads the columns it needs and returns a new
object without mutating its input. Division by a zero or missing denominator is a
hard error rather than a silent infinity, so a broken input can never leak into a
training run.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# Raw geometric parameters from the merged DrivAerNet++ tabular dataset.
RAW_GEOMETRY_COLUMNS: tuple[str, ...] = (
    "a_car_length",
    "a_car_width",
    "a_car_roof_height",
    "a_car_green_house_angle",
    "b_ramp_angle",
    "b_diffusor_angle",
    "b_trunklid_angle",
    "c_side_mirrors_rotation",
    "c_side_mirrors_translate_x",
    "c_side_mirrors_translate_z",
    "d_rear_window_inclination",
    "d_winscreen_inclination",
    "d_winscreen_length",
    "d_rear_window_length",
    "e_a_b_c_pillar_thickness",
    "e_fenders_arch_offset",
    "f_door_handles_thickness",
    "f_door_handles_z_position",
    "f_door_handles_x_position",
    "g_trunklid_curvature",
    "g_trunklid_length",
    "h_front_bumper_curvature",
    "h_front_bumper_length",
)

# Columns consumed by the derived features.
CAR_LENGTH = "a_car_length"
CAR_WIDTH = "a_car_width"
ROOF_HEIGHT = "a_car_roof_height"
WINDSCREEN_LENGTH = "d_winscreen_length"
REAR_WINDOW_LENGTH = "d_rear_window_length"
FRONT_BUMPER_LENGTH = "h_front_bumper_length"

_GEOMETRY_INPUT_COLUMNS: tuple[str, ...] = (
    CAR_LENGTH,
    CAR_WIDTH,
    ROOF_HEIGHT,
    WINDSCREEN_LENGTH,
    REAR_WINDOW_LENGTH,
    FRONT_BUMPER_LENGTH,
)

LENGTH_TO_ROOF_HEIGHT = "length_to_roof_height"
WIDTH_TO_ROOF_HEIGHT = "width_to_roof_height"
GLAZING_LENGTH_OVER_CAR_LENGTH = "glazing_length_over_car_length"
FRONT_BUMPER_LENGTH_OVER_CAR_LENGTH = "front_bumper_length_over_car_length"

DERIVED_GEOMETRY_COLUMNS: tuple[str, ...] = (
    LENGTH_TO_ROOF_HEIGHT,
    WIDTH_TO_ROOF_HEIGHT,
    GLAZING_LENGTH_OVER_CAR_LENGTH,
    FRONT_BUMPER_LENGTH_OVER_CAR_LENGTH,
)

# Regression targets and the grouping column used for per-body-style reporting.
TARGET_COLUMNS: tuple[str, ...] = ("average_cd", "average_cl")
GROUP_COLUMN = "body_style"


def _as_float_series(values: pd.Series | Sequence[float], name: str) -> pd.Series:
    """Coerce values to a float Series, preserving the input index when present."""
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    try:
        return series.astype("float64").rename(name)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc


def _safe_ratio(
    numerator: pd.Series | Sequence[float],
    denominator: pd.Series | Sequence[float],
    name: str,
) -> pd.Series:
    """Divide two aligned series, rejecting zero and missing denominators."""
    num = _as_float_series(numerator, name)
    den = _as_float_series(denominator, name)
    if len(num) != len(den):
        raise ValueError(f"{name}: numerator and denominator lengths differ")
    if den.isna().any():
        raise ValueError(f"{name}: denominator contains missing values")
    if bool((den == 0).any()):
        raise ValueError(f"{name}: denominator contains zeros")
    return (num / den).rename(name)


def slenderness(
    length: pd.Series | Sequence[float],
    roof_height: pd.Series | Sequence[float],
) -> pd.Series:
    """Car length relative to roof height (a_car_length / a_car_roof_height)."""
    return _safe_ratio(length, roof_height, LENGTH_TO_ROOF_HEIGHT)


def width_to_height_ratio(
    width: pd.Series | Sequence[float],
    roof_height: pd.Series | Sequence[float],
) -> pd.Series:
    """Car width relative to roof height (a_car_width / a_car_roof_height)."""
    return _safe_ratio(width, roof_height, WIDTH_TO_ROOF_HEIGHT)


def glazing_to_length_ratio(
    windscreen_length: pd.Series | Sequence[float],
    rear_window_length: pd.Series | Sequence[float],
    car_length: pd.Series | Sequence[float],
) -> pd.Series:
    """Windscreen plus rear window length, normalised by overall car length."""
    glazing = _as_float_series(windscreen_length, "glazing_length") + _as_float_series(
        rear_window_length, "glazing_length"
    )
    return _safe_ratio(glazing, car_length, GLAZING_LENGTH_OVER_CAR_LENGTH)


def front_bumper_to_length_ratio(
    front_bumper_length: pd.Series | Sequence[float],
    car_length: pd.Series | Sequence[float],
) -> pd.Series:
    """Front bumper length relative to overall car length."""
    return _safe_ratio(
        front_bumper_length, car_length, FRONT_BUMPER_LENGTH_OVER_CAR_LENGTH
    )


def add_geometric_features(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``data`` with the derived ratio features appended.

    Raises ``KeyError`` if a raw column needed by a ratio is absent.
    """
    missing = [column for column in _GEOMETRY_INPUT_COLUMNS if column not in data.columns]
    if missing:
        raise KeyError(f"Missing geometry columns: {missing}")

    features = data.copy()
    features[LENGTH_TO_ROOF_HEIGHT] = slenderness(
        features[CAR_LENGTH], features[ROOF_HEIGHT]
    )
    features[WIDTH_TO_ROOF_HEIGHT] = width_to_height_ratio(
        features[CAR_WIDTH], features[ROOF_HEIGHT]
    )
    features[GLAZING_LENGTH_OVER_CAR_LENGTH] = glazing_to_length_ratio(
        features[WINDSCREEN_LENGTH], features[REAR_WINDOW_LENGTH], features[CAR_LENGTH]
    )
    features[FRONT_BUMPER_LENGTH_OVER_CAR_LENGTH] = front_bumper_to_length_ratio(
        features[FRONT_BUMPER_LENGTH], features[CAR_LENGTH]
    )
    return features


def model_feature_columns() -> tuple[str, ...]:
    """Ordered feature list handed to every model: raw parameters, then ratios."""
    return RAW_GEOMETRY_COLUMNS + DERIVED_GEOMETRY_COLUMNS


def build_model_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Build the finite, numeric design matrix used for training and inference."""
    missing = [column for column in RAW_GEOMETRY_COLUMNS if column not in data.columns]
    if missing:
        raise KeyError(f"Missing raw geometry columns: {missing}")

    engineered = add_geometric_features(data)
    columns = model_feature_columns()
    matrix = engineered[list(columns)].astype("float64")
    non_finite = ~np.isfinite(matrix.to_numpy())
    if non_finite.any():
        bad = sorted({column for column in matrix.columns[non_finite.any(axis=0)]})
        raise ValueError(f"Non-finite feature values in columns: {bad}")
    return matrix


def build_features(data: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Select and numerically validate model input features."""
    features = data[list(columns)].copy()
    if features.isna().any().any():
        raise ValueError("Model features contain missing values")
    return features
