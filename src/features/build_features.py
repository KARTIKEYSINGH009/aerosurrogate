"""Feature engineering for AeroSurrogate.

Constructs interpretable geometric ratios and aerodynamic dimensionless indicators
from the 23 raw DrivAerNet++ CAD deformation parameters.

All ratios use absolute reconstructed baseline dimensions (length=4610mm, width=1800mm,
roof_height=1440mm, etc.) to ensure positive, physically meaningful denominators and
prevent division by zero or negative lengths.
Every feature is a pure function of geometry columns with zero target leakage.
"""

from __future__ import annotations
from typing import Sequence
import numpy as np
import pandas as pd

# 23 raw geometric parameter column names
RAW_GEOMETRY_COLUMNS: tuple[str, ...] = (
    "b_ramp_angle",
    "b_diffusor_angle",
    "b_trunklid_angle",
    "c_side_mirrors_rotation",
    "d_rear_window_inclination",
    "d_winscreen_inclination",
    "c_side_mirrors_translate_x",
    "c_side_mirrors_translate_z",
    "d_winscreen_length",
    "d_rear_window_length",
    "e_a_b_c_pillar_thickness",
    "g_trunklid_curvature",
    "g_trunklid_length",
    "h_front_bumper_curvature",
    "h_front_bumper_length",
    "f_door_handles_thickness",
    "f_door_handles_z_position",
    "e_fenders_arch_offset",
    "a_car_length",
    "f_door_handles_x_position",
    "a_car_width",
    "a_car_roof_height",
    "a_car_green_house_angle",
)

# Reference baseline dimensions (mm) for standard full-scale DrivAer model
BASE_LENGTH_MM = 4610.0
BASE_WIDTH_MM = 1800.0
BASE_HEIGHT_MM = 1440.0
BASE_WINDSCREEN_MM = 850.0
BASE_REAR_WINDOW_MM = 700.0
BASE_FRONT_BUMPER_MM = 800.0
BASE_TRUNKLID_MM = 450.0
BASE_WHEELBASE_MM = 2785.0
BASE_WHEEL_RADIUS_MM = 317.5  # ~205/60 R16 standard passenger wheel radius
BASE_PILLAR_THICKNESS_MM = 70.0
BASE_MIRROR_WIDTH_MM = 200.0
BASE_MIRROR_HEIGHT_MM = 120.0
BASE_MIRROR_AREA_MM2 = BASE_MIRROR_WIDTH_MM * BASE_MIRROR_HEIGHT_MM


def slenderness_ratio(length: pd.Series, height: pd.Series) -> pd.Series:
    """Slenderness ratio: total car length divided by total roof height."""
    return (BASE_LENGTH_MM + length) / (BASE_HEIGHT_MM + height)


def width_to_height_ratio(width: pd.Series, height: pd.Series) -> pd.Series:
    """Aspect ratio: total car width divided by total roof height."""
    return (BASE_WIDTH_MM + width) / (BASE_HEIGHT_MM + height)


def frontal_box_area_proxy(width: pd.Series, height: pd.Series) -> pd.Series:
    """Frontal bounding-box area proxy (width * roof_height) in square meters."""
    return ((BASE_WIDTH_MM + width) * (BASE_HEIGHT_MM + height)) / 1e6


def effective_frontal_area_ratio(width: pd.Series, height: pd.Series, pillar_thick: pd.Series) -> pd.Series:
    """Greenhouse-tapered frontal area ratio relative to base reference frontal area."""
    box_area = (BASE_WIDTH_MM + width) * (BASE_HEIGHT_MM + height)
    pillar_adj = 1.0 + (pillar_thick / BASE_PILLAR_THICKNESS_MM) * 0.03
    base_box = BASE_WIDTH_MM * BASE_HEIGHT_MM
    return (box_area * pillar_adj) / base_box


def glazing_to_length_ratio(windscreen: pd.Series, rear_window: pd.Series, length: pd.Series) -> pd.Series:
    """Glazing share: total combined longitudinal windshield and rear window length divided by car length."""
    glazing_len = (BASE_WINDSCREEN_MM + windscreen) + (BASE_REAR_WINDOW_MM + rear_window)
    return glazing_len / (BASE_LENGTH_MM + length)


def front_bumper_to_length_ratio(bumper: pd.Series, length: pd.Series) -> pd.Series:
    """Front overhang ratio: front bumper length divided by total car length."""
    return (BASE_FRONT_BUMPER_MM + bumper) / (BASE_LENGTH_MM + length)


def trunklid_to_length_ratio(trunk: pd.Series, length: pd.Series) -> pd.Series:
    """Rear overhang/deck ratio: trunklid length divided by total car length."""
    return (BASE_TRUNKLID_MM + trunk) / (BASE_LENGTH_MM + length)


def wheel_to_body_ratio(fender_offset: pd.Series, width: pd.Series) -> pd.Series:
    """Wheel arch flare ratio: fender arch offset divided by half-car width (track clearance ratio)."""
    half_width = (BASE_WIDTH_MM + width) * 0.5
    return fender_offset / half_width


def wheel_arch_to_height_ratio(fender_offset: pd.Series, height: pd.Series) -> pd.Series:
    """Wheel arch clearance to roof height: fender arch offset divided by roof height."""
    return fender_offset / (BASE_HEIGHT_MM + height)


def mirror_projected_area_ratio(rot: pd.Series, width: pd.Series, height: pd.Series) -> pd.Series:
    """Mirror frontal area ratio: mirror frontal projected area (accounting for rotation) to car frontal box."""
    cos_rot = np.cos(np.radians(rot))
    mirror_frontal = 2.0 * BASE_MIRROR_AREA_MM2 * np.abs(cos_rot)
    car_frontal = (BASE_WIDTH_MM + width) * (BASE_HEIGHT_MM + height)
    return mirror_frontal / car_frontal


def underbody_diffuser_expansion_ratio(ramp_angle: pd.Series, diffusor_angle: pd.Series) -> pd.Series:
    """Underbody expansion indicator: tangent of underbody diffuser and ramp angles."""
    return np.tan(np.radians(ramp_angle)) + np.tan(np.radians(diffusor_angle))


def boat_tailing_angle_sum(windscreen_inc: pd.Series, rear_window_inc: pd.Series, trunk_angle: pd.Series) -> pd.Series:
    """Aft aerodynamic taper index: cumulative boat-tailing angles (rear window inclination + trunklid angle)."""
    return rear_window_inc + trunk_angle


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    """Compute and append all engineered geometric features to data frame."""
    df = data.copy()
    
    # 1. Slenderness ratio
    df["slenderness_ratio"] = slenderness_ratio(df["a_car_length"], df["a_car_roof_height"])
    
    # 2. Width-to-height ratio
    df["width_to_height_ratio"] = width_to_height_ratio(df["a_car_width"], df["a_car_roof_height"])
    
    # 3. Frontal box area proxy (m^2)
    df["frontal_box_area_proxy"] = frontal_box_area_proxy(df["a_car_width"], df["a_car_roof_height"])
    
    # 4. Effective frontal area ratio
    df["effective_frontal_area_ratio"] = effective_frontal_area_ratio(
        df["a_car_width"], df["a_car_roof_height"], df["e_a_b_c_pillar_thickness"]
    )
    
    # 5. Glazing-to-car-length ratio
    df["glazing_to_length_ratio"] = glazing_to_length_ratio(
        df["d_winscreen_length"], df["d_rear_window_length"], df["a_car_length"]
    )
    
    # 6. Front bumper overhang ratio
    df["front_bumper_to_length_ratio"] = front_bumper_to_length_ratio(
        df["h_front_bumper_length"], df["a_car_length"]
    )
    
    # 7. Trunklid deck ratio
    df["trunklid_to_length_ratio"] = trunklid_to_length_ratio(
        df["g_trunklid_length"], df["a_car_length"]
    )
    
    # 8. Wheel arch flare to half-width ratio
    df["wheel_to_body_ratio"] = wheel_to_body_ratio(
        df["e_fenders_arch_offset"], df["a_car_width"]
    )
    
    # 9. Wheel arch offset to roof height ratio
    df["wheel_arch_to_height_ratio"] = wheel_arch_to_height_ratio(
        df["e_fenders_arch_offset"], df["a_car_roof_height"]
    )
    
    # 10. Mirror projected area to frontal area ratio
    df["mirror_projected_area_ratio"] = mirror_projected_area_ratio(
        df["c_side_mirrors_rotation"], df["a_car_width"], df["a_car_roof_height"]
    )
    
    # 11. Underbody expansion index
    df["underbody_diffuser_expansion_ratio"] = underbody_diffuser_expansion_ratio(
        df["b_ramp_angle"], df["b_diffusor_angle"]
    )
    
    # 12. Boat-tailing angle index
    df["boat_tailing_angle_sum"] = boat_tailing_angle_sum(
        df["d_winscreen_inclination"], df["d_rear_window_inclination"], df["b_trunklid_angle"]
    )
    
    return df


ENGINEERED_FEATURE_NAMES: tuple[str, ...] = (
    "slenderness_ratio",
    "width_to_height_ratio",
    "frontal_box_area_proxy",
    "effective_frontal_area_ratio",
    "glazing_to_length_ratio",
    "front_bumper_to_length_ratio",
    "trunklid_to_length_ratio",
    "wheel_to_body_ratio",
    "wheel_arch_to_height_ratio",
    "mirror_projected_area_ratio",
    "underbody_diffuser_expansion_ratio",
    "boat_tailing_angle_sum",
)


def get_feature_columns() -> list[str]:
    """Return all input feature names (23 raw parameters + 12 engineered features = 35 features)."""
    return list(RAW_GEOMETRY_COLUMNS) + list(ENGINEERED_FEATURE_NAMES)


def build_and_save_featurized_splits(
    merged_parquet_path: str = "data/processed/drivaer_merged.parquet",
    output_dir: str = "data/processed"
) -> dict[str, str]:
    """Load merged parquet, add engineered features, and save train/val/test splits as parquet."""
    df = pd.read_parquet(merged_parquet_path)
    featurized = add_engineered_features(df)
    
    train_df = featurized[featurized["split"] == "train"].reset_index(drop=True)
    val_df = featurized[featurized["split"] == "val"].reset_index(drop=True)
    test_df = featurized[featurized["split"] == "test"].reset_index(drop=True)
    
    paths = {
        "train": f"{output_dir}/drivaer_features_train.parquet",
        "val": f"{output_dir}/drivaer_features_val.parquet",
        "test": f"{output_dir}/drivaer_features_test.parquet",
        "all": f"{output_dir}/drivaer_featurized.parquet",
    }
    
    train_df.to_parquet(paths["train"], index=False)
    val_df.to_parquet(paths["val"], index=False)
    test_df.to_parquet(paths["test"], index=False)
    featurized.to_parquet(paths["all"], index=False)
    
    print(f"Featurized frames saved: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return paths


if __name__ == "__main__":
    build_and_save_featurized_splits()
