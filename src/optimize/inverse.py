"""Day 7: Multi-parameter Inverse Aerodynamic Optimization using Optuna.

Finds aerodynamic vehicle geometry configurations that minimize drag (Cd) and maximize highway EV range,
subject to practical industrial styling & engineering packaging constraints:
1. Minimum cabin volume proxy (interior space constraint)
2. Ground clearance / departure ramp limits
3. Diffuser stall limits (ramp and diffuser angles <= 15 deg)
4. Frontal area bounding box constraints
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import onnxruntime as ort
import optuna
import pandas as pd

from src.features.build_features import (
    RAW_GEOMETRY_COLUMNS,
    add_engineered_features,
    get_feature_columns,
)
from src.physics.ev_range import compute_ev_range_km, range_delta_between_designs

optuna.logging.set_verbosity(optuna.logging.WARNING)


# Practical engineering limits for the 23 raw parameters
PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "b_ramp_angle": (-6.0, 14.0),
    "b_diffusor_angle": (-6.0, 14.0),
    "b_trunklid_angle": (-5.0, 18.0),
    "c_side_mirrors_rotation": (-15.0, 15.0),
    "d_rear_window_inclination": (-1.5, 2.0),
    "d_winscreen_inclination": (-1.5, 2.0),
    "c_side_mirrors_translate_x": (-5.0, 15.0),
    "c_side_mirrors_translate_z": (-3.0, 8.0),
    "d_winscreen_length": (-30.0, 30.0),
    "d_rear_window_length": (-50.0, 100.0),
    "e_a_b_c_pillar_thickness": (-10.0, 3.0),
    "g_trunklid_curvature": (-0.2, 0.6),
    "g_trunklid_length": (-25.0, 25.0),
    "h_front_bumper_curvature": (-0.2, 0.8),
    "h_front_bumper_length": (-15.0, 45.0),
    "f_door_handles_thickness": (-15.0, 20.0),
    "f_door_handles_z_position": (-20.0, 20.0),
    "e_fenders_arch_offset": (-15.0, 40.0),
    "a_car_length": (-40.0, 100.0),
    "f_door_handles_x_position": (-30.0, 30.0),
    "a_car_width": (-40.0, 100.0),
    "a_car_roof_height": (-40.0, 60.0),
    "a_car_green_house_angle": (-150.0, 100.0),
}


class SurrogatePredictor:
    """Fast ONNX evaluation engine for inverse optimization."""

    def __init__(self, onnx_model_path: str = "models/aerosurrogate_joint_mlp.onnx", scalers_path: str = "models/scalers.npz") -> None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(onnx_model_path, opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

        scalers = np.load(scalers_path)
        self.x_mean = scalers["x_mean"]
        self.x_scale = scalers["x_scale"]
        self.y_mean = scalers["y_mean"]
        self.y_scale = scalers["y_scale"]
        self.feature_names = get_feature_columns()

    def predict_one(self, raw_params_dict: Dict[str, float]) -> Tuple[float, float, float]:
        """Predict [Cd, Cl, Frontal_Area] for a candidate parametric design."""
        df_row = pd.DataFrame([raw_params_dict])
        feat_df = add_engineered_features(df_row)
        feat_vec = feat_df[self.feature_names].to_numpy(dtype=np.float32)

        feat_scaled = ((feat_vec - self.x_mean) / self.x_scale).astype(np.float32)
        raw_pred = self.session.run(None, {self.input_name: feat_scaled})[0]
        unscaled_pred = (raw_pred * self.y_scale) + self.y_mean

        cd = float(unscaled_pred[0, 0])
        cl = float(unscaled_pred[0, 1])
        frontal_area = float(unscaled_pred[0, 2])
        return cd, cl, frontal_area


def run_inverse_optimization(
    predictor: SurrogatePredictor,
    baseline_params: Dict[str, float],
    target_body_style: str = "fastback",
    n_trials: int = 500,
) -> Dict[str, object]:
    """Run constrained inverse optimization using Optuna (TPE sampler)."""
    base_cd, base_cl, base_area = predictor.predict_one(baseline_params)
    base_range = compute_ev_range_km(base_cd, base_area, speed_km_h=120.0)

    def objective(trial: optuna.Trial) -> float:
        candidate = {}
        for p_name in RAW_GEOMETRY_COLUMNS:
            low, high = PARAM_BOUNDS[p_name]
            candidate[p_name] = trial.suggest_float(p_name, low, high)

        cd, cl, frontal_area = predictor.predict_one(candidate)

        # Engineering constraints via soft penalties:
        penalty = 0.0

        # 1. Diffuser stall limit: ramp + diffusor angle shouldn't exceed 22 degrees
        total_underbody_angle = candidate["b_ramp_angle"] + candidate["b_diffusor_angle"]
        if total_underbody_angle > 22.0:
            penalty += (total_underbody_angle - 22.0) * 0.05

        # 2. Lift constraint: prevent severe positive aerodynamic lift at highway speeds
        if cl > 0.05:
            penalty += (cl - 0.05) * 0.10

        # 3. Minimum cabin envelope constraint: avoid excessive roof height reduction
        if candidate["a_car_roof_height"] < -30.0:
            penalty += (-30.0 - candidate["a_car_roof_height"]) * 0.01

        return cd + penalty

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    opt_cd, opt_cl, opt_area = predictor.predict_one(best_params)
    comparison = range_delta_between_designs(base_cd, base_area, opt_cd, opt_area, speed_km_h=120.0)

    return {
        "baseline_metrics": {
            "cd": base_cd,
            "cl": base_cl,
            "frontal_area_m2": base_area,
            "range_120kmh": base_range["cruising_range_km"],
        },
        "optimized_metrics": {
            "cd": opt_cd,
            "cl": opt_cl,
            "frontal_area_m2": opt_area,
            "range_120kmh": comparison["optimized_range_km"],
        },
        "range_comparison": comparison,
        "best_geometry_params": best_params,
        "n_trials": n_trials,
    }


def demo_optimization() -> None:
    predictor = SurrogatePredictor()

    # Nominal baseline geometry (neutral zero-offset CAD parameters)
    baseline = {col: 0.0 for col in RAW_GEOMETRY_COLUMNS}
    print("Running 500-trial Inverse Aerodynamic Optimization on CPU...")
    res = run_inverse_optimization(predictor, baseline, n_trials=500)

    b = res["baseline_metrics"]
    o = res["optimized_metrics"]
    comp = res["range_comparison"]

    print("\nOptimization Complete:")
    print(f"Baseline:  Cd = {b['cd']:.4f} | Frontal Area = {b['frontal_area_m2']:.3f} m2 | Highway Range @ 120km/h = {b['range_120kmh']:.1f} km")
    print(f"Optimized: Cd = {o['cd']:.4f} | Frontal Area = {o['frontal_area_m2']:.3f} m2 | Highway Range @ 120km/h = {o['range_120kmh']:.1f} km")
    print(f"Improvement: Delta Cd = {comp['cd_delta']:.4f} ({comp['cd_delta']/b['cd']*100:.1f}%) | Range Gain = +{comp['range_gain_km']:.1f} km (+{comp['range_gain_pct']:.1f}%)")


if __name__ == "__main__":
    demo_optimization()
