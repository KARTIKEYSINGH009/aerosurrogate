"""Physics calculations for automotive aerodynamics and EV highway energy consumption.

Provides first-principles aerodynamic force, drag area (CdA), required mechanical/electrical
power, and estimated EV range delta under standardized driving cycles (WLTP highway / constant speed).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

AIR_DENSITY_SEA_LEVEL = 1.225  # kg/m^3 (ISA at 15°C, 1013.25 hPa)
GRAVITY = 9.80665  # m/s^2


@dataclass(frozen=True)
class EVVehicleSpec:
    """Standard C/D-segment EV reference specification (e.g. Tesla Model 3 / Hyundai Ioniq 6)."""
    battery_capacity_kwh: float = 75.0
    usable_soc_fraction: float = 0.90  # 90% usable battery buffer
    curb_mass_kg: float = 1850.0
    rolling_resistance_coeff: float = 0.0085  # Low-rolling-resistance EV tire
    powertrain_efficiency: float = 0.92  # Inverter + motor + gearbox combined efficiency
    auxiliary_power_w: float = 500.0  # HVAC, infotainment, computer baseline draw
    base_cd: float = 0.250
    base_frontal_area_m2: float = 2.30


def drag_area(cd: float, frontal_area_m2: float) -> float:
    """Aerodynamic effective drag area CdA in m^2."""
    if cd <= 0 or frontal_area_m2 <= 0:
        raise ValueError("Cd and frontal area must be strictly positive.")
    return cd * frontal_area_m2


def aero_drag_force_n(
    cd: float,
    frontal_area_m2: float,
    speed_m_s: float,
    air_density: float = AIR_DENSITY_SEA_LEVEL,
) -> float:
    """Aerodynamic resistive drag force F_aero = 0.5 * rho * CdA * v^2 in Newtons."""
    cda = drag_area(cd, frontal_area_m2)
    return 0.5 * air_density * cda * (speed_m_s ** 2)


def rolling_resistance_force_n(
    mass_kg: float,
    crr: float = 0.0085,
    road_grade_rad: float = 0.0,
) -> float:
    """Tire rolling resistance force F_rr = Crr * m * g * cos(grade) in Newtons."""
    import numpy as np
    return crr * mass_kg * GRAVITY * float(np.cos(road_grade_rad))


def steady_state_power_w(
    cd: float,
    frontal_area_m2: float,
    speed_km_h: float,
    spec: EVVehicleSpec | None = None,
) -> Dict[str, float]:
    """Calculate steady-state mechanical resistance power and electrical battery power draw in Watts."""
    if spec is None:
        spec = EVVehicleSpec()

    v_m_s = speed_km_h / 3.6
    f_aero = aero_drag_force_n(cd, frontal_area_m2, v_m_s)
    f_rr = rolling_resistance_force_n(spec.curb_mass_kg, spec.rolling_resistance_coeff)

    p_aero_mech = f_aero * v_m_s
    p_rr_mech = f_rr * v_m_s
    p_total_mech = p_aero_mech + p_rr_mech

    # Electrical power drawn from the high-voltage traction battery
    p_electrical = (p_total_mech / spec.powertrain_efficiency) + spec.auxiliary_power_w

    return {
        "speed_km_h": speed_km_h,
        "speed_m_s": v_m_s,
        "aero_drag_force_n": f_aero,
        "rolling_force_n": f_rr,
        "p_aero_mech_w": p_aero_mech,
        "p_rolling_mech_w": p_rr_mech,
        "p_total_mech_w": p_total_mech,
        "p_electrical_battery_w": p_electrical,
        "aero_fraction_of_mechanical_power": p_aero_mech / p_total_mech,
    }


def compute_ev_range_km(
    cd: float,
    frontal_area_m2: float,
    speed_km_h: float = 120.0,
    spec: EVVehicleSpec | None = None,
) -> Dict[str, float]:
    """Compute constant cruising range in km and specific consumption in Wh/km."""
    if spec is None:
        spec = EVVehicleSpec()

    power_dict = steady_state_power_w(cd, frontal_area_m2, speed_km_h, spec)
    p_elec_w = power_dict["p_electrical_battery_w"]

    usable_energy_wh = spec.battery_capacity_kwh * spec.usable_soc_fraction * 1000.0

    # Wh per km = (Electrical Watts / speed_km_h)
    consumption_wh_per_km = p_elec_w / speed_km_h
    total_range_km = usable_energy_wh / consumption_wh_per_km

    return {
        "speed_km_h": speed_km_h,
        "cd": cd,
        "frontal_area_m2": frontal_area_m2,
        "cda_m2": drag_area(cd, frontal_area_m2),
        "consumption_wh_per_km": consumption_wh_per_km,
        "cruising_range_km": total_range_km,
        "aero_power_kw": power_dict["p_aero_mech_w"] / 1000.0,
        "battery_power_kw": p_elec_w / 1000.0,
        "aero_power_share_pct": power_dict["aero_fraction_of_mechanical_power"] * 100.0,
    }


def range_delta_between_designs(
    baseline_cd: float,
    baseline_frontal_area: float,
    optimized_cd: float,
    optimized_frontal_area: float,
    speed_km_h: float = 120.0,
    spec: EVVehicleSpec | None = None,
) -> Dict[str, float]:
    """Calculate the delta in range, energy consumption, and power saved between baseline and optimized designs."""
    base = compute_ev_range_km(baseline_cd, baseline_frontal_area, speed_km_h, spec)
    opt = compute_ev_range_km(optimized_cd, optimized_frontal_area, speed_km_h, spec)

    range_gain_km = opt["cruising_range_km"] - base["cruising_range_km"]
    range_gain_pct = (range_gain_km / base["cruising_range_km"]) * 100.0
    energy_saved_wh_per_km = base["consumption_wh_per_km"] - opt["consumption_wh_per_km"]
    power_saved_kw = base["battery_power_kw"] - opt["battery_power_kw"]

    return {
        "speed_km_h": speed_km_h,
        "baseline_cd": baseline_cd,
        "optimized_cd": optimized_cd,
        "cd_delta": optimized_cd - baseline_cd,
        "baseline_range_km": base["cruising_range_km"],
        "optimized_range_km": opt["cruising_range_km"],
        "range_gain_km": range_gain_km,
        "range_gain_pct": range_gain_pct,
        "energy_saved_wh_per_km": energy_saved_wh_per_km,
        "power_saved_kw": power_saved_kw,
    }
