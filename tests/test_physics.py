from src.physics.ev_range import aero_power_w, drag_area, range_delta_km


def test_drag_area() -> None:
    assert abs(drag_area(0.25, 2.0) - 0.5) < 1e-12


def test_aero_power_is_positive() -> None:
    assert aero_power_w(0.25, 2.0, 30.0) > 0


def test_range_delta_decreases() -> None:
    assert range_delta_km(400, 0.30, 0.33) < 0
