from src.optimize.inverse import optimize


def test_scalar_optimize_finds_center() -> None:
    trial = optimize(lambda value: (value - 2.0) ** 2, (0.0, 4.0), n_trials=40)
    assert abs(trial.params["design_value"] - 2.0) < 0.5
