import numpy as np

from fitr import fit_mll, fitr_minimize


def test_fit_mll_decreases_nll():
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 1.0, size=(12, 2))
    y = np.sin(2.0 * np.pi * x[:, 0]) + 0.1 * x[:, 1]
    fitted = fit_mll(x, y, steps=40, lr=0.05, seed=0)
    assert float(fitted["nll_after"]) < float(fitted["nll_before"])


def test_fitr_beats_initial_design_on_quadratic():
    def f(x):
        return (x[0] - 0.2) ** 2 + 8.0 * (x[1] - 0.8) ** 2

    out = fitr_minimize(
        f,
        d=2,
        n_init=6,
        n_steps=10,
        seed=0,
        fit_steps=0,
    )
    assert out["best_y"] < float(np.min(out["y"][:6]))
    assert out["x"].shape == (16, 2)
    assert out["y"].shape == (16,)


def test_proposals_stay_in_unit_cube():
    def f(x):
        return (x[0] - 0.2) ** 2 + 8.0 * (x[1] - 0.8) ** 2

    out = fitr_minimize(
        f,
        d=2,
        n_init=6,
        n_steps=10,
        seed=0,
        fit_steps=0,
    )
    assert np.all(out["x"] >= 0.0)
    assert np.all(out["x"] <= 1.0)
