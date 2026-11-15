"""Mechanics checks for restarted FPRLL and the density-mixture master."""

import math
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np

from fprll.aggregate import FixedShare
from fprll.core import (
    GRID_N,
    density,
    indicator_cost,
    lazy_sample,
    log_density,
    make_grid,
    potential_grad,
    potential_hess,
    sample_index,
    total_variation,
)
from fprll.simulate import FPRLL, dyadic_scales, rollout_scale, run


def test_density_integrates_on_documented_grid():
    centers, dx = make_grid(GRID_N)
    assert GRID_N == 1024
    for cum_grad in (0.0, -1.5):
        q = density(centers, cum_grad, gamma=1.0, sigma=1.0)
        mass = float(np.sum(q) * dx)
        assert abs(mass - 1.0) < 0.05, (cum_grad, mass)


def test_hessian_factor_matches_finite_difference():
    x = 0.25
    cum_grad = -0.7
    gamma = 1.3
    sigma = 1.0
    eps = 1e-5

    def spec_grad(point):
        return cum_grad + (1.0 / gamma) * (sigma * point + 2.0 * point / (1.0 - point * point))

    fd = (spec_grad(x + eps) - spec_grad(x - eps)) / (2.0 * eps)
    grad = spec_grad(x)
    p = -grad
    for mu in (1.0, 0.4):
        # nu(p) = (2*mu)^{-1} * exp(-|p| / mu), the paper's 1D Laplace.
        nu = math.exp(-abs(p) / mu) / (2.0 * mu)
        q = float(np.asarray(density(x, cum_grad, gamma, sigma, laplace_scale=mu)))
        factor = q / nu
        assert abs(factor - abs(-fd)) < 1e-4
        expected_log = -math.log(2.0 * mu) - abs(p) / mu + math.log(abs(-fd))
        got = float(np.asarray(log_density(x, cum_grad, gamma, sigma, laplace_scale=mu)))
        assert abs(got - expected_log) < 1e-6
    lib_grad = float(np.asarray(potential_grad(x, cum_grad, gamma, sigma)))
    assert abs(lib_grad - grad) < 1e-12

    # Product Laplace: log nu(p) = -d*log(2*mu) - ||p||_1 / mu.
    x2 = np.array([0.2, -0.35])
    cum2 = np.array([-0.5, 0.25])
    mu = 0.7
    grad2 = potential_grad(x2, cum2, gamma, sigma)
    hess2 = potential_hess(x2, cum2, gamma, sigma)
    log_nu = -2.0 * math.log(2.0 * mu) - float(np.sum(np.abs(-grad2))) / mu
    log_det = float(np.sum(np.log(np.abs(-hess2))))
    got2 = float(np.asarray(log_density(x2, cum2, gamma, sigma, laplace_scale=mu)))
    assert abs(got2 - (log_nu + log_det)) < 1e-8


def test_lazy_sample_marginal_and_switch_rate():
    centers, dx = make_grid(GRID_N)
    gamma, sigma = 1.0, 1.0
    q = density(centers, 0.0, gamma, sigma)
    p = density(centers, -0.8, gamma, sigma)
    tv = total_variation(p, q, dx)
    assert 0.05 < tv < 0.8
    rng = np.random.default_rng(7)
    n = 4000
    ys = np.empty(n, dtype=np.float64)
    switches = 0
    for i in range(n):
        idx = sample_index(q, rng)
        new_idx, _kept = lazy_sample(idx, q, p, rng)
        ys[i] = centers[new_idx]
        if indicator_cost(centers[idx], centers[new_idx], GRID_N):
            switches += 1
    # 32 equal bins: a 1024-bin histogram TV is dominated by multinomial noise.
    nbins = 32
    emp, _ = np.histogram(ys, bins=nbins, range=(-1.0, 1.0))
    emp = emp / emp.sum()
    true = p.reshape(nbins, -1).sum(axis=1)
    true = true / true.sum()
    emp_tv = 0.5 * float(np.sum(np.abs(emp - true)))
    assert emp_tv < 0.15, emp_tv
    assert abs(switches / n - tv) < 0.1, (switches / n, tv)


def test_indicator_is_one_iff_bins_differ():
    centers, _dx = make_grid(GRID_N)
    assert indicator_cost(centers[10], centers[10], GRID_N) == 0
    assert indicator_cost(centers[10], centers[11], GRID_N) == 1
    assert indicator_cost(centers[0], centers[-1], GRID_N) == 1
    # A point inside the same cell as centers[20] must not count as a switch.
    dx = 2.0 / GRID_N
    same = centers[20] + 0.2 * dx
    assert indicator_cost(centers[20], same, GRID_N) == 0
    other = centers[20] + 1.2 * dx
    assert indicator_cost(centers[20], other, GRID_N) == 1


def test_large_horizon_switches_less_than_unit_on_constant_comparator():
    T = 64
    g = -np.ones(T)
    small = rollout_scale(g, horizon=1, gamma=1.0, sigma=1.0, seed=0)
    large = rollout_scale(g, horizon=32, gamma=1.0, sigma=1.0, seed=0)
    assert large["switches"] < small["switches"]


def test_large_gamma_tracks_positive_corner():
    steps = 40
    small = FPRLL(gamma=0.2, sigma=1.0)
    large = FPRLL(gamma=5.0, sigma=1.0)
    for _ in range(steps):
        small.observe(-1.0)
        large.observe(-1.0)
    mean_small = small.mean_action(n_samples=2000, seed=1)
    mean_large = large.mean_action(n_samples=2000, seed=1)
    assert mean_large > mean_small
    assert mean_large > 0.95
    assert mean_small < 0.9


def test_fixed_share_tiny_eta_flat_losses_stay_close():
    # alg:fs-scale. Large tau keeps the fixed-share injection small.
    start = np.array([0.55, 0.15, 0.15, 0.15])
    fs = FixedShare(n=4, tau=2000.0, D=0.01, eta=1e-8, weights=start)
    assert fs.active
    prev = fs.weights()
    fs.update(np.full(4, 0.4))
    assert float(np.sum(np.abs(fs.weights() - prev))) < 0.02
    assert abs(float(fs.weights().sum()) - 1.0) < 1e-12


def test_mixture_regret_not_worse_than_best_scale():
    """Short horizon, comparator switches between the corners every 16 rounds.

    Indicator-augmented regret of the mixture is compared with the best
    dyadic scale after averaging seeds 0 through 7. Additive slack is 8.
    On these seeds the averaged excess is about 4.2, from sampling and from
    the master not putting all mass on one scale by time T=32.
    """
    T = 32
    seg = 16
    u = np.array([1.0 if (t // seg) % 2 == 0 else -1.0 for t in range(T)])
    g = -u
    assert dyadic_scales(T) == [1, 2, 4, 8, 16, 32]
    mix = []
    scales = []
    for seed in range(8):
        out = run(g, gamma=1.0, sigma=1.0, lam=0.25, comparators=u, seed=seed)
        mix.append(out["mixture_regret"])
        scales.append(out["scale_regret"])
    mix_mean = float(np.mean(mix))
    scale_mean = np.mean(np.vstack(scales), axis=0)
    best = float(np.min(scale_mean))
    slack = 8.0
    assert mix_mean <= best + slack, (mix_mean, best, scale_mean, slack)
