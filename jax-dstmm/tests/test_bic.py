"""The implementation-based parameter count of equation (30) and the BIC of Section 3.7."""

import jax
import numpy as np
import pytest

from dstmm import bic, log_likelihood, n_params, paper_style_truth, sample_dstmm


def test_paper_simulation_architecture_count():
    """Worked by hand for the Section 4.1 architecture: p = 20, (r1, r2) = (5, 2),
    (K1, K2) = (2, 2), active set {1}, common nu.

    Layer 1: (2-1) + 2*(20*5 + 2*20 - 2*1/2) + 2*20 = 1 + 260 + 40 = 301.
    Layer 2: (2-1) + 2*(5*2 + 2*5 - 2*1/2) = 1 + 38 = 39.
    Plus d_nu = 1. Total 341.
    """
    assert n_params((20, 5, 2), (2, 2), active_set=(1,), df_mode="common") == 341


def test_skewness_active_set_adds_k_times_r():
    """A layer in A contributes exactly K_l r_{l-1} more, per equation (30)."""
    dims, ks = (20, 5, 2), (2, 2)
    off = n_params(dims, ks, active_set=(), df_mode="common")
    lay1 = n_params(dims, ks, active_set=(1,), df_mode="common")
    both = n_params(dims, ks, active_set=(1, 2), df_mode="common")
    assert lay1 - off == ks[0] * dims[0]
    assert both - lay1 == ks[1] * dims[1]


def test_rotational_correction_is_equation_19():
    """The loading block is r_{l-1} r_l - r_l (r_l - 1)/2, not r_{l-1} r_l."""
    dims, ks = (9, 4), (3,)
    got = n_params(dims, ks, active_set=(), df_mode="common")
    by_hand = (3 - 1) + 3 * (9 * 4 + 2 * 9 - 4 * 3 // 2) + 1
    assert got == by_hand
    naive = (3 - 1) + 3 * (9 * 4 + 2 * 9) + 1
    assert naive - got == ks[0] * dims[1] * (dims[1] - 1) // 2


def test_df_mode_contributions():
    """d_nu is 1 for a common nu, K_1 for a first-layer-specific one, and
    prod_l K_l for pathway-specific."""
    dims, ks = (20, 5, 2), (3, 4)
    base = n_params(dims, ks, active_set=(1,), df_mode="common")
    assert n_params(dims, ks, active_set=(1,), df_mode="layer1") - base == 3 - 1
    assert n_params(dims, ks, active_set=(1,), df_mode="path") - base == 12 - 1
    with pytest.raises(ValueError):
        n_params(dims, ks, df_mode="nonsense")


def test_bic_definition():
    """BIC = -2 l(Theta_hat) + d_code log n, so smaller is preferred."""
    truth = paper_style_truth(kappa=1.0, nu=6.0)
    y, _, _, _ = sample_dstmm(jax.random.key(0), truth, 200)
    d = n_params(truth.dims, truth.ks, active_set=(1,), df_mode="common")
    got = float(bic(truth, y, active_set=(1,), df_mode="common"))
    ref = -2.0 * float(log_likelihood(truth, y)) + d * np.log(200)
    assert abs(got - ref) < 1e-8
