"""Invariants from arXiv:2609.11918 (eqs. 8–11, Algorithm 1)."""

from __future__ import annotations

import jax.numpy as jnp
from jax import random

from datashifts import concept_shift, datashifts, debiased_w_beta, plugin_w_beta, sinkhorn
from datashifts.sinkhorn import cost_matrix, entropic_ot


def test_sinkhorn_marginals():
    key = random.PRNGKey(0)
    k1, k2 = random.split(key)
    n_s, n_t, d = 40, 50, 3
    x_s = random.normal(k1, (n_s, d))
    x_t = random.normal(k2, (n_t, d))
    _, gamma = entropic_ot(x_s, x_t, beta=0.01, n_iter=800)
    row = gamma.sum(axis=1)
    col = gamma.sum(axis=0)
    assert jnp.max(jnp.abs(row - 1.0 / n_s)) < 1e-4
    assert jnp.max(jnp.abs(col - 1.0 / n_t)) < 1e-4
    assert jnp.abs(gamma.sum() - 1.0) < 1e-4


def test_w_beta_same_vs_shifted():
    key = random.PRNGKey(1)
    keys = random.split(key, 4)
    n, d = 64, 2
    mu = random.normal(keys[0], (n, d))
    mu2 = random.normal(keys[1], (n, d))
    nu = random.normal(keys[2], (n, d)) + jnp.array([4.0, 0.0])
    nu2 = random.normal(keys[3], (n, d)) + jnp.array([4.0, 0.0])
    w_same = plugin_w_beta(mu, mu2, beta=0.01, n_iter=600)
    w_shift = plugin_w_beta(mu, nu, beta=0.01, n_iter=600)
    w_shift2 = plugin_w_beta(mu2, nu2, beta=0.01, n_iter=600)
    assert float(w_same) < 0.5 * float(w_shift)
    assert float(w_same) < 0.5 * float(w_shift2)


def test_debiased_estimator_eq9():
    key = random.PRNGKey(2)
    keys = random.split(key, 4)
    n = 80
    # identical 1-d Gaussians: debiased should not explode vs plug-in
    s = random.normal(keys[0], (n, 1))
    t = random.normal(keys[1], (n, 1))
    w_plug = plugin_w_beta(s, t, beta=0.01, n_iter=600)
    w_deb = debiased_w_beta(s, t, beta=0.01, n_iter=600)
    assert float(w_deb) >= 0.0
    assert float(w_deb) < 8.0 * float(w_plug) + 1.0

    # known mean shift of 3 in 1-d
    s2 = random.normal(keys[2], (n, 1))
    t2 = random.normal(keys[3], (n, 1)) + 3.0
    s_cov = debiased_w_beta(s2, t2, beta=0.01, n_iter=600)
    assert 1.5 < float(s_cov) < 5.0


def test_deterministic_labeling_corollary_3_3():
    key = random.PRNGKey(3)
    k1, k2 = random.split(key)
    n, d = 48, 3
    x_s = random.normal(k1, (n, d))
    x_t = random.normal(k2, (n, d)) + 1.5
    y_s = jnp.zeros((n,))
    y_t = jnp.ones((n,))
    s_cpt = concept_shift(x_s, y_s, x_t, y_t, beta=0.01, n_iter=600)
    assert abs(float(s_cpt) - 1.0) < 1e-5

    # same labeling, i.i.d. X
    k3, k4 = random.split(random.PRNGKey(31))
    x_s2 = random.normal(k3, (n, d))
    x_t2 = random.normal(k4, (n, d))
    y_s2 = jnp.zeros((n,))
    y_t2 = jnp.zeros((n,))
    s_cpt0 = concept_shift(x_s2, y_s2, x_t2, y_t2, beta=0.01, n_iter=600)
    assert abs(float(s_cpt0)) < 1e-5


def test_stochastic_synthetic_appendix():
    # X ~ N(0, I_10), Y^S ~ N(||x||, σ^2), Y^T ~ N(||x||+1, σ^2), σ=0.01
    # true S_Cpt^{γ*} = 1.0
    key = random.PRNGKey(4)
    k1, k2, k3, k4 = random.split(key, 4)
    n, d, sigma = 96, 10, 0.01
    x_s = random.normal(k1, (n, d))
    x_t = random.normal(k2, (n, d))
    y_s = jnp.linalg.norm(x_s, axis=1) + sigma * random.normal(k3, (n,))
    y_t = jnp.linalg.norm(x_t, axis=1) + 1.0 + sigma * random.normal(k4, (n,))
    s_cpt = concept_shift(x_s, y_s, x_t, y_t, beta=0.01, n_iter=800)
    assert abs(float(s_cpt) - 1.0) < 0.25


def test_bound_arithmetic():
    key = random.PRNGKey(5)
    k1, k2 = random.split(key)
    n, d = 32, 2
    x_s = random.normal(k1, (n, d))
    x_t = random.normal(k2, (n, d)) + 0.5
    y_s = jnp.zeros((n,))
    y_t = jnp.ones((n,))
    eps_s = 0.2
    out = datashifts(
        x_s,
        y_s,
        x_t,
        y_t,
        beta=0.01,
        n_iter=400,
        lipschitz_ell=1.0,
        lipschitz_ell_prime=1.0,
        lipschitz_h=1.0,
        source_error=eps_s,
    )
    expected = eps_s + float(out.s_cov) + float(out.s_cpt)
    assert abs(float(out.bound) - expected) < 1e-8


def test_coupling_unique_for_beta_positive():
    key = random.PRNGKey(6)
    k1, k2 = random.split(key)
    n, d = 32, 2
    x_s = random.normal(k1, (n, d))
    x_t = random.normal(k2, (n, d))
    c = cost_matrix(x_s, x_t)
    _, g1 = sinkhorn(c, beta=0.05, n_iter=500)
    _, g2 = sinkhorn(c, beta=0.05, n_iter=500)
    assert jnp.max(jnp.abs(g1 - g2)) < 1e-10
