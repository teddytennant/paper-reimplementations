"""Closed-form checks for KL, kappa, gates, rates, quadrature, and J."""

import jax
import jax.numpy as jnp

from spvi.elbo import (
    diagonal_gaussian_kl,
    frozen_path_objective,
    kappa,
    kl_standard_normal,
    objective_j,
    stratified_times,
)
from spvi.gates import rate_link, straight_through_gate


def test_diagonal_gaussian_kl_matches_closed_form():
    mu = jnp.asarray([0.2, -0.4, 0.0])
    var = jnp.asarray([0.5, 1.2, jnp.e])
    got = kl_standard_normal(mu, var)
    expected = 0.5 * jnp.sum(-jnp.log(var) + var + mu**2 - 1.0)
    assert jnp.allclose(got, expected, atol=1e-6, rtol=1e-6)
    # N(0, I) special case, including a coordinate that is exactly standard.
    via_general = diagonal_gaussian_kl(mu, var, jnp.zeros_like(mu), jnp.ones_like(var))
    assert jnp.allclose(via_general, expected, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(kl_standard_normal(jnp.zeros((3,)), jnp.ones((3,))), 0.0, atol=1e-6)

    mu_p = jnp.asarray([0.1, -0.2, 0.3])
    v_p = jnp.asarray([2.0, 0.3, 1.5])
    got_p = diagonal_gaussian_kl(mu, var, mu_p, v_p)
    expected_p = 0.5 * jnp.sum(
        jnp.log(v_p / var) + (var + (mu - mu_p) ** 2) / v_p - 1.0
    )
    assert jnp.allclose(got_p, expected_p, atol=1e-6, rtol=1e-6)
    # Hand value: q = N(0, e) vs N(0, 1) on one coordinate.
    one = kl_standard_normal(jnp.asarray([0.0]), jnp.asarray([jnp.e]))
    assert jnp.allclose(one, 0.5 * (jnp.e - 2.0), atol=1e-6)


def test_kappa_matches_hand_example():
    # Equal rates and equal standard marks: integrand is zero.
    zero = kappa(1.0, 1.0, jnp.asarray([0.0]), jnp.asarray([1.0]), jnp.asarray([0.0]), jnp.asarray([1.0]))
    assert jnp.allclose(zero, 0.0, atol=1e-6)

    # lambda_q = 2, lambda_theta = 1, identical N(0, I) marks.
    # kappa = 2 log 2 - 2 + 1 = 2 log 2 - 1.
    simple = kappa(2.0, 1.0, jnp.asarray([0.0, 0.0]), jnp.ones((2,)), jnp.zeros((2,)), jnp.ones((2,)))
    assert jnp.allclose(simple, 2.0 * jnp.log(2.0) - 1.0, atol=1e-6)

    lambda_q = jnp.asarray(0.4)
    lambda_theta = jnp.asarray(0.8)
    mu_q = jnp.asarray([0.1, -0.2])
    v_q = jnp.asarray([0.5, 1.5])
    mu_p = jnp.asarray([0.0, 0.3])
    v_p = jnp.asarray([1.0, 0.7])
    kl = 0.5 * jnp.sum(jnp.log(v_p / v_q) + (v_q + (mu_q - mu_p) ** 2) / v_p - 1.0)
    expected = (
        lambda_q * jnp.log(lambda_q / lambda_theta)
        - lambda_q
        + lambda_theta
        + lambda_q * kl
    )
    got = kappa(lambda_q, lambda_theta, mu_q, v_q, mu_p, v_p)
    assert jnp.allclose(got, expected, atol=1e-6, rtol=1e-6)
    # Independent numeric value of that hand KL, not copied from the helper.
    assert jnp.allclose(kl, 0.5 * (
        jnp.log(1.0 / 0.5) + (0.5 + 0.1**2) / 1.0 - 1.0
        + jnp.log(0.7 / 1.5) + (1.5 + (-0.2 - 0.3) ** 2) / 0.7 - 1.0
    ))


def test_straight_through_forward_and_gradient():
    logit = jnp.asarray([0.5, -1.2, 2.0, 0.0, -0.3])
    zeta = jnp.asarray([0.1, -0.3, 0.7, -2.0, 0.3])
    tau = jnp.asarray(0.7)
    gates = straight_through_gate(logit, zeta, tau)
    hard = (logit + zeta > 0).astype(jnp.float32)
    assert jnp.all((gates == 0.0) | (gates == 1.0))
    assert jnp.allclose(gates, hard)

    def sum_gates(logit_p):
        return jnp.sum(straight_through_gate(logit_p, zeta, tau))

    grad = jax.grad(sum_gates)(logit)
    soft = jax.nn.sigmoid((logit + zeta) / tau)
    expected = soft * (1.0 - soft) / tau
    assert jnp.allclose(grad, expected, atol=1e-6, rtol=1e-6)

    # The gradient matches the soft branch alone, so it does not use the hard threshold.
    def sum_soft(logit_p):
        return jnp.sum(jax.nn.sigmoid((logit_p + zeta) / tau))

    assert jnp.allclose(grad, jax.grad(sum_soft)(logit), atol=1e-6, rtol=1e-6)
    # Flip a hard label without changing the soft input and the analytic grad is unchanged.
    flipped = jnp.where(logit + zeta > 0, -1.0, 1.0)
    assert jnp.any(flipped != (logit + zeta))
    assert jnp.allclose(expected, soft * (1.0 - soft) / tau)


def test_rate_link_matches_formula_and_bounds():
    ell = jnp.linspace(-6.0, 6.0, 13)
    got = rate_link(ell)
    a = jnp.log(1e-3)
    b = jnp.log(10.0)
    expected = jnp.exp(a + (b - a) * jax.nn.sigmoid(ell))
    assert jnp.allclose(got, expected, atol=1e-6, rtol=1e-6)
    wide = jnp.concatenate([ell, jnp.asarray([-40.0, -20.0, 20.0, 40.0])])
    wide_got = rate_link(wide)
    assert jnp.all(wide_got > 1e-3)
    assert jnp.all(wide_got < 10.0)
    # Saturated logits sit next to the endpoints, still strictly inside.
    assert float(rate_link(-40.0)) < 1.1e-3
    assert float(rate_link(40.0)) > 9.9


def test_quadrature_weights_and_frozen_objective():
    t_left = jnp.asarray([0.0, 1.0, 2.5])
    t_right = jnp.asarray([1.0, 2.5, 4.0])
    horizon = 4.0
    uniforms = jax.random.uniform(jax.random.PRNGKey(4), (3, 5))
    xi, weights = stratified_times(t_left, t_right, uniforms)
    assert jnp.allclose(jnp.sum(weights), horizon, atol=1e-6)
    delta = t_right - t_left
    # Every probe sits in its stratum of the interval.
    for k in range(3):
        for s in range(5):
            left = t_left[k] + delta[k] * s / 5.0
            right = t_left[k] + delta[k] * (s + 1) / 5.0
            assert float(xi[k, s]) > float(left) - 1e-6
            assert float(xi[k, s]) < float(right) + 1e-6
            assert jnp.allclose(weights[k, s], delta[k] / 5.0)

    y = jnp.asarray([[0.2, -0.1], [0.0, 0.4], [1.0, -0.3]])
    z_obs = jnp.asarray([[0.1, -0.2, 9.0], [0.2, 0.1, 8.0], [0.7, -0.4, 7.0]])
    mu0 = jnp.asarray([0.3, -0.2])
    v0 = jnp.asarray([0.8, 1.4])
    lambda_q = jnp.asarray([0.2, 0.5, 0.3])
    obs_var = 0.01
    w_kl = 1.7
    beta = 0.3
    quad_lambda_q = lambda_q[:, None]
    quad_lambda_theta = jnp.asarray([[0.4, 0.5], [0.2, 0.6], [0.3, 0.1]])
    quad_mu_q = jnp.asarray([0.1, -0.2])[None, None, :] * jnp.ones((3, 2, 1))
    quad_v_q = jnp.asarray([0.5, 1.1])[None, None, :] * jnp.ones((3, 2, 1))
    quad_mu_p = jnp.asarray([[0.0, 0.2], [0.1, -0.1]])[None, :, :] * jnp.ones((3, 1, 1))
    quad_v_p = jnp.asarray([[1.0, 0.7], [1.2, 0.4]])[None, :, :] * jnp.ones((3, 1, 1))
    # Broadcast mark parameters to (K, M, mark).
    quad_mu_q = jnp.broadcast_to(jnp.asarray([0.1, -0.2]), (3, 2, 2))
    quad_v_q = jnp.broadcast_to(jnp.asarray([0.6, 1.3]), (3, 2, 2))
    quad_mu_p = jnp.asarray([0.0, 0.2])[None, None, :] + 0.05 * jnp.arange(2)[None, :, None]
    quad_v_p = jnp.asarray([1.0, 0.8])[None, None, :] * jnp.asarray([1.0, 1.1])[None, :, None]
    quad_weights = delta[:, None] / 2.0 * jnp.ones((3, 2))

    got = frozen_path_objective(
        y, z_obs, mu0, v0, lambda_q, delta,
        quad_lambda_q, quad_lambda_theta,
        quad_mu_q, quad_v_q, quad_mu_p, quad_v_p,
        quad_weights, obs_var, w_kl, beta,
    )
    mean = z_obs[:, :2]
    loglik = -0.5 * jnp.sum(jnp.log(2.0 * jnp.pi * obs_var) + (y - mean) ** 2 / obs_var)
    kl0 = 0.5 * jnp.sum(-jnp.log(v0) + v0 + mu0**2 - 1.0)
    kl_mark = 0.5 * jnp.sum(
        jnp.log(quad_v_p / quad_v_q) + (quad_v_q + (quad_mu_q - quad_mu_p) ** 2) / quad_v_p - 1.0,
        axis=-1,
    )
    integrand = (
        quad_lambda_q * jnp.log(quad_lambda_q / quad_lambda_theta)
        - quad_lambda_q
        + quad_lambda_theta
        + quad_lambda_q * kl_mark
    )
    k_dyn = jnp.sum(quad_weights * integrand)
    sparsity = jnp.sum(lambda_q * delta)
    expected = loglik - w_kl * (kl0 + k_dyn) - beta * sparsity
    assert jnp.allclose(got, expected, atol=1e-5, rtol=1e-5)
    # The helper that only combines the three terms agrees on the same numbers.
    combined = objective_j(loglik, kl0 + k_dyn, lambda_q, delta, w_kl, beta)
    assert jnp.allclose(combined, expected, atol=1e-6)
