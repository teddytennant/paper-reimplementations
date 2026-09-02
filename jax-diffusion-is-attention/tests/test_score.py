"""Sec. II-B: eq (4) is the score of eq (2), and it carries eq (11) inside it."""

import jax
import jax.numpy as jnp
import pytest

from diffusion_is_attention import (
    mixture_log_density,
    mixture_weights,
    optimal_denoiser,
    posterior_mean,
    posterior_mean_via_closed_form,
    probability_flow_rhs,
    reverse_ode,
    score,
    sigma_from_alpha,
)

ALPHAS = [0.9, 0.5, 0.1, 0.01]


def toy(seed=0, n=10, d=4):
    k1, k2 = jax.random.split(jax.random.key(seed))
    return jax.random.normal(k1, (n, d)), jax.random.normal(k2, (d,)) * 1.5


@pytest.mark.parametrize("alpha", ALPHAS)
def test_eq4_matches_autodiff_of_eq2(alpha):
    data, y = toy()
    auto = jax.grad(mixture_log_density)(y, data, alpha)
    assert jnp.allclose(score(y, data, alpha), auto, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_eq4_matches_central_finite_differences(alpha):
    """Analytic score against a numerical gradient of log p_t, no autodiff."""
    data, y = toy(seed=1)
    h = 1e-5
    eye = jnp.eye(y.shape[-1])
    fd = jnp.array(
        [
            (
                mixture_log_density(y + h * e, data, alpha)
                - mixture_log_density(y - h * e, data, alpha)
            )
            / (2 * h)
            for e in eye
        ]
    )
    analytic = score(y, data, alpha)
    assert jnp.allclose(analytic, fd, rtol=2e-6, atol=2e-6 * float(jnp.max(jnp.abs(analytic))) + 1e-8)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_eq11_is_the_posterior_mean_of_eq4(alpha):
    """The Tweedie relation. Eq (11) at sigma^2 = (1 - alpha)/alpha equals E[x0|y].

    This is the connection the paper flags right after eq (15): (11) can be read
    off (4) and (5), which is what [8] stopped short of doing.
    """
    data, y = toy(seed=2, n=16, d=6)
    from_score = posterior_mean(y, data, alpha)
    from_closed_form = posterior_mean_via_closed_form(y, data, alpha)
    assert jnp.allclose(from_score, from_closed_form, rtol=1e-11, atol=1e-11)

    # Same statement without the helper: eq (5) weights are the eq (11) weights.
    direct = optimal_denoiser(y / jnp.sqrt(alpha), data, sigma_from_alpha(alpha))
    assert jnp.allclose(from_score, direct, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_eq5_weights_are_a_distribution(alpha):
    data, y = toy(seed=3)
    w = mixture_weights(y, data, alpha)
    assert jnp.all(w >= 0.0)
    assert float(jnp.sum(w)) == pytest.approx(1.0, abs=1e-12)


def test_eq2_is_a_normalized_density():
    """Numerically integrate p_t over a 1D dataset and get 1."""
    data = jnp.array([[-1.0], [0.4], [2.0]])
    grid = jnp.linspace(-12.0, 12.0, 40001)[:, None]
    p = jnp.exp(jax.vmap(mixture_log_density, in_axes=(0, None, None))(grid, data, 0.5))
    mass = jnp.trapezoid(p, grid[:, 0])
    assert float(mass) == pytest.approx(1.0, abs=1e-8)


@pytest.mark.parametrize("alpha", [0.9, 0.5, 0.1])
def test_eq3_fixed_point_for_a_single_signal(alpha):
    """With |D| = 1 the flow (3) stalls exactly at s / sqrt(alpha_t).

    Score is (sqrt(a) s - y)/(1 - a), so y + s_t(y) = 0 at y = s/sqrt(a). That
    point is where the reverse process would deposit the sample.
    """
    s = jnp.array([[0.7, -1.3, 0.2]])
    y_star = s[0] / jnp.sqrt(alpha)
    assert jnp.allclose(probability_flow_rhs(y_star, s, alpha), 0.0, atol=1e-12)
    off = y_star + jnp.array([0.3, 0.0, -0.2])
    assert jnp.linalg.norm(probability_flow_rhs(off, s, alpha)) > 1e-3


def test_eq3_run_backward_moves_toward_the_data():
    data, y = toy(seed=4, n=8, d=3)
    y = y * 4.0
    alphas = jnp.linspace(0.05, 0.95, 60)
    traj = reverse_ode(y, data, alphas, gammas=0.05, dt=1.0)
    start = jnp.min(jnp.linalg.norm(data - traj[0], axis=-1))
    end = jnp.min(jnp.linalg.norm(data - traj[-1], axis=-1))
    assert jnp.all(jnp.isfinite(traj))
    assert float(end) < float(start)
