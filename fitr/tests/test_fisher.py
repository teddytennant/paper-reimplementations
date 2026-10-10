import jax
import jax.numpy as jnp

from fitr import (
    gaussian_fim,
    grad_norm,
    gradient_bound,
    mc_pullback_diag,
    pullback_fisher,
)


def _mu_and_v(x):
    mu = 2.0 * x[0] + 0.5 * x[1]
    v = jnp.asarray(1.5) + 0.0 * jnp.sum(x)
    return mu, v


def test_gaussian_fim_closed_form():
    info = gaussian_fim(0.0, 4.0)
    expected = jnp.diag(jnp.array([0.25, 1.0 / (2.0 * 16.0)]))
    assert jnp.allclose(info, expected, atol=1e-6)


def test_pullback_matches_outer_product():
    g = pullback_fisher(_mu_and_v, jnp.array([0.3, 0.7]))
    expected = (1.0 / 1.5) * jnp.array([[4.0, 1.0], [1.0, 0.25]])
    assert jnp.allclose(g, expected, atol=1e-5)
    diag = jnp.array([4.0 / 1.5, 0.25 / 1.5])
    assert jnp.allclose(jnp.diag(g), diag, atol=1e-5)


def test_mc_diagonal_matches_closed_form():
    x = jnp.array([0.3, 0.7])
    closed = jnp.diag(pullback_fisher(_mu_and_v, x))
    estimate = mc_pullback_diag(_mu_and_v, x, jax.random.PRNGKey(0), 2000)
    assert jnp.allclose(estimate, closed, atol=0.15)


def test_gradient_bound_covers_mean_acquisition():
    x = jnp.array([0.3, 0.7])

    def alpha_bar(theta, zeta):
        return theta[0] + jnp.sqrt(theta[1]) * zeta

    def alpha_mean(z):
        return 2.0 * z[0] + 0.5 * z[1]

    bound = gradient_bound(alpha_bar, _mu_and_v, x, jax.random.PRNGKey(0), 2000)
    assert float(grad_norm(alpha_mean, x)) <= float(bound) + 1e-4
    assert float(jnp.trace(pullback_fisher(_mu_and_v, x))) > 0.0
