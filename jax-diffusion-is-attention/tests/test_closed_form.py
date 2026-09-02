"""Sec. II-C: eq (11) really is the minimizer, and it really is attention."""

import jax
import jax.numpy as jnp
import pytest

from diffusion_is_attention import (
    attention_weights,
    dot_product_attention,
    gaussian_likelihood,
    one_hot_dataset,
    optimal_denoiser,
    qp_objective,
    softmax_as_attention,
)


def toy(seed=0, n=12, d=5, scale=1.0):
    k1, k2 = jax.random.split(jax.random.key(seed))
    return jax.random.normal(k1, (n, d)) * scale, jax.random.normal(k2, (d,))


@pytest.mark.parametrize("sigma", [0.3, 1.0, 2.5])
def test_eq11_is_a_stationary_point_of_eq7(sigma):
    """The gradient condition of eq (10) holds at z*, for both forms of p_i."""
    data, y = toy()
    z = optimal_denoiser(y, data, sigma)
    for normalized in (True, False):
        g = jax.grad(qp_objective)(z, y, data, sigma, normalized)
        scale = jnp.sum(gaussian_likelihood(y, data, sigma)) if normalized else 1.0
        assert jnp.linalg.norm(g) < 1e-9 * max(float(scale), 1.0)


def test_eq8_is_strictly_convex():
    """Eq (9), p_i >= 0, makes the QP (8) strictly convex: Hessian is 2 (sum p_i) I."""
    data, y = toy()
    sigma = 0.7
    z = optimal_denoiser(y, data, sigma)
    h = jax.hessian(qp_objective)(z, y, data, sigma, False)
    p_sum = jnp.sum(jnp.exp(-jnp.sum((y - data) ** 2, axis=-1) / (2 * sigma**2)))
    assert jnp.allclose(h, 2.0 * p_sum * jnp.eye(data.shape[-1]), atol=1e-10)
    assert jnp.min(jnp.linalg.eigvalsh(h)) > 0.0


@pytest.mark.parametrize("sigma", [0.3, 1.0, 2.5])
def test_eq11_beats_random_perturbations(sigma):
    data, y = toy()
    z = optimal_denoiser(y, data, sigma)
    best = qp_objective(z, y, data, sigma)
    noise = jax.random.normal(jax.random.key(3), (2000, data.shape[-1]))
    scales = jnp.geomspace(1e-3, 3.0, 2000)[:, None]
    others = jax.vmap(lambda zz: qp_objective(zz, y, data, sigma))(z + scales * noise)
    assert jnp.all(others > best)


@pytest.mark.parametrize("sigma", [0.3, 1.0])
def test_eq11_matches_an_independent_numerical_solve(sigma):
    """Gradient descent on eq (8) from a random start lands on eq (11)."""
    data, y = toy()
    p_sum = jnp.sum(jnp.exp(-jnp.sum((y - data) ** 2, axis=-1) / (2 * sigma**2)))
    lr = 0.05 / float(p_sum)
    grad = jax.grad(lambda z: qp_objective(z, y, data, sigma, False))
    z0 = jax.random.normal(jax.random.key(11), (data.shape[-1],)) * 4.0
    z = jax.jit(
        lambda z0: jax.lax.fori_loop(0, 2000, lambda _, z: z - lr * grad(z), z0)
    )(z0)
    assert jnp.allclose(z, optimal_denoiser(y, data, sigma), atol=1e-8)


def test_eq15_softmax_is_attention():
    """Eqs (13)-(15): with one-hot signals and sigma = 1, eq (11) is the softmax."""
    y = jax.random.normal(jax.random.key(5), (32,)) * 3.0
    assert jnp.allclose(softmax_as_attention(y), jax.nn.softmax(y), atol=1e-12)
    assert jnp.allclose(
        optimal_denoiser(y, one_hot_dataset(32), 1.0), jax.nn.softmax(y), atol=1e-12
    )


def test_eq15_holds_batched():
    y = jax.random.normal(jax.random.key(6), (7, 16))
    assert jnp.allclose(softmax_as_attention(y), jax.nn.softmax(y, axis=-1), atol=1e-12)


def test_eq12_is_exact_when_the_norms_are_constant():
    """Eq (12) drops exp(-||s_i||^2 / 2 sigma^2), which is a constant on a sphere."""
    key = jax.random.key(7)
    raw = jax.random.normal(key, (40, 6))
    data = raw / jnp.linalg.norm(raw, axis=-1, keepdims=True) * 1.7
    y = jax.random.normal(jax.random.key(8), (6,))
    for sigma in (0.2, 0.9, 3.0):
        assert jnp.allclose(
            dot_product_attention(y, data, sigma),
            optimal_denoiser(y, data, sigma),
            atol=1e-11,
        )


def test_eq12_error_grows_with_the_spread_of_the_norms():
    """Off the sphere eq (12) is an approximation, and this is the regime it needs.

    Same directions every time, only the radii spread out. Sigma is fixed at 1
    so the only thing moving is the term eq (12) throws away.
    """
    key = jax.random.key(9)
    kd, ku, ky = jax.random.split(key, 3)
    raw = jax.random.normal(kd, (40, 6))
    dirs = raw / jnp.linalg.norm(raw, axis=-1, keepdims=True)
    u = jax.random.uniform(ku, (40, 1), minval=-1.0, maxval=1.0)
    y = jax.random.normal(ky, (6,))

    errors = []
    for spread in (0.0, 0.05, 0.1, 0.2, 0.4, 0.8):
        data = dirs * (1.0 + spread * u)
        errors.append(
            float(
                jnp.linalg.norm(
                    dot_product_attention(y, data, 1.0) - optimal_denoiser(y, data, 1.0)
                )
            )
        )
    assert errors[0] < 1e-12
    assert all(a < b for a, b in zip(errors[1:], errors[2:]))


def test_large_sigma_gives_the_dataset_mean():
    data, y = toy(seed=1, n=30, d=4)
    z = optimal_denoiser(y, data, 1e5)
    assert jnp.allclose(z, jnp.mean(data, axis=0), atol=1e-8)
    w = attention_weights(y, data, 1e5)
    assert jnp.allclose(w, jnp.full_like(w, 1.0 / data.shape[0]), atol=1e-8)


def test_small_sigma_gives_the_nearest_neighbour():
    data, y = toy(seed=2, n=30, d=4)
    nearest = int(jnp.argmin(jnp.sum((y - data) ** 2, axis=-1)))
    z = optimal_denoiser(y, data, 1e-3)
    assert jnp.allclose(z, data[nearest], atol=1e-12)
    w = attention_weights(y, data, 1e-3)
    assert float(w[nearest]) == pytest.approx(1.0, abs=1e-12)


def test_weights_are_a_convex_combination():
    data, y = toy(seed=4, n=20, d=3)
    for sigma in (0.1, 1.0, 10.0):
        w = attention_weights(y, data, sigma)
        assert jnp.all(w >= 0.0)
        assert float(jnp.sum(w)) == pytest.approx(1.0, abs=1e-12)
