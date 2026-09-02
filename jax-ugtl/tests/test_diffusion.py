"""Trajectory modeling: the forward process, classifier-free guidance (Eq. 25),
the reverse step (Eq. 26), context conditioning (Eq. 27), and whether a trained
model actually reproduces the data it saw.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ugtl.diffusion import (
    apply_eps_model,
    cosine_schedule,
    guided_eps,
    init_eps_model,
    q_sample,
    reverse_step,
    sample_trajectories,
    train_diffusion,
)


def test_schedule_is_consistent():
    sched = cosine_schedule(100)
    assert float(sched.abar[0]) == pytest.approx(1.0)
    assert np.all(np.diff(np.asarray(sched.abar)) < 0)
    assert np.allclose(np.asarray(jnp.cumprod(sched.alphas)), np.asarray(sched.abar), atol=1e-5)


@pytest.mark.parametrize("t", [1, 7, 40, 100])
def test_forward_marginal_matches_the_step_by_step_forward_process(t):
    """q(X^(t) | X^(0)) = N(sqrt(abar_t) X^(0), (1 - abar_t) I) has to agree with
    applying the one-step forward kernel t times."""
    sched = cosine_schedule(100)
    key = jax.random.PRNGKey(0)
    n = 40_000
    x0 = jnp.full((n, 2, 1), 0.7)

    x = x0
    for s in range(1, t + 1):
        key, sub = jax.random.split(key)
        alpha = sched.alphas[s]
        x = jnp.sqrt(alpha) * x + jnp.sqrt(1.0 - alpha) * jax.random.normal(sub, x.shape)

    key, sub = jax.random.split(key)
    direct, _ = q_sample(sub, x0, jnp.full((n,), t), sched)

    tol = 4.0 / np.sqrt(n)
    assert float(jnp.abs(x.mean() - direct.mean())) < tol
    assert float(jnp.abs(x.std() - direct.std())) < tol
    assert float(direct.mean()) == pytest.approx(float(jnp.sqrt(sched.abar[t]) * 0.7), abs=tol)
    assert float(direct.var()) == pytest.approx(float(1.0 - sched.abar[t]), abs=5 * tol)


def test_guidance_with_scale_one_is_the_plain_conditional_prediction():
    """Eq. (25) at s_cfg = 1 collapses to eps_theta(X, t, y_tar)."""
    key = jax.random.PRNGKey(0)
    params = init_eps_model(key, 6, 2, width=32, depth=2)
    X = jax.random.normal(jax.random.PRNGKey(1), (5, 6, 2))
    t = jnp.full((5,), 3)
    y = jnp.linspace(0.0, 1.0, 5)

    cond = apply_eps_model(params, X, t, y, jnp.zeros((5,), dtype=bool))
    assert np.allclose(np.asarray(guided_eps(params, X, t, y, 1.0)), np.asarray(cond), atol=1e-6)

    uncond = apply_eps_model(params, X, t, y, jnp.ones((5,), dtype=bool))
    assert np.allclose(np.asarray(guided_eps(params, X, t, y, 0.0)), np.asarray(uncond), atol=1e-6)
    assert not np.allclose(np.asarray(cond), np.asarray(uncond), atol=1e-4)

    # Any other scale is the stated extrapolation through the conditional.
    for s in (1.5, 3.0):
        expected = np.asarray(uncond) + s * (np.asarray(cond) - np.asarray(uncond))
        assert np.allclose(np.asarray(guided_eps(params, X, t, y, s)), expected, atol=1e-5)


def test_reverse_step_matches_eq26_and_is_noiseless_at_the_last_step():
    sched = cosine_schedule(50)
    key = jax.random.PRNGKey(0)
    X = jax.random.normal(key, (3, 4, 2))
    eps = jax.random.normal(jax.random.PRNGKey(1), X.shape)

    for t in (2, 25, 50):
        alpha, abar_t, abar_prev = sched.alphas[t], sched.abar[t], sched.abar[t - 1]
        mean = (X - (1 - alpha) / jnp.sqrt(1 - abar_t) * eps) / jnp.sqrt(alpha)
        sigma = jnp.sqrt((1 - abar_prev) / (1 - abar_t) * (1 - alpha))
        z = jax.random.normal(jax.random.PRNGKey(9), X.shape)
        got = reverse_step(jax.random.PRNGKey(9), X, t, eps, sched)
        assert np.allclose(np.asarray(got), np.asarray(mean + sigma * z), atol=1e-5)

    # z_1 = 0 at the final reverse step, so two keys give the same answer.
    a = reverse_step(jax.random.PRNGKey(0), X, 1, eps, sched)
    b = reverse_step(jax.random.PRNGKey(7), X, 1, eps, sched)
    assert np.allclose(np.asarray(a), np.asarray(b))


def test_context_conditioning_leaves_the_prefix_exactly_equal_to_the_context():
    """Eq. (27) is applied after every reverse step, so the prefix survives."""
    key = jax.random.PRNGKey(0)
    data = jax.random.normal(key, (256, 8, 2)) * 0.3
    model = train_diffusion(key, data, jnp.zeros((256,)), T=20, steps=200, width=64, depth=2)
    ctx = jax.random.uniform(jax.random.PRNGKey(4), (16, 3, 2), minval=-1.0, maxval=1.0)
    X = sample_trajectories(jax.random.PRNGKey(5), model, 16, 0.0, ctx, s_cfg=1.5)
    assert np.array_equal(np.asarray(X[:, :3]), np.asarray(ctx))
    assert not np.allclose(np.asarray(X[:, 3:]), 0.0)


def test_the_reverse_process_recovers_the_data_mean():
    """A trained model sampled from noise reproduces the distribution it saw."""
    key = jax.random.PRNGKey(0)
    mean = jnp.array([[0.5, -0.4], [0.1, 0.3], [-0.6, 0.2], [0.0, 0.7]])
    data = mean[None, :, :] + 0.05 * jax.random.normal(key, (1024, 4, 2))
    model = train_diffusion(key, data, jnp.zeros((1024,)), T=100, steps=4000, width=128, depth=3)
    X = sample_trajectories(jax.random.PRNGKey(6), model, 512, 0.0, s_cfg=1.0)
    assert np.abs(np.asarray(X.mean(axis=0)) - np.asarray(mean)).max() < 0.1
    assert float(X.std(axis=0).mean()) < 0.2


def test_conditioning_on_a_higher_score_moves_the_samples():
    """The model has to use y, otherwise guidance towards y_tar does nothing."""
    key = jax.random.PRNGKey(0)
    y = jax.random.uniform(jax.random.PRNGKey(2), (1024,))
    data = jnp.broadcast_to(y[:, None, None], (1024, 4, 2)) * 1.5 - 0.75
    data = data + 0.02 * jax.random.normal(key, (1024, 4, 2))
    model = train_diffusion(key, data, y, T=100, steps=4000, width=128, depth=3, y_scale=(0.0, 1.0))
    low = sample_trajectories(jax.random.PRNGKey(7), model, 256, 0.1, s_cfg=1.0)
    high = sample_trajectories(jax.random.PRNGKey(7), model, 256, 0.9, s_cfg=1.0)
    assert float(high.mean()) > float(low.mean()) + 0.5
