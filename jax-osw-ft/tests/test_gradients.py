"""The weighted-SFT gradient: finite differences, and what the weight does to it.

Algorithm 1 line 10 is L(theta) = -(1/|M|) sum w log pi(o*|q). Two things have
to be true for the paper's story to hold in code: the autodiff gradient is the
gradient of that loss, and the weight rescales it without rotating it away from
g_0 = grad log pi(o*|q).
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from osw_ft.loss import g0, sft_loss, weighted_sft_loss
from osw_ft.model import ModelConfig, init_params
from osw_ft.rollout import success_probability
from osw_ft.task import PROMPT_LEN, SEQ_LEN, VOCAB_SIZE, build_dataset


def _setup(seed=0, n=3):
    config = ModelConfig(vocab_size=VOCAB_SIZE, max_len=SEQ_LEN, d_model=16, n_heads=2, d_ff=32)
    params = init_params(jr.PRNGKey(seed), config, dtype=jnp.float64)
    data = jnp.asarray(build_dataset(4))
    batch = data[jnp.asarray([0, 5, 11])[:n]]
    return config, params, batch


def _dot(a, b):
    return sum(
        float(jnp.sum(x * y)) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b))
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_weighted_gradient_matches_finite_differences(seed):
    """Central differences of L(theta) along a random direction, in float64."""
    config, params, batch = _setup(seed)
    w = jnp.asarray([0.25, 1.0, 0.5])

    direction = jax.tree.map(
        lambda leaf, k: jr.normal(k, leaf.shape, leaf.dtype),
        params,
        _tree_keys(jr.PRNGKey(seed + 100), params),
    )
    grads = jax.grad(weighted_sft_loss)(params, config, batch, PROMPT_LEN, w)
    analytic = _dot(grads, direction)

    eps = 1e-6
    plus = weighted_sft_loss(
        jax.tree.map(lambda p, d: p + eps * d, params, direction), config, batch, PROMPT_LEN, w
    )
    minus = weighted_sft_loss(
        jax.tree.map(lambda p, d: p - eps * d, params, direction), config, batch, PROMPT_LEN, w
    )
    numeric = float(plus - minus) / (2 * eps)

    assert numeric == pytest.approx(analytic, rel=1e-6, abs=1e-8)


def _tree_keys(key, tree):
    leaves, treedef = jax.tree.flatten(tree)
    return jax.tree.unflatten(treedef, list(jr.split(key, len(leaves))))


@pytest.mark.parametrize("weight", [0.0, 0.25, 1.0])
def test_gradient_is_the_weight_times_g0(weight):
    """eq (9): the OSW-FT update is w * g_0, so direction is untouched by w.

    Single-query batch, so the mean drops out and the two sides are the same
    vector up to the scalar. w = 0 gives exactly the zero update of Remark 1's
    signal loss.
    """
    config, params, batch = _setup(seed=0, n=1)
    grads = jax.grad(weighted_sft_loss)(params, config, batch, PROMPT_LEN, jnp.asarray([weight]))
    reference = g0(params, config, batch[0], PROMPT_LEN)
    for got, want in zip(jax.tree.leaves(grads), jax.tree.leaves(reference)):
        assert jnp.allclose(got, -weight * want, rtol=1e-12, atol=1e-14)


def test_uniform_weight_recovers_sft():
    """eq (8): w = 1 for every query is standard SFT, gradient and value."""
    config, params, batch = _setup(seed=1)
    ones = jnp.ones((batch.shape[0],))
    assert float(weighted_sft_loss(params, config, batch, PROMPT_LEN, ones)) == pytest.approx(
        float(sft_loss(params, config, batch, PROMPT_LEN)), rel=1e-14
    )
    a = jax.grad(weighted_sft_loss)(params, config, batch, PROMPT_LEN, ones)
    b = jax.grad(sft_loss)(params, config, batch, PROMPT_LEN)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        assert jnp.allclose(x, y, rtol=1e-13, atol=1e-15)


def test_weight_carries_a_stop_gradient():
    """Remark 4: no grad_theta p_theta(q) term leaks into the update.

    Feeding the weight in as a differentiable function of theta has to give the
    same gradient as feeding in its frozen value, or the loop would be
    optimizing something other than the surrogate J^(t) of eq (16).
    """
    config, params, batch = _setup(seed=2)

    def live_weight_loss(params):
        w = 1.0 - success_probability(params, config, batch, PROMPT_LEN)
        return weighted_sft_loss(params, config, batch, PROMPT_LEN, w)

    frozen = jax.lax.stop_gradient(1.0 - success_probability(params, config, batch, PROMPT_LEN))
    a = jax.grad(live_weight_loss)(params)
    b = jax.grad(weighted_sft_loss)(params, config, batch, PROMPT_LEN, frozen)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        assert jnp.allclose(x, y, rtol=1e-13, atol=1e-15)


def test_measured_second_moment_matches_eq11():
    """eq (11) against the second moment of the estimator, enumerated over rollouts.

    For each query, average ||(1 - p_hat) g_0||^2 over the K+1 binomial outcomes
    weighted by their exact probabilities, and compare with the closed form.
    """
    from osw_ft.loss import g0_sq_norms  # noqa: PLC0415
    from osw_ft.weighting import sigma_squared, weight_pmf  # noqa: PLC0415

    config, params, batch = _setup(seed=0)
    K = 3
    p_s = success_probability(params, config, batch, PROMPT_LEN)
    sq = g0_sq_norms(params, config, batch, PROMPT_LEN)

    enumerated = 0.0
    for p, n in zip(p_s.tolist(), sq.tolist()):
        values, probs = weight_pmf(p, K)
        enumerated += float(jnp.sum(probs * values**2)) * n
    enumerated /= len(sq)

    assert float(sigma_squared(p_s, sq, K)) == pytest.approx(enumerated, rel=1e-11)
