import jax
import jax.numpy as jnp
import pytest

from cachebridge import support
from cachebridge.fused_fit import fused_fit, scratch_elements, streaming_stats
from cachebridge.ridge import fit_affine, sufficient_statistics


def _problem(key, n=97, ls=5, hs=4, ds=8, ht=4, dt=6, k=3):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    src = jax.random.normal(k1, (ls, n, hs, ds))
    tgt = jax.random.normal(k2, (n, ht, dt))
    layers = jnp.sort(jax.random.choice(k3, ls, (k,), replace=False))
    assign = support.identity_assignment(ht)
    w = jax.random.uniform(k4, (ht, n), minval=0.1, maxval=3.0)
    w = n * w / w.sum(-1, keepdims=True)
    return src, tgt, layers, assign, w


@pytest.mark.parametrize("chunk", [1, 7, 32, 97, 128])
def test_streaming_stats_match_single_shot(chunk):
    """Eqs 20-21 streamed in chunks equal the batch computation, dividing n or not."""
    src, tgt, layers, assign, w = _problem(jax.random.PRNGKey(0))
    X = support.head_local_features(src, layers, assign)
    Y = jnp.transpose(tgt, (1, 0, 2))
    A_ref, B_ref, mux_ref, muy_ref = sufficient_statistics(X, Y, w)

    A, B, mu_x, mu_y = streaming_stats(src, tgt, layers, assign, w, chunk)
    assert jnp.max(jnp.abs(mu_x - mux_ref)) < 1e-11
    assert jnp.max(jnp.abs(mu_y - muy_ref)) < 1e-11
    assert jnp.max(jnp.abs(A - A_ref)) / jnp.max(jnp.abs(A_ref)) < 1e-11
    assert jnp.max(jnp.abs(B - B_ref)) / jnp.max(jnp.abs(B_ref)) < 1e-11


@pytest.mark.parametrize("chunk", [5, 97, 100])
def test_fused_fit_reproduces_the_ridge_solution(chunk):
    """Section 3.4 claims the construction path leaves W and b unchanged."""
    src, tgt, layers, assign, w = _problem(jax.random.PRNGKey(1))
    X = support.head_local_features(src, layers, assign)
    Y = jnp.transpose(tgt, (1, 0, 2))
    W_ref, b_ref = fit_affine(X, Y, w, lam=0.01)
    W, b = fused_fit(src, tgt, layers, assign, w, lam=0.01, chunk=chunk)
    assert jnp.max(jnp.abs(W - W_ref)) < 1e-9
    assert jnp.max(jnp.abs(b - b_ref)) < 1e-9


def test_uniform_weights_path():
    src, tgt, layers, assign, _ = _problem(jax.random.PRNGKey(2))
    w = jnp.ones((tgt.shape[1], tgt.shape[0]))
    X = support.head_local_features(src, layers, assign)
    Y = jnp.transpose(tgt, (1, 0, 2))
    W_ref, b_ref = fit_affine(X, Y, None, lam=0.01)
    W, b = fused_fit(src, tgt, layers, assign, w, lam=0.01, chunk=13)
    assert jnp.max(jnp.abs(W - W_ref)) < 1e-9
    assert jnp.max(jnp.abs(b - b_ref)) < 1e-9


def test_scratch_bound_is_linear_in_chunk_and_independent_of_n():
    """Eq 22: O(Ht b (2 p_l + dt)), so doubling b doubles scratch and n does not enter."""
    small = scratch_elements(8, 64, 1024, 128)
    large = scratch_elements(8, 128, 1024, 128)
    assert large == 2 * small
    assert small == 8 * 64 * (2 * 1024 + 128)
