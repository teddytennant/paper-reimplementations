"""Discrete convex interpolant and a tiny categorical flow-matching trainer."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from flow_duality.coefficients import k_gaussian
from flow_duality.interpolant import one_hot_tokens


def sample_xt(key, x0, x1, kt):
    """Draw X_t from the discrete convex interpolant (eq 4).

    Independently per position: X_t = X_1 with probability k_t, else X_0.
    When X_0 = X_1 the two branches agree.
    """
    kt = jnp.asarray(kt, dtype=jnp.float32)
    extra = x0.ndim - kt.ndim
    if extra < 0:
        raise ValueError("kt has more axes than x0")
    p = jnp.reshape(kt, kt.shape + (1,) * extra)
    flip = jax.random.bernoulli(key, p, shape=x0.shape)
    return jnp.where(flip, x1, x0)


def dfm_cross_entropy(logits, x1):
    """Mean token cross-entropy, the inner sum of eq 7."""
    logp = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(logp, x1[..., None], axis=-1)[..., 0]
    return jnp.mean(nll)


def _mlp_init(key, in_dim, hidden, out_dim):
    k1, k2, k3 = jax.random.split(key, 3)
    w1 = jax.random.normal(k1, (in_dim, hidden)) * jnp.sqrt(2.0 / in_dim)
    w2 = jax.random.normal(k2, (hidden, hidden)) * jnp.sqrt(2.0 / hidden)
    w3 = jax.random.normal(k3, (hidden, out_dim)) * jnp.sqrt(2.0 / hidden)
    return {
        "W1": w1,
        "b1": jnp.zeros((hidden,)),
        "W2": w2,
        "b2": jnp.zeros((hidden,)),
        "W3": w3,
        "b3": jnp.zeros((out_dim,)),
    }


def _mlp_apply(params, x):
    h = jax.nn.relu(x @ params["W1"] + params["b1"])
    h = jax.nn.relu(h @ params["W2"] + params["b2"])
    return h @ params["W3"] + params["b3"]


def predict_logits(params, xt, t, vocab_size):
    """Tiny MLP: one-hot(xt) concatenated with t, logits per position."""
    seq_len = xt.shape[-1]
    xoh = one_hot_tokens(xt, vocab_size, dtype=jnp.float32)
    flat = xoh.reshape(xt.shape[0], seq_len * vocab_size)
    tcol = jnp.reshape(jnp.asarray(t, dtype=jnp.float32), (xt.shape[0], 1))
    feats = jnp.concatenate([flat, tcol], axis=-1)
    raw = _mlp_apply(params, feats)
    return raw.reshape(xt.shape[0], seq_len, vocab_size)


def _adam_init(params):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return zeros, zeros


def _adam_update(params, grads, m, v, step, lr, b1=0.9, b2=0.999, eps=1e-8):
    m = jax.tree_util.tree_map(lambda mm, g: b1 * mm + (1.0 - b1) * g, m, grads)
    v = jax.tree_util.tree_map(lambda vv, g: b2 * vv + (1.0 - b2) * (g * g), v, grads)
    mhat = jax.tree_util.tree_map(lambda mm: mm / (1.0 - b1**step), m)
    vhat = jax.tree_util.tree_map(lambda vv: vv / (1.0 - b2**step), v)
    params = jax.tree_util.tree_map(
        lambda p, mm, vv: p - lr * mm / (jnp.sqrt(vv) + eps), params, mhat, vhat
    )
    return params, m, v


def _sample_batch(key, batch_size, seq_len, vocab_size):
    k0, k1, kt, kx = jax.random.split(key, 4)
    x0 = jax.random.randint(k0, (batch_size, seq_len), 0, vocab_size)
    x1 = jax.random.randint(k1, (batch_size, seq_len), 0, vocab_size)
    t = jax.random.uniform(kt, (batch_size,))
    kt_val = k_gaussian(t, vocab_size)
    xt = sample_xt(kx, x0, x1, kt_val)
    return xt, t, x1


def train_tiny(
    key,
    *,
    n_steps=200,
    batch_size=256,
    seq_len=4,
    vocab_size=2,
    hidden=32,
    lr=0.03,
    eval_size=1024,
):
    """SGD/Adam on eq 7 using Gaussian-induced k_t. Synthetic uniform tokens.

    Returns (loss_step0, loss_final, params).
    """
    in_dim = seq_len * vocab_size + 1
    out_dim = seq_len * vocab_size
    key, k_init, k_eval, k_train = jax.random.split(key, 4)
    params = _mlp_init(k_init, in_dim, hidden, out_dim)
    m, v = _adam_init(params)

    def loss_fn(p, xt, t, x1):
        logits = predict_logits(p, xt, t, vocab_size)
        return dfm_cross_entropy(logits, x1)

    eval_batch = _sample_batch(k_eval, eval_size, seq_len, vocab_size)
    loss0 = loss_fn(params, *eval_batch)

    def step(state, skey):
        p, mm, vv, step_i = state
        xt, t, x1 = _sample_batch(skey, batch_size, seq_len, vocab_size)
        loss, grads = jax.value_and_grad(loss_fn)(p, xt, t, x1)
        p, mm, vv = _adam_update(p, grads, mm, vv, step_i + 1, lr)
        return (p, mm, vv, step_i + 1), loss

    keys = jax.random.split(k_train, n_steps)

    @jax.jit
    def run(p, mm, vv, step_keys):
        (p, mm, vv, _), _ = jax.lax.scan(step, (p, mm, vv, jnp.array(0)), step_keys)
        return p

    params = run(params, m, v, keys)
    loss1 = loss_fn(params, *eval_batch)
    return loss0, loss1, params
