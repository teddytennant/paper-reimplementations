"""Multiclass log-softmax loss, N3, temporal smoothness, Adagrad step."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from bsca.model import n_relations, object_fusion, scores_all_objects


def n3_penalty(*qs):
    """Sum of |z|^3 over every complex component of each argument."""
    if len(qs) == 1 and isinstance(qs[0], (tuple, list)):
        qs = tuple(qs[0])
    total = jnp.array(0.0, dtype=jnp.float32)
    for q in qs:
        total = total + jnp.sum(jnp.abs(jnp.asarray(q)) ** 3)
    return total


def temporal_penalty(time_params, p=3):
    """L_τ = sum_i ||e_{τ_{i+1}} - e_{τ_i}||_p^p.

    If ``time_params`` is a param dict, sums over time_mul, time_add, time_joint.
    If it is a (T, 4, k) tensor, applies the adjacent-difference penalty once.
    """
    if isinstance(time_params, dict):
        total = jnp.array(0.0, dtype=jnp.float32)
        for key in ("time_mul", "time_add", "time_joint"):
            if key in time_params:
                total = total + temporal_penalty(time_params[key], p=p)
        return total
    x = jnp.asarray(time_params)
    if x.shape[0] <= 1:
        return jnp.array(0.0, dtype=jnp.float32)
    diff = x[1:] - x[:-1]
    return jnp.sum(jnp.abs(diff) ** p)


def _nll_objects(params, s, r, o, tau):
    scores, e_str, e_rtt, _ = scores_all_objects(params, s, r, tau)
    labels = jnp.asarray(o)
    nll = optax.softmax_cross_entropy_with_integer_labels(scores, labels)
    return jnp.mean(nll), e_str, e_rtt


def multiclass_loss(params, s, r, o, tau, lambda_n3=1e-3, lambda_tau=1e-3):
    """L = L_ω + λ_τ L_τ with N3 inside L_ω, plus reciprocal subject prediction."""
    s = jnp.atleast_1d(jnp.asarray(s))
    r = jnp.atleast_1d(jnp.asarray(r))
    o = jnp.atleast_1d(jnp.asarray(o))
    tau = jnp.atleast_1d(jnp.asarray(tau))

    nll_o, e_str, e_rtt = _nll_objects(params, s, r, o, tau)
    n_rel = n_relations(params)
    nll_s, _, _ = _nll_objects(params, o, r + n_rel, s, tau)

    e_otr, _ = object_fusion(params, o, r, tau)
    n3 = (
        jnp.mean(jnp.sum(jnp.abs(e_str) ** 3, axis=(-2, -1)))
        + jnp.mean(jnp.sum(jnp.abs(e_rtt) ** 3, axis=(-2, -1)))
        + jnp.mean(jnp.sum(jnp.abs(e_otr) ** 3, axis=(-2, -1)))
    )
    l_omega = nll_o + nll_s + lambda_n3 * n3
    l_tau = temporal_penalty(params, p=3)
    return l_omega + lambda_tau * l_tau


def _to_real(params):
    """Stack real/imag so autodiff is ordinary real calculus (JAX Wirtinger is holomorphic)."""
    return jax.tree.map(
        lambda x: jnp.stack([x.real, x.imag], axis=-1) if jnp.iscomplexobj(x) else x,
        params,
    )


def _to_complex(real_params, template):
    def rec(real, tmpl):
        if jnp.iscomplexobj(tmpl):
            return jax.lax.complex(real[..., 0], real[..., 1]).astype(tmpl.dtype)
        return real
    return jax.tree.map(rec, real_params, template)


def init_opt_state(params):
    real_params = _to_real(params)
    # Ones, not zeros: first Adagrad step is then ~lr * g instead of lr * sign(g).
    return jax.tree.map(lambda x: jnp.ones(jnp.shape(x), dtype=jnp.float32), real_params)


def _adagrad_apply(params, grads, state, lr, eps=1e-8):
    p_leaves, treedef = jax.tree_util.tree_flatten(params)
    g_leaves = jax.tree_util.tree_flatten(grads)[0]
    s_leaves = jax.tree_util.tree_flatten(state)[0]
    new_p, new_s = [], []
    lr = jnp.asarray(lr)
    eps = jnp.asarray(eps)
    for p, g, acc in zip(p_leaves, g_leaves, s_leaves):
        acc_n = acc + jnp.square(g).astype(acc.dtype)
        scale = lr / (jnp.sqrt(acc_n) + eps)
        p_n = p - scale.astype(p.dtype) * g
        new_p.append(p_n)
        new_s.append(acc_n)
    return treedef.unflatten(new_p), treedef.unflatten(new_s)


def train_step(params, batch, opt_state=None, lr=0.1, lambda_n3=1e-3, lambda_tau=1e-3):
    """One Adagrad step (paper uses Adagrad, lr=0.1) on a batch (s, r, o, tau)."""
    s, r, o, tau = batch
    real_params = _to_real(params)
    if opt_state is None:
        opt_state = init_opt_state(params)

    def loss_fn(rp):
        return multiclass_loss(_to_complex(rp, params), s, r, o, tau, lambda_n3=lambda_n3, lambda_tau=lambda_tau)

    loss, grads = jax.value_and_grad(loss_fn)(real_params)
    real_params, opt_state = _adagrad_apply(real_params, grads, opt_state, lr)
    params = _to_complex(real_params, params)
    return params, opt_state, loss
