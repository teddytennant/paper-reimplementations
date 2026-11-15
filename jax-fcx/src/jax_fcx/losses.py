"""Seven-part FCx training loss (Algorithm 1 line 17, §4.2)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_fcx.classifier import forward as classifier_forward
from jax_fcx.constraints import feasibility_loss
from jax_fcx.lof import mean_lof
from jax_fcx.vae import assemble_counterfactual, decode, encode, sample_z


def reconstruction_loss(x: jax.Array, x_cf: jax.Array) -> jax.Array:
    """Lrec: L1/L2 proximity between x' and x."""
    diff = x_cf - x
    return jnp.mean(jnp.abs(diff)) + jnp.mean(diff * diff)


def sparsity_loss(x: jax.Array, x_cf: jax.Array) -> jax.Array:
    """Lspar: L1 as a smooth L0 proxy between x' and x."""
    return jnp.mean(jnp.abs(x_cf - x))


def kl_standard_normal(mu: jax.Array, var: jax.Array) -> jax.Array:
    """LKL: KL(q(z|x_M,y) || N(0,I)) for a diagonal Gaussian."""
    var = jnp.clip(var, 1e-6, 1.0)
    kl = 0.5 * jnp.sum(mu * mu + var - jnp.log(var) - 1.0, axis=-1)
    return jnp.mean(kl)


def hinge_validity(logits_cf: jax.Array, y_target: jax.Array, margin: float = 1.0) -> jax.Array:
    """Lval: HingeLoss(f(x'), y_target; m) with y_target = 1 - argmax f(X)."""
    y = y_target.astype(jnp.int32)
    target_logit = jnp.take_along_axis(logits_cf, y[:, None], axis=1)[:, 0]
    other_logit = jnp.take_along_axis(logits_cf, (1 - y)[:, None], axis=1)[:, 0]
    return jnp.mean(jnp.maximum(0.0, margin - (target_logit - other_logit)))


def soft_causal_loss(
    x_cf: jax.Array,
    adjacency: jax.Array,
    theta: float = 0.1,
    lambda_nc: float = 1.0,
    lambda_c: float = 1.0,
) -> jax.Array:
    """Lcausal (Eq 10) on a batch of generated CFs.

    Adjacency A is taken as input (no PC-algorithm). Pairs are treated as
    undirected: an edge exists if A[p,q] or A[q,p] is nonzero. Diagonal is
    excluded. Unconnected pairs pay Cov^2; connected pairs pay
    max(0, theta - Cov)^2.
    """
    b = x_cf.shape[0]
    xc = x_cf - jnp.mean(x_cf, axis=0, keepdims=True)
    cov = (xc.T @ xc) / jnp.maximum(b - 1, 1)
    d = cov.shape[0]
    undirected = jnp.maximum(adjacency, adjacency.T)
    upper = jnp.triu(jnp.ones((d, d), dtype=x_cf.dtype), k=1)
    connected = undirected * upper
    unconnected = (1.0 - undirected) * upper
    l_nc = lambda_nc * jnp.sum(unconnected * (cov * cov))
    gap = jnp.maximum(0.0, theta - cov)
    l_c = lambda_c * jnp.sum(connected * (gap * gap))
    return l_c + l_nc


def combine_losses(
    l_rec: jax.Array,
    l_kl: jax.Array,
    l_val: jax.Array,
    l_feas: jax.Array,
    l_causal: jax.Array,
    l_spar: jax.Array,
    l_lof: jax.Array,
    lambda_rec: float = 1.0,
    lambda_val: float = 1.0,
    lambda_feas: float = 1.0,
    lambda_causal: float = 1.0,
    lambda_spar: float = 1.0,
    lambda_lof: float = 1.0,
) -> jax.Array:
    """L = λrec Lrec + LKL + λval Lval + λfeas Lfeas + λcausal Lcausal + λspar Lspar + λlof Llof."""
    return (
        lambda_rec * l_rec
        + l_kl
        + lambda_val * l_val
        + lambda_feas * l_feas
        + lambda_causal * l_causal
        + lambda_spar * l_spar
        + lambda_lof * l_lof
    )


def fcx_objective(
    vae_params: dict,
    clf_params: dict,
    x: jax.Array,
    y_target: jax.Array,
    rng: jax.Array,
    mutable_idx: jax.Array,
    adjacency: jax.Array,
    binary: bool = False,
    margin: float = 1.0,
    k_lof: int = 5,
    age_idx: int = 0,
    edu_idx: int = 1,
    alpha: float = 0.0,
    beta: float = 1.0,
    theta: float = 0.1,
    lambda_rec: float = 1.0,
    lambda_val: float = 1.0,
    lambda_feas: float = 1.0,
    lambda_causal: float = 1.0,
    lambda_spar: float = 1.0,
    lambda_lof: float = 1.0,
    lambda_nc: float = 1.0,
    lambda_c: float = 1.0,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Full Algorithm 1 minibatch objective, all seven terms."""
    x_m = x[:, mutable_idx]
    mu, var = encode(vae_params, x_m, y_target)
    z = sample_z(rng, mu, var)
    x_m_cf = decode(vae_params, z, y_target)
    x_cf = assemble_counterfactual(x, x_m_cf, mutable_idx)
    logits_cf = classifier_forward(clf_params, x_cf)

    l_rec = reconstruction_loss(x, x_cf)
    l_kl = kl_standard_normal(mu, var)
    l_val = hinge_validity(logits_cf, y_target, margin=margin)
    l_feas = feasibility_loss(
        x, x_cf, binary=binary, age_idx=age_idx, edu_idx=edu_idx, alpha=alpha, beta=beta
    )
    l_causal = soft_causal_loss(x_cf, adjacency, theta=theta, lambda_nc=lambda_nc, lambda_c=lambda_c)
    l_spar = sparsity_loss(x, x_cf)
    l_lof = mean_lof(z, k=k_lof)
    total = combine_losses(
        l_rec,
        l_kl,
        l_val,
        l_feas,
        l_causal,
        l_spar,
        l_lof,
        lambda_rec=lambda_rec,
        lambda_val=lambda_val,
        lambda_feas=lambda_feas,
        lambda_causal=lambda_causal,
        lambda_spar=lambda_spar,
        lambda_lof=lambda_lof,
    )
    parts = {
        "rec": l_rec,
        "kl": l_kl,
        "val": l_val,
        "feas": l_feas,
        "causal": l_causal,
        "spar": l_spar,
        "lof": l_lof,
        "total": total,
    }
    return total, parts
