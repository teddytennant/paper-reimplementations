"""LAM block: LHA then BDM, plus a tiny LPS-downsampled stack."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from pointlam.bdm import BDMParams, apply_bdm, init_bdm
from pointlam.lha import LHAParams, apply_lha, init_lha
from pointlam.lps import MLPParams, apply_lps, init_mlp
from pointlam.nn import Linear, apply_linear, init_linear


class LAMBlockParams(NamedTuple):
    lha: LHAParams
    bdm: BDMParams


class PointLAMParams(NamedTuple):
    lift: Linear
    mlp: MLPParams
    blocks: tuple[LAMBlockParams, ...]
    head: Linear


def init_lam_block(key: jax.Array, dim: int, state: int = 8) -> LAMBlockParams:
    k1, k2 = jax.random.split(key)
    return LAMBlockParams(lha=init_lha(k1, dim), bdm=init_bdm(k2, dim, state))


def apply_lam_block(
    params: LAMBlockParams,
    points: jax.Array,
    features: jax.Array,
    cell_size: float,
) -> jax.Array:
    h = apply_lha(params.lha, points, features, cell_size)
    return apply_bdm(params.bdm, points, h)


def init_pointlam(
    key: jax.Array,
    dim: int = 16,
    n_blocks: int = 2,
    state: int = 8,
    in_dim: int = 3,
) -> PointLAMParams:
    keys = jax.random.split(key, n_blocks + 3)
    lift_w = jnp.zeros((in_dim, dim)).at[:, :in_dim].set(jnp.eye(in_dim))
    lift_w = lift_w + 0.01 * jax.random.normal(keys[0], (in_dim, dim))
    lift = Linear(lift_w, jnp.zeros((dim,)))
    blocks = tuple(
        init_lam_block(keys[i + 1], dim, state) for i in range(n_blocks)
    )
    return PointLAMParams(
        lift=lift,
        mlp=init_mlp(keys[n_blocks + 1], dim, dim, dim),
        blocks=blocks,
        head=init_linear(keys[n_blocks + 2], dim, 1, bias=True),
    )


def apply_pointlam(
    params: PointLAMParams,
    points: jax.Array,
    key: jax.Array,
    m: int,
    cell_size: float,
    features: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """LPS downsample, then stacked LAM blocks. Returns pts, feats, idx, logits."""
    if features is None:
        features = points
    h = apply_linear(params.lift, features)
    pts, feats, idx, _dev = apply_lps(
        points, h, key, m, cell_size, mlp_params=params.mlp
    )
    for block in params.blocks:
        feats = apply_lam_block(block, pts, feats, cell_size)
    logits = apply_linear(params.head, feats)
    return pts, feats, idx, logits
