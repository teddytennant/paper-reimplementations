"""Recurrent-depth encoder and softmax recovery head, Appendix B.2.

The encoder runs causal attention over four positions, the three context tokens
and one query slot, and reads the representation off the query slot. A block of
`n_sub_layers` transformer sub-layers is applied `depth` times with the same
parameters, which is the class R^L(G, R^0) of Section 3.2: the L applications
share one parameter set, so depth changes computation and not parameter count.

The head is the softmax kernel K(Z, E_v) = exp(E_v^T Z + b_v) of Assumption 1.2,
with the input embedding tied to E. Two norm constraints from Appendix B.2 are
kept because the theory uses them: the representation is read by exact
normalization h = 4 r / ||r||, so ||Theta_j(X)|| = 4 identically and B_Theta of
Assumption 5.2 is exactly 4, and the augmented head rows (E_v, b_v) are
projected onto a ball of radius 4 after every update, which is the B_E of
Corollary 1.
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

REPRESENTATION_NORM = 4.0
HEAD_RADIUS = 4.0


@dataclasses.dataclass(frozen=True)
class EncoderConfig:
    vocab_size: int
    width: int = 16
    n_heads: int = 2
    ffn_width: int = 64
    n_sub_layers: int = 1
    depth: int = 1
    context_length: int = 3

    @property
    def n_positions(self) -> int:
        return self.context_length + 1


def init_params(config: EncoderConfig, seed: int) -> dict:
    """N(0, 1/width) for every weight matrix and the token embeddings.

    Deliberately depth-independent, as in Appendix B.2, so one seed gives
    byte-identical initial parameters at every depth and the comparison across
    L is paired.
    """
    rng = np.random.default_rng(seed)
    s, w = config.width, config.ffn_width
    scale = 1.0 / np.sqrt(s)

    def normal(*shape):
        return jnp.asarray(rng.standard_normal(shape) * scale)

    block = [
        {
            "wq": normal(s, s),
            "wk": normal(s, s),
            "wv": normal(s, s),
            "wo": normal(s, s),
            "ln1_scale": jnp.ones(s),
            "ln1_shift": jnp.zeros(s),
            "ln2_scale": jnp.ones(s),
            "ln2_shift": jnp.zeros(s),
            "w_in": normal(s, w),
            "b_in": jnp.zeros(w),
            "w_out": normal(w, s),
            "b_out": jnp.zeros(s),
        }
        for _ in range(config.n_sub_layers)
    ]
    return {
        "embed": normal(config.vocab_size, s),
        "head_bias": jnp.zeros(config.vocab_size),
        "pos": normal(config.n_positions, s),
        "block": block,
    }


def parameter_count(params: dict) -> int:
    return int(sum(x.size for x in jax.tree.leaves(params)))


def _layer_norm(x, scale, shift, eps=1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * scale + shift


def _attention(p, x, n_heads):
    n, t, s = x.shape
    head_dim = s // n_heads
    q = (x @ p["wq"]).reshape(n, t, n_heads, head_dim)
    k = (x @ p["wk"]).reshape(n, t, n_heads, head_dim)
    v = (x @ p["wv"]).reshape(n, t, n_heads, head_dim)
    logits = jnp.einsum("nqhd,nkhd->nhqk", q, k) / jnp.sqrt(head_dim)
    mask = jnp.tril(jnp.ones((t, t), dtype=bool))
    logits = jnp.where(mask, logits, -jnp.inf)
    attn = jax.nn.softmax(logits, axis=-1)
    out = jnp.einsum("nhqk,nkhd->nqhd", attn, v).reshape(n, t, s)
    return out @ p["wo"]


def _sub_layer(p, x, n_heads):
    x = x + _attention(p, _layer_norm(x, p["ln1_scale"], p["ln1_shift"]), n_heads)
    y = _layer_norm(x, p["ln2_scale"], p["ln2_shift"])
    return x + jax.nn.gelu(y @ p["w_in"] + p["b_in"]) @ p["w_out"] + p["b_out"]


def initial_representation(params, contexts, config: EncoderConfig):
    """Psi^0, the R^0 of Section 3.2: tied token embeddings plus positions.

    The query slot carries only its positional embedding, so Psi^0 at the target
    position does not depend on X. That is why Figure 3 omits l = 0.
    """
    ctx = params["embed"][contexts] + params["pos"][None, : config.context_length]
    query = jnp.broadcast_to(
        params["pos"][None, config.context_length][None], (ctx.shape[0], 1, config.width)
    )
    return jnp.concatenate([ctx, query], axis=1)


@functools.partial(jax.jit, static_argnums=2)
def encode_states(params, contexts, config: EncoderConfig):
    """Every intermediate state Psi^0, ..., Psi^L. Shape (L+1, n, positions, s).

    The L applications are a `lax.scan` rather than a Python loop. That is the
    same computation, since the block parameters are shared across depth, and it
    keeps the compiled graph the size of one block instead of L of them.
    """
    initial = initial_representation(params, contexts, config)

    def step(x, _):
        for p in params["block"]:
            x = _sub_layer(p, x, config.n_heads)
        return x, x

    _, states = jax.lax.scan(step, initial, None, length=config.depth)
    return jnp.concatenate([initial[None], states], axis=0)


def read_representation(x, config: EncoderConfig):
    """h = 4 r / ||r|| read off the query slot, so ||Theta_j(X)|| = 4 exactly."""
    r = x[..., config.context_length, :]
    return REPRESENTATION_NORM * r / jnp.linalg.norm(r, axis=-1, keepdims=True)


def representation(params, contexts, config: EncoderConfig):
    """Theta_j(X) for a batch of (n, context_length) token contexts."""
    return read_representation(encode_states(params, contexts, config)[-1], config)


def layer_representations(params, contexts, config: EncoderConfig):
    """Psi^l_j(X) for l = 0, ..., L, each read and normalized the same way."""
    return read_representation(encode_states(params, contexts, config), config)


def logits_from_representation(params, z):
    """log K(Z, E_v) + b_v, the augmented softmax head."""
    return z @ params["embed"].T + params["head_bias"]


def log_predictions(params, contexts, config: EncoderConfig):
    """log p_hat_E(v | Theta_j(X))."""
    z = representation(params, contexts, config)
    return jax.nn.log_softmax(logits_from_representation(params, z), axis=-1)


def cross_entropy(params, contexts, targets, config: EncoderConfig):
    """The sampled training objective, equal to the KL of Section 2.2 up to a
    constant that does not depend on E or Theta."""
    logp = log_predictions(params, contexts, config)
    return -jnp.mean(jnp.take_along_axis(logp, targets[:, None], axis=1))


def project_head(params):
    """Project the augmented rows (E_v, b_v) onto the ball of radius 4."""
    norms = jnp.sqrt(
        jnp.sum(params["embed"] ** 2, axis=1) + params["head_bias"] ** 2
    )
    factor = jnp.minimum(1.0, HEAD_RADIUS / jnp.maximum(norms, 1e-12))
    return {
        **params,
        "embed": params["embed"] * factor[:, None],
        "head_bias": params["head_bias"] * factor,
    }
