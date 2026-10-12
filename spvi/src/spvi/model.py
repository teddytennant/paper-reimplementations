"""Recognition net, block-triangular jump, and state-dependent prior."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from spvi.gates import (
    RHO_MIN,
    rate_link,
    retention_from_raw,
    variance_from_log,
)


@dataclass(frozen=True)
class Config:
    obs_dim: int = 2
    state_dim: int = 2
    mark_dim: int = 2
    hidden: int = 8
    embed_dim: int = 8
    conv_width: int = 3
    block_sizes: tuple = (1, 1)
    obs_var: float = 0.01
    tau_st: float = 1.0
    rho: float = 1e-3
    rho_min: float = RHO_MIN
    l_max: int = 16
    n_quad: int = 2
    w_kl: float = 1.0
    beta_sparse: float = 0.1
    lambda_bar_theta: float = 10.0
    # Posterior rate bias is set so the initial link sits near this value.
    init_rate: float = 2.0


def _mlp(key, in_dim, hidden, out_dim, scale=0.1):
    k1, k2 = jax.random.split(key)
    return {
        "w1": jax.random.normal(k1, (in_dim, hidden)) * scale,
        "b1": jnp.zeros((hidden,)),
        "w2": jax.random.normal(k2, (hidden, out_dim)) * scale,
        "b2": jnp.zeros((out_dim,)),
    }


def mlp_apply(params, x):
    hidden = jnp.tanh(x @ params["w1"] + params["b1"])
    return hidden @ params["w2"] + params["b2"]


def _rate_bias(target):
    """Inverse of the bounded rate link, so rate_link(bias) = target at w = 0."""
    log_target = jnp.log(target)
    a = jnp.log(1e-3)
    b = jnp.log(10.0)
    sigmoid = (log_target - a) / (b - a)
    return jnp.log(sigmoid) - jnp.log1p(-sigmoid)


def init_params(key, config=None):
    """Parameter pytree. `jump` is the block-triangular jump net; `rate_head` is the posterior rate."""
    if config is None:
        config = Config()
    if sum(config.block_sizes) != config.state_dim:
        raise ValueError("block_sizes must sum to state_dim")
    keys = jax.random.split(key, 8)
    embed = config.embed_dim
    in_ch = config.obs_dim + 1
    params = {
        "gamma_raw": jnp.zeros((config.state_dim,)),
        "encoder": {
            "conv_w": jax.random.normal(keys[0], (config.conv_width, in_ch, embed)) * 0.1,
            "conv_b": jnp.zeros((embed,)),
            "rec_w": jax.random.normal(keys[1], (embed, embed)) * 0.1,
            "in_w": jax.random.normal(keys[1], (embed, embed)) * 0.1,
            "rec_b": jnp.zeros((embed,)),
        },
        "rate_head": {
            "w": jnp.zeros((embed, 1)),
            "b": jnp.asarray([_rate_bias(config.init_rate)]),
        },
        "mark_head": {
            "mu_w": jax.random.normal(keys[2], (embed, config.mark_dim)) * 0.1,
            "mu_b": jnp.zeros((config.mark_dim,)),
            "log_var_w": jax.random.normal(keys[3], (embed, config.mark_dim)) * 0.1,
            "log_var_b": jnp.zeros((config.mark_dim,)),
        },
        "init_head": {
            "mu_w": jax.random.normal(keys[4], (embed, config.state_dim)) * 0.1,
            "mu_b": jnp.zeros((config.state_dim,)),
            "log_var_w": jax.random.normal(keys[4], (embed, config.state_dim)) * 0.1,
            "log_var_b": jnp.zeros((config.state_dim,)),
        },
        "jump": {"blocks": []},
        "prior_rate": _mlp(keys[5], config.state_dim, config.hidden, 1),
        "prior_mark": _mlp(keys[6], config.state_dim, config.hidden, 2 * config.mark_dim),
    }
    k_blocks = jax.random.split(keys[7], len(config.block_sizes))
    lower = 0
    blocks = []
    for j, width in enumerate(config.block_sizes):
        in_dim = config.mark_dim + (0 if j == 0 else 2 * lower)
        blocks.append(_mlp(k_blocks[j], in_dim, config.hidden, 2 * width))
        lower += width
    params["jump"]["blocks"] = blocks
    return params


def causal_conv(x, kernel, bias):
    """Causal conv over time. x: (B, K, C). kernel: (W, C, E)."""
    width = kernel.shape[0]
    padded = jnp.pad(x, ((0, 0), (width - 1, 0), (0, 0)))
    y = jax.lax.conv_general_dilated(
        padded,
        kernel,
        window_strides=(1,),
        padding="VALID",
        dimension_numbers=("NWC", "WIO", "NWC"),
    )
    return y + bias


def encode(params, times, y):
    """Causal conv followed by a causal MLP over (time, y). Not a transformer.

    times: (B, K), y: (B, K, obs). Returns embeddings (B, K, E).
    The final embedding reads the whole context through the recurrence.
    """
    x = jnp.concatenate([y, times[..., None]], axis=-1)
    local = jnp.tanh(causal_conv(x, params["encoder"]["conv_w"], params["encoder"]["conv_b"]))
    rec_w = params["encoder"]["rec_w"]
    in_w = params["encoder"]["in_w"]
    rec_b = params["encoder"]["rec_b"]

    def step(carry, h_t):
        out = jnp.tanh(carry @ rec_w + h_t @ in_w + rec_b)
        return out, out

    h0 = jnp.zeros((y.shape[0], rec_w.shape[0]), dtype=y.dtype)
    _, seq = jax.lax.scan(step, h0, jnp.swapaxes(local, 0, 1))
    return jnp.swapaxes(seq, 0, 1)


def posterior_event_params(params, embeddings):
    """Observation-conditioned event law. Does not read the latent state."""
    ell = jnp.squeeze(embeddings @ params["rate_head"]["w"] + params["rate_head"]["b"], axis=-1)
    lam = rate_link(ell)
    mu = embeddings @ params["mark_head"]["mu_w"] + params["mark_head"]["mu_b"]
    log_var = embeddings @ params["mark_head"]["log_var_w"] + params["mark_head"]["log_var_b"]
    var = variance_from_log(log_var)
    return lam, mu, var


def initial_posterior(params, embeddings):
    """q0 from the final context embedding."""
    final = embeddings[:, -1, :]
    mu = final @ params["init_head"]["mu_w"] + params["init_head"]["mu_b"]
    var = variance_from_log(final @ params["init_head"]["log_var_w"] + params["init_head"]["log_var_b"])
    return mu, var


def jump_coefficients(params, block_index, marks, lower_pre, lower_post, rho_min=RHO_MIN):
    """(R, B) for one block. Inputs are the mark and lower pre/post states only.

    marks: (B, L, mark_dim)
    lower_pre, lower_post: (B, L, lower_dim), lower_dim = 0 for the first block.
    R is in (rho_min, 1). B is an unconstrained shift.
    """
    block = params["jump"]["blocks"][block_index]
    if lower_pre.shape[-1] == 0:
        inputs = marks
    else:
        inputs = jnp.concatenate([marks, lower_pre, lower_post], axis=-1)
    raw = mlp_apply(block, inputs)
    width = raw.shape[-1] // 2
    retention = retention_from_raw(raw[..., :width], rho_min=rho_min)
    shift = raw[..., width:]
    return retention, shift


def make_coeff_fn(params, rho_min=RHO_MIN):
    def coeff_fn(block_index, marks, lower_pre, lower_post):
        return jump_coefficients(
            params, block_index, marks, lower_pre, lower_post, rho_min=rho_min
        )

    return coeff_fn


def prior_event_params(params, z):
    """State-dependent prior rate and diagonal-Gaussian mark."""
    ell = jnp.squeeze(mlp_apply(params["prior_rate"], z), axis=-1)
    lam = rate_link(ell)
    raw = mlp_apply(params["prior_mark"], z)
    mark_dim = raw.shape[-1] // 2
    mu = raw[..., :mark_dim]
    var = variance_from_log(raw[..., mark_dim:])
    return lam, mu, var


def emission_mean(z, obs_dim):
    """Identity emission on the first obs_dim coordinates."""
    return z[..., :obs_dim]


def observation_intervals(times):
    """Intervals (t_{k-1}, t_k] with t_0 = 0.

    times: (B, K) observation times, each positive and increasing.
    Returns t_left, t_right, delta, T.
    """
    zeros = jnp.zeros_like(times[:, :1])
    t_left = jnp.concatenate([zeros, times[:, :-1]], axis=1)
    t_right = times
    delta = t_right - t_left
    return t_left, t_right, delta, times[:, -1]
