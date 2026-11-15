"""The neural field of Fig. 1: a positional encoding followed by an MLP.

The paper's architecture is a 2D coordinate in [-1, 1]^2, encoded either as a raw
positional encoding (RPE) or a Fourier positional encoding (FPE) with k frequency
bands, then a 5 x 256 MLP with a scalar or RGB head. Everything here is the same
shape at a much smaller width, which is what makes the NTK solves cheap enough to
run on a CPU.

The network is written functionally: `mlp_apply(params, Z)` maps already-encoded
features to outputs, and the encoding is a separate function. NTK-KIP needs both,
because the paper optimizes support points either in raw coordinate space or
directly in Fourier feature space (Sec. IV-A).
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

Activation = str


class MLPConfig(NamedTuple):
    """Shape of the neural field f(x; theta) of Fig. 1.

    in_dim is the encoded input width, so it must match whatever `encode`
    produces: d for raw PE and d + 2 * d * k for Fourier PE (Appendix B).
    """

    in_dim: int
    width: int = 32
    depth: int = 2  # number of hidden layers
    out_dim: int = 1
    activation: Activation = "tanh"


def raw_pe(coords: jnp.ndarray) -> jnp.ndarray:
    """Raw positional encoding (RPE): the coordinate itself, Sec. II-A."""
    return coords


def fourier_pe(coords: jnp.ndarray, k_bands: int) -> jnp.ndarray:
    """Fourier positional encoding (FPE), Appendix B.

    Each d-dimensional coordinate becomes d + 2 * d * k features: the raw
    coordinate, then sin and cos at k octave frequencies. The paper uses
    k_bands = 20 with d = 2, so 82 features per coordinate.
    """
    if k_bands <= 0:
        return coords
    freqs = 2.0 ** jnp.arange(k_bands, dtype=coords.dtype) * jnp.pi
    scaled = coords[..., None] * freqs  # (..., d, k)
    flat = scaled.reshape(*coords.shape[:-1], -1)
    return jnp.concatenate([coords, jnp.sin(flat), jnp.cos(flat)], axis=-1)


def encode(coords: jnp.ndarray, kind: str = "fourier", k_bands: int = 4) -> jnp.ndarray:
    """Dispatch between the two encodings the paper compares (Table I)."""
    if kind == "raw":
        return raw_pe(coords)
    if kind == "fourier":
        return fourier_pe(coords, k_bands)
    raise ValueError(f"unknown encoding {kind!r}")


def encoded_dim(d: int, kind: str = "fourier", k_bands: int = 4) -> int:
    """Feature count of `encode`, i.e. d + 2 * d * k for FPE (Appendix B)."""
    if kind == "raw" or k_bands <= 0:
        return d
    return d + 2 * d * k_bands


def _act(name: Activation):
    if name == "tanh":
        return jnp.tanh
    if name == "relu":
        return jax.nn.relu
    if name == "gelu":
        return jax.nn.gelu
    if name == "sin":  # SIREN-style
        return jnp.sin
    raise ValueError(f"unknown activation {name!r}")


def init_params(cfg: MLPConfig, key: jax.Array) -> list:
    """Unit-variance weights under the NTK parameterization.

    `mlp_apply` divides by sqrt(fan_in), so drawing W ~ N(0, 1) here is the
    standard NTK scaling under which K(x, x') stays O(1) as the width grows.
    """
    dims = [cfg.in_dim] + [cfg.width] * cfg.depth + [cfg.out_dim]
    params = []
    for i in range(len(dims) - 1):
        key, wkey = jax.random.split(key)
        params.append(
            {
                "w": jax.random.normal(wkey, (dims[i], dims[i + 1]), dtype=jnp.float64),
                "b": jnp.zeros((dims[i + 1],), dtype=jnp.float64),
            }
        )
    return params


def mlp_apply(params: list, Z: jnp.ndarray, activation: Activation = "tanh") -> jnp.ndarray:
    """f(x; theta) of Eq. 1, taking encoded features Z of shape (n, in_dim).

    Returns (n, out_dim). The 1/sqrt(fan_in) factor is the NTK parameterization.
    """
    act = _act(activation)
    h = Z
    for layer in params[:-1]:
        h = act(h @ layer["w"] / jnp.sqrt(h.shape[-1]) + layer["b"])
    last = params[-1]
    return h @ last["w"] / jnp.sqrt(h.shape[-1]) + last["b"]


def make_field(cfg: MLPConfig):
    """Bind the activation, giving f(params, Z) -> (n, out_dim)."""

    def f(params, Z):
        return mlp_apply(params, Z, cfg.activation)

    return f
