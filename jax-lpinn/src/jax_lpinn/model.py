"""Mean and fluctuation MLPs (Eqs 8–10).

Hidden layers use tanh. The output layer is identity. The fluctuation
network has no output bias (b_{N+1} = 0), so

    ĥ'(z; θ̃, W) = W · ψ(z; θ̃)

where ψ is the last-hidden-layer activation of size Nη.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class LPINNConfig:
    """Tiny defaults; the paper uses Nη ≈ 50 and much larger collocation sets."""

    hidden_widths: tuple[int, ...] = (8, 8)
    L: float = 86.0
    T: float = 200.0
    u0: float = 1.0
    V_star: float = 0.213
    D_star: float = 0.061
    lam: float = 1.0e-4
    lam_ic: float = 1.0
    lam_bc: float = 1.0
    lam_t: float = 1.0
    lam_x: float = 1.0
    lam_xx: float = 1.0
    lam_data: float = 1.0


def n_eta(cfg: LPINNConfig) -> int:
    return int(cfg.hidden_widths[-1])


def scale_z(x, t, L: float, T: float):
    """Affine map of (x, t) onto [-1, 1]^2. Not specified in the paper."""
    x = jnp.asarray(x)
    t = jnp.asarray(t)
    return jnp.stack([2.0 * x / L - 1.0, 2.0 * t / T - 1.0], axis=-1)


def _glorot(key, shape):
    fan_in, fan_out = int(shape[0]), int(shape[1])
    return jax.random.normal(key, shape) * jnp.sqrt(2.0 / (fan_in + fan_out))


def _init_layers(key, in_dim: int, widths: tuple[int, ...]):
    keys = jax.random.split(key, len(widths))
    layers = []
    d = in_dim
    for k, width in zip(keys, widths):
        kW, kb = jax.random.split(k)
        layers.append({"W": _glorot(kW, (d, width)), "b": jnp.zeros((width,))})
        d = width
    return tuple(layers), d


def init_mean_params(key, cfg: LPINNConfig):
    """Mean network ĥ̄(z; γ): tanh hidden layers, identity output, with bias."""
    k_h, k_out = jax.random.split(key)
    layers, width = _init_layers(k_h, 2, cfg.hidden_widths)
    kW, kb = jax.random.split(k_out)
    W_out = _glorot(kW, (width, 1)).reshape((width,))
    b_out = jnp.zeros(())
    return {"layers": layers, "W_out": W_out, "b_out": b_out}


def init_fluct_hidden(key, cfg: LPINNConfig):
    """Hidden parameters θ̃ of the fluctuation network (no output bias)."""
    layers, _ = _init_layers(key, 2, cfg.hidden_widths)
    return {"layers": layers}


def last_hidden(params, z):
    """ψ(z; θ̃), shape (..., Nη). Uses only ``params['layers']``."""
    h = z
    for layer in params["layers"]:
        h = jnp.tanh(h @ layer["W"] + layer["b"])
    return h


def mean_value(mean_params, x, t, *, L: float, T: float):
    """ĥ̄(x, t; γ)."""
    z = scale_z(x, t, L, T)
    psi = last_hidden(mean_params, z)
    return psi @ mean_params["W_out"] + mean_params["b_out"]


def fluct_value(hidden_params, W, x, t, *, L: float, T: float):
    """ĥ'(x, t; θ̃, W) = W · ψ(x, t; θ̃). Output bias is identically 0."""
    z = scale_z(x, t, L, T)
    psi = last_hidden(hidden_params, z)
    return psi @ W


def features(hidden_params, x, t, *, L: float, T: float):
    """Last-hidden activations ψ, shape (..., Nη)."""
    return last_hidden(hidden_params, scale_z(x, t, L, T))


def predict(mean_params, hidden_params, W, x, t, *, L: float, T: float):
    """Full reconstruction ĥ = ĥ̄ + ĥ' (Eq 6)."""
    return mean_value(mean_params, x, t, L=L, T=T) + fluct_value(
        hidden_params, W, x, t, L=L, T=T
    )
