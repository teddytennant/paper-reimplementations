"""SLiSDE stack: linear layers, gated in-flow, linear readout, optional Girsanov."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from . import gates as G
from . import girsanov as gir
from . import scan as S
from . import sde as sde


def init_model(
    key,
    *,
    n_layers=2,
    d=4,
    m=1,
    p=1,
    structure="dense",
    noise="additive",
    gate_scale=0.01,
    use_time=True,
    omega=2.0 * jnp.pi,
):
    """Few-layer SLiSDE with shared z0 and last-layer readout (Eq. 9)."""
    n_time = 4 if use_time else 0
    x_dim = d + n_time
    keys = jax.random.split(key, n_layers + n_layers + 1)
    layers = [
        sde.init_layer(keys[i], d, m, structure=structure, noise=noise) for i in range(n_layers)
    ]
    gate_params = [
        G.init_gates(keys[n_layers + i], d, x_dim, scale=gate_scale) for i in range(1, n_layers)
    ]
    readout = 0.1 * jax.random.normal(keys[-1], (p, d))
    return {
        "layers": layers,
        "gates": gate_params,
        "readout": readout,
        "shift": jnp.zeros((p,)),
        "omega": jnp.asarray(omega),
    }


def _simulate_layer(layer, dW, dt, z0, method, prev_path, times, gate, use_time, omega, parallel):
    F, g = sde.transition_pairs(layer, dW, dt, method=method)
    if gate is not None and prev_path is not None:
        t_left = times[:-1]
        z_left = prev_path[:, :-1, :]
        F, g = G.gated_pairs(gate, F, g, z_left, t_left, omega=omega, use_time=use_time)
    sim = S.parallel_simulate if parallel else S.sequential_simulate
    return sim(F, g, z0)


def simulate(
    params,
    z0,
    dWs,
    dt,
    *,
    times=None,
    method="euler",
    parallel=True,
    rho=0.0,
    mu=None,
    u=None,
    d_eps_Q=None,
    use_time=True,
):
    """Simulate the gated stack.

    dWs: (B, N, m) shared prefix driver, or a list of per-layer increments.
    Optional last-layer Girsanov: pass ``mu`` and/or ``u`` (and ``d_eps_Q``).
    ``z0`` is the N(0,I) draw ξ; under Q the shared state is μ+ξ.
    Returns (Y, latent_path, log dP/dQ).
    """
    layers = params["layers"]
    gate_params = params["gates"]
    n_layers = len(layers)
    omega = params.get("omega", 2.0 * jnp.pi)

    if isinstance(dWs, (list, tuple)):
        dW_list = list(dWs)
    else:
        dW_list = [dWs] * n_layers
    n_steps = dW_list[0].shape[1]
    if times is None:
        times = jnp.linspace(0.0, dt * n_steps, n_steps + 1)
    times = jnp.asarray(times)

    z0_used = z0 if mu is None else z0 + mu
    log_w = jnp.zeros((z0.shape[0],), dtype=z0.dtype)
    path = None

    for i, layer in enumerate(layers):
        dW = dW_list[i]
        last = i == n_layers - 1
        if last and (u is not None or mu is not None):
            d_eps = dW if d_eps_Q is None else d_eps_Q
            u_use = jnp.zeros_like(d_eps) if u is None else u
            d_eps_P = gir.tilt_innovation(d_eps, u_use, dt)
            if n_layers > 1:
                dW = gir.correlated_driver(dW_list[0], d_eps_P, rho)
            else:
                dW = d_eps_P
            log_w = gir.log_dp_over_dq(u_use, d_eps, dt, mu=mu, z0=z0_used)
        elif last and n_layers > 1 and rho != 0.0:
            d_eps = dW if d_eps_Q is None else d_eps_Q
            dW = gir.correlated_driver(dW_list[0], d_eps, rho)

        gate = gate_params[i - 1] if i > 0 and (i - 1) < len(gate_params) else None
        path = _simulate_layer(
            layer,
            dW,
            dt,
            z0_used,
            method,
            path,
            times,
            gate,
            use_time,
            omega,
            parallel,
        )

    Y = jnp.einsum("pd,bnd->bnp", params["readout"], path) + params["shift"]
    return Y, path, log_w


def sample_z0(key, batch, d):
    """Shared initial latent Z_0 ~ N(0, I_d)."""
    return jax.random.normal(key, (batch, d))
