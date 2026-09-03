"""The DDIM sampler with a per-step stochasticity parameter, Equations 3-5 of
arXiv:2609.00955, driven under the CIM noise model of Section 3.1.

ASSERT adds no network evaluations and changes no parameters. All it changes is
the eta_i handed to :func:`ddim_step` and the random draws that go with it.
"""

from __future__ import annotations

import functools
from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp

from . import cim
from .diffusion import Schedule, analytic_eps, uniform_subsequence


def sigma_t_to_s(eta: jnp.ndarray, ab_t: jnp.ndarray, ab_s: jnp.ndarray) -> jnp.ndarray:
    """Equation 4.

    sigma_{t->s}(eta) = eta sqrt((1 - ab_s) / (1 - ab_t)) sqrt(1 - ab_t / ab_s).

    eta is dimensionless and is not itself the injected standard deviation; the
    pair (t, s) sets the scale. eta = 0 is deterministic DDIM, eta = 1 is the
    stochastic DDIM variance for that pair.
    """
    ratio = jnp.clip((1.0 - ab_s) / (1.0 - ab_t), 0.0, None)
    return eta * jnp.sqrt(ratio) * jnp.sqrt(jnp.clip(1.0 - ab_t / ab_s, 0.0, None))


def predict_x0(x: jnp.ndarray, eps: jnp.ndarray, ab_t: jnp.ndarray) -> jnp.ndarray:
    """Equation 3."""
    return (x - jnp.sqrt(1.0 - ab_t) * eps) / jnp.sqrt(ab_t)


def ddim_step(x, eps, ab_t, ab_s, eta, z, clip_x0: Optional[float] = None):
    """Equation 5.

    x_s = sqrt(ab_s) xhat_0 + sqrt(1 - ab_s - sigma^2) eps_theta + sigma z.

    ``clip_x0`` clamps xhat_0 to [-clip_x0, clip_x0] before the update. The paper
    does not write this down, but it is the sample clipping the public DDIM
    checkpoints ship with, and without it a perturbed network can send the
    Equation 3 estimate to infinity through the 1 / sqrt(ab_t) factor at the
    high-noise end. Leave it None to get Equation 5 verbatim.
    """
    sigma = sigma_t_to_s(eta, ab_t, ab_s)
    x0 = predict_x0(x, eps, ab_t)
    if clip_x0 is not None:
        x0 = jnp.clip(x0, -clip_x0, clip_x0)
    direction = jnp.sqrt(jnp.clip(1.0 - ab_s - sigma ** 2, 0.0, None)) * eps
    return jnp.sqrt(ab_s) * x0 + direction + sigma * z


def step_coefficient(ab_t, ab_s, eta):
    """Equation 6, the local coefficient c_i with delta x_s = c_i delta eps_i.

    c_i = sqrt(1 - ab_s - sigma^2) - sqrt(ab_s / ab_t) sqrt(1 - ab_t).
    """
    sigma = sigma_t_to_s(eta, ab_t, ab_s)
    return (jnp.sqrt(jnp.clip(1.0 - ab_s - sigma ** 2, 0.0, None))
            - jnp.sqrt(ab_s / ab_t) * jnp.sqrt(1.0 - ab_t))


def x0_sensitivity(ab_t):
    """Equation 7, ||delta xhat_0|| = sqrt((1 - ab_t) / ab_t) ||delta eps_i||.

    This grows without bound toward the high-noise end, which is half of why the
    early stage is the fragile one.
    """
    return jnp.sqrt((1.0 - ab_t) / ab_t)


class Trajectory(NamedTuple):
    x: jnp.ndarray  # (S + 1, N, D), states x_{t_0} .. x_{t_S}
    etas: jnp.ndarray  # (S,)
    timesteps: jnp.ndarray  # (S,)

    @property
    def sample(self) -> jnp.ndarray:
        return self.x[-1]


def timestep_pairs(sched: Schedule, num_steps: int):
    """Returns (timesteps, ab_t, ab_s) for the S transitions of an S-step run.

    The last transition targets alpha_bar = 1, so it lands exactly on xhat_0.
    """
    ts = uniform_subsequence(sched.num_train_steps, num_steps)
    prev = jnp.concatenate([ts[1:], jnp.array([-1], dtype=ts.dtype)])
    return ts, sched.alpha_bar(ts), sched.alpha_bar(prev)


@functools.lru_cache(maxsize=8)
def _mlp_eps_fn(sigma_y: float) -> Callable:
    """The default ``eps_fn``, cached so the jitted step below is compiled once."""
    from . import denoiser  # imported here so cim and denoiser stay independent

    def fn(params, x, t_norm, ab_t, key):
        return denoiser.apply(params, x, t_norm, key=key, sigma_y=sigma_y)

    return fn


@functools.lru_cache(maxsize=8)
def _jitted_step(eps_fn: Callable, clip_x0) -> Callable:
    """One compiled denoising step: apply eps_W, evaluate the network, update."""

    @jax.jit
    def go(params, mapping, gate, x, t_norm, ab_t, ab_s, eta, z, key):
        eff = (cim.perturb(params, [{"w": gate * m["w"]} for m in mapping])
               if mapping is not None else params)
        eps = eps_fn(eff, x, t_norm, ab_t, key)
        return ddim_step(x, eps, ab_t, ab_s, eta, z, clip_x0)

    return go


def sample(key, params, sched: Schedule, num_steps: int, etas: jnp.ndarray,
           x_init: jnp.ndarray, *, eps_fn: Optional[Callable] = None,
           noise: cim.CIMNoise = cim.CIMNoise(), mapping=None,
           active: Optional[jnp.ndarray] = None,
           clip_x0: Optional[float] = None) -> Trajectory:
    """Run S DDIM steps from ``x_init``.

    ``eps_fn(params, x, t_norm, alpha_bar_t, key)`` returns the epsilon prediction.
    The default is the MLP denoiser. ``mapping`` is the persistent eps_W of one deployment
    realization; pass None for clean inference. ``active`` is the a_i in
    Equation 8, a 0/1 mask that lets the hardware mapping be switched on over a
    chosen interval only, which is the Figure 4 stage experiment.
    """
    if eps_fn is None:
        eps_fn = _mlp_eps_fn(float(noise.sigma_y))
    etas = jnp.asarray(etas)
    if etas.shape != (num_steps,):
        raise ValueError(f"etas must have shape ({num_steps},), got {etas.shape}")
    if active is None:
        active = jnp.ones((num_steps,))

    ts, ab_t, ab_s = timestep_pairs(sched, num_steps)
    if mapping is None:
        # No hardware mapping: eps_W = 0. ``params`` may be None when ``eps_fn``
        # ignores it, as the analytic-score tests do.
        mapping = cim.zero_mapping(params) if params is not None else None

    step = _jitted_step(eps_fn, clip_x0)
    t_norm = ts / sched.num_train_steps
    xs = [x_init]
    x = x_init
    for i in range(num_steps):
        key, k_map, k_out, k_z = jax.random.split(key, 4)
        step_mapping = (cim.sample_mapping(k_map, params, noise)
                        if noise.resample_per_step and not noise.is_clean
                        else mapping)
        # active[i] is a_i in Equation 8: hardware mapping on or off for this step.
        z = jax.random.normal(k_z, x.shape)
        x = step(params, step_mapping, active[i], x, t_norm[i], ab_t[i], ab_s[i],
                 etas[i], z, k_out if noise.sigma_y > 0.0 else None)
        xs.append(x)
    return Trajectory(x=jnp.stack(xs), etas=etas, timesteps=ts)


def analytic_eps_fn(mix):
    """An ``eps_fn`` that uses the exact eps* of the mixture, ignoring params.

    Used by the tests that check the sampler itself, with no network error in the
    way.
    """

    def fn(params, x, t_norm, ab_t, key):
        return analytic_eps(mix, x, ab_t)

    return fn
