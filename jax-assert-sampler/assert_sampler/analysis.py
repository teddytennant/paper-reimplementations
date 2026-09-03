"""First-order trajectory analysis, Equations 8-11 of arXiv:2609.00955.

With f_i the DDIM update of Equation 5, Delta theta the fixed spatial weight
perturbation of one hardware mapping, and the same z_i used on the clean and
noisy trajectories, linearizing gives

    e_{i+1} = J_i e_i + a_i H_i Delta theta,   J_i = df_i/dx,  H_i = df_i/dtheta,

and unrolling to the final sample gives

    e_S = sum_i P_{S,i+1} d_i,  d_i = a_i H_i Delta theta,
    P_{S,i+1} = J_{S-1} ... J_{i+1},

so with u_i = P_{S,i+1} d_i,

    ||e_S||^2 = sum_i ||u_i||^2 + 2 sum_{i<j} <u_i, u_j>.

Every J_i e_i and H_i Delta theta here is one JVP, so nothing is materialized as
a matrix. The cross terms are the point: they exist only because one Delta theta
is reused across steps, and Equation 11 says stochastic sampling helps by making
them less positive.

Every trajectory and derivative in this module runs under
jax.default_matmul_precision("highest"). XLA's GPU default for float32 matmuls
is TF32 (10 mantissa bits), which is fine for training and sampling but not
here: the finite differences below subtract two nearly-identical 20-step
trajectories, and Equation 9's unrolled sum is an algebraic identity, both of
which need the bits TF32 throws away. Measured on an H200, forcing "highest"
brings GPU results back in line with CPU: the unrolled-sum residual drops from
7e-4 relative (fails the 1e-4 check) to 5e-7, and the finite-difference tests
go from diverging as h shrinks to matching CPU's convergence curve within a few
percent at every h tested.
"""

from __future__ import annotations

import functools
from typing import Callable, Optional

import jax
import jax.numpy as jnp

from . import denoiser
from .sampler import ddim_step, timestep_pairs


def make_step_fn(sched, num_steps: int, etas: jnp.ndarray,
                 eps_fn: Optional[Callable] = None,
                 clip_x0: Optional[float] = None) -> Callable:
    """Build f_i(theta, x, i, z), one DDIM transition as a function of weights and state."""
    ts, ab_t, ab_s = timestep_pairs(sched, num_steps)
    t_norm = ts / sched.num_train_steps
    if eps_fn is None:
        eps_fn = lambda th, x, i: denoiser.apply(th, x, t_norm[i])

    def f(theta, x, i, z):
        eps = eps_fn(theta, x, i)
        return ddim_step(x, eps, ab_t[i], ab_s[i], etas[i], z, clip_x0)

    return f


@functools.lru_cache(maxsize=8)
def _jitted_jvp(step_fn: Callable) -> Callable:
    """One compiled JVP of f_i, with the step index traced so every step reuses it."""

    @jax.jit
    def go(params, x, i, z, tangent_theta, tangent_x):
        return jax.jvp(lambda th, xx: step_fn(th, xx, i, z),
                       (params, x), (tangent_theta, tangent_x))[1]

    return go


def reference_trajectory(key, params, sched, num_steps: int, etas: jnp.ndarray,
                         x_init: jnp.ndarray, eps_fn: Optional[Callable] = None,
                         clip_x0: Optional[float] = None):
    """Run the clean trajectory and keep the z_i, so the noisy comparison can reuse them.

    ``clip_x0`` defaults to off here. The clamp is not differentiable at its corners,
    and the point of this module is a linearization, so the analysis runs on the
    unclamped map of Equation 5.
    """
    step_fn = make_step_fn(sched, num_steps, etas, eps_fn, clip_x0)
    xs, zs = [x_init], []
    x = x_init
    with jax.default_matmul_precision("highest"):
        for i in range(num_steps):
            key, sub = jax.random.split(key)
            z = jax.random.normal(sub, x.shape)
            zs.append(z)
            x = step_fn(params, x, i, z)
            xs.append(x)
    return jnp.stack(xs), jnp.stack(zs), step_fn


def as_param_tangent(params, mapping):
    """Turn a weight-perturbation map into a tangent shaped like ``params``.

    Biases are digital and get zero.
    """
    return [{"w": m["w"], "b": jnp.zeros_like(p["b"])}
            for p, m in zip(params, mapping)]


def _scaled(tangent, s):
    return [{"w": s * l["w"], "b": s * l["b"]} for l in tangent]


def _zeros_like(tangent):
    return [{"w": jnp.zeros_like(l["w"]), "b": jnp.zeros_like(l["b"])}
            for l in tangent]


def _per_step(dtheta, num_steps):
    """Accept one Delta theta reused at every step, or a list of S of them.

    A list is the resampled-noise control: it breaks the "same Delta theta" premise
    that Equation 10's cross terms rely on.
    """
    if isinstance(dtheta, (list, tuple)) and dtheta and isinstance(dtheta[0], (list, tuple)):
        if len(dtheta) != num_steps:
            raise ValueError("expected one Delta theta per step")
        return list(dtheta)
    return [dtheta] * num_steps


def linearized_error(params, dtheta, xs, zs, step_fn,
                     active: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """Iterate Equation 8 and return e_S."""
    num_steps = zs.shape[0]
    if active is None:
        active = jnp.ones((num_steps,))
    thetas = _per_step(dtheta, num_steps)
    jvp = _jitted_jvp(step_fn)
    e = jnp.zeros_like(xs[0])
    with jax.default_matmul_precision("highest"):
        for i in range(num_steps):
            e = jvp(params, xs[i], i, zs[i], _scaled(thetas[i], active[i]), e)
    return e


def propagated_terms(params, dtheta, xs, zs, step_fn,
                     active: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """u_i = P_{S,i+1} d_i for every i, Equation 9. Shape (S, ...)."""
    num_steps = zs.shape[0]
    if active is None:
        active = jnp.ones((num_steps,))
    thetas = _per_step(dtheta, num_steps)
    jvp = _jitted_jvp(step_fn)
    zero = _zeros_like(thetas[0])
    us = []
    with jax.default_matmul_precision("highest"):
        for i in range(num_steps):
            v = jvp(params, xs[i], i, zs[i], _scaled(thetas[i], active[i]),
                    jnp.zeros_like(xs[i]))
            for j in range(i + 1, num_steps):
                v = jvp(params, xs[j], j, zs[j], zero, v)
            us.append(v)
    return jnp.stack(us)


def decompose(us: jnp.ndarray):
    """Split ||e_S||^2 into its marginal and cross parts, Equation 10.

    Returns (marginal, cross, total) summed over the batch.
    """
    sq = jnp.sum(us ** 2, axis=-1)  # (S, N)
    marginal = jnp.sum(sq)
    total = jnp.sum(jnp.sum(jnp.sum(us, axis=0) ** 2, axis=-1))
    return marginal, total - marginal, total


def cross_step_cosine(us: jnp.ndarray) -> jnp.ndarray:
    """Mean cosine between u_i and u_j over pairs i < j, averaged over the batch.

    This is the alignment Equation 11 predicts stochastic sampling should reduce.
    Only steps with a nonzero contribution take part, so masked-out steps do not
    dilute the average.
    """
    s = us.shape[0]
    norms = jnp.linalg.norm(us, axis=-1)  # (S, N)
    unit = us / jnp.maximum(norms, 1e-12)[..., None]
    gram = jnp.einsum("ind,jnd->ijn", unit, unit)
    live = (norms > 1e-10).astype(us.dtype)
    pair = live[:, None, :] * live[None, :, :]
    tri = jnp.triu(jnp.ones((s, s), us.dtype), k=1)[..., None]
    w = pair * tri
    return jnp.sum(gram * w) / jnp.maximum(jnp.sum(w), 1e-12)


def finite_difference_error(params, dtheta, xs, zs, step_fn, h: float,
                            active: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """(x_S(theta + h dtheta) - x_S(theta)) / h, reusing the reference z_i.

    As h -> 0 this must converge to :func:`linearized_error`.
    """
    num_steps = zs.shape[0]
    if active is None:
        active = jnp.ones((num_steps,))
    thetas = _per_step(dtheta, num_steps)
    x = xs[0]
    with jax.default_matmul_precision("highest"):
        for i in range(num_steps):
            pert = [{"w": p["w"] + h * active[i] * d["w"], "b": p["b"]}
                    for p, d in zip(params, thetas[i])]
            x = step_fn(pert, x, i, zs[i])
    return (x - xs[-1]) / h
