"""Table 2 two-template control: mixture score vs conditioned component score."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from intrinsic_atlas.gaussian import (
    FIG4_LAMBDAS,
    make_conditioned_x0_fn,
    make_mixture_x0_fn,
)
from intrinsic_atlas.sampler import sample_posterior_mean, spread_offset
from intrinsic_atlas.schedules import make_vp_schedule


def test_mixture_two_attractors_conditioned_collapses():
    """Unconditioned mixture has two attractors; conditioning collapses to one component."""
    mus = jnp.array([[-0.8, 0.0], [0.8, 0.0]])
    cov = jnp.diag(jnp.array(FIG4_LAMBDAS))
    sched = make_vp_schedule(T=1000)
    mix_fn = make_mixture_x0_fn(mus, cov, sched)
    cond0 = make_conditioned_x0_fn(mus, cov, sched, component=0)
    cond1 = make_conditioned_x0_fn(mus, cov, sched, component=1)

    n = 24
    key = jax.random.PRNGKey(3)
    x_T = jax.random.normal(key, (n, 2))

    mix = jax.jit(jax.vmap(lambda x: sample_posterior_mean(x, mix_fn, sched)))(x_T)
    y0 = jax.jit(jax.vmap(lambda x: sample_posterior_mean(x, cond0, sched)))(x_T)
    y1 = jax.jit(jax.vmap(lambda x: sample_posterior_mean(x, cond1, sched)))(x_T)

    # Conditioned: every seed goes to the chosen component mean.
    s0, o0 = spread_offset(y0, mus[0])
    s1, o1 = spread_offset(y1, mus[1])
    assert float(s0) < 0.05
    assert float(s1) < 0.05
    assert float(o0) < 0.05
    assert float(o1) < 0.05

    # Unconditioned: endpoints split across the two templates (not a single blob at 0).
    assigned = jnp.argmin(
        jnp.stack(
            [
                jnp.linalg.norm(mix - mus[0], axis=-1),
                jnp.linalg.norm(mix - mus[1], axis=-1),
            ]
        ),
        axis=0,
    )
    n0 = int(jnp.sum(assigned == 0))
    n1 = int(jnp.sum(assigned == 1))
    assert n0 >= 3 and n1 >= 3

    # Unconditioned centroids sit off the true component means (late-step pull; Table 2a).
    mix0 = mix[assigned == 0]
    mix1 = mix[assigned == 1]
    _, off0 = spread_offset(mix0, mus[0])
    _, off1 = spread_offset(mix1, mus[1])
    assert float(off0) > float(o0)
    assert float(off1) > float(o1)
    origin_spread, _ = spread_offset(mix, jnp.zeros(2))
    assert float(origin_spread) > 0.2
