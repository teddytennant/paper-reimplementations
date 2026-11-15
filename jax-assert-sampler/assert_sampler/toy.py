"""The 2D toy benchmark: everything the paper measures with FID on images, measured
here with an energy distance on an eight-mode Gaussian mixture.

Section 4.1 of arXiv:2609.00955 evaluates image checkpoints under one deployment
noise realization per seed, with all compared samplers sharing the same initial
latents and the same realization, and reports the mean over seeds. The same
protocol is used here, with a mapping index standing in for a seed.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp

from . import cim, denoiser, metrics, sampler, schedules
from .diffusion import Schedule

CLIP_X0 = 4.0
"""Clamp on Equation 3's xhat_0. See :func:`assert_sampler.sampler.ddim_step`."""


class Toy(NamedTuple):
    params: list
    mix: object
    sched: Schedule
    x_init: jnp.ndarray
    reference: jnp.ndarray


def setup(seed: int = 0, n_samples: int = 1200, train_steps: int = 4000) -> Toy:
    params, mix, sched, _ = denoiser.pretrained(seed, train_steps)
    k_init, k_ref = jax.random.split(jax.random.key(seed + 1))
    return Toy(params=params, mix=mix, sched=sched,
               x_init=jax.random.normal(k_init, (n_samples, mix.means.shape[1])),
               reference=mix.sample(k_ref, n_samples))


def run(toy: Toy, num_steps: int, etas, key, *, noise=cim.CIMNoise(), mapping=None,
        active=None, eps_fn: Optional[Callable] = None, clip_x0=CLIP_X0):
    return sampler.sample(key, toy.params, toy.sched, num_steps, etas, toy.x_init,
                          eps_fn=eps_fn, noise=noise, mapping=mapping,
                          active=active, clip_x0=clip_x0).sample


def error(toy: Toy, samples) -> float:
    return float(metrics.energy_distance(samples, toy.reference))


def sweep(toy: Toy, num_steps: int, level: float, named_etas: dict, *,
          num_mappings: int = 12, resample_per_step: bool = False,
          active=None, key_seed: int = 11, mapping_seed: int = 100):
    """Evaluate each named eta schedule over ``num_mappings`` deployment realizations.

    Within one realization every sampler sees the same eps_W and the same sampler
    key, so schedules are compared on identical hardware and identical latents.
    Returns {name: (mean, per_mapping_list)}.
    """
    noise = cim.calibrate(toy.params, level, resample_per_step=resample_per_step)
    base = jax.random.key(key_seed)
    out = {name: [] for name in named_etas}
    for m in range(num_mappings):
        mapping = cim.sample_mapping(jax.random.key(mapping_seed + m), toy.params, noise)
        key = jax.random.fold_in(base, m)
        for name, etas in named_etas.items():
            out[name].append(error(toy, run(toy, num_steps, etas, key, noise=noise,
                                            mapping=mapping, active=active)))
    return {name: (sum(v) / len(v), v) for name, v in out.items()}


def default_schedules(num_steps: int, eta_max: float = 0.8):
    """The samplers compared in Section 4.2."""
    return {
        "ddim": schedules.constant_schedule(num_steps, 0.0),
        "assert": schedules.assert_schedule(num_steps, eta_max=eta_max, tau_trans=0.1),
        "step": schedules.step_schedule(num_steps, eta_max=eta_max, tau_trans=0.4),
        "linear": schedules.linear_decay_schedule(num_steps, eta_max=eta_max),
        "reverse_step": schedules.reverse_step_schedule(num_steps, eta_max=eta_max,
                                                        tau_trans=0.4),
    }


def main():
    toy = setup()
    steps, level = 50, 0.03
    named = default_schedules(steps)
    clean = error(toy, run(toy, steps, named["ddim"], jax.random.key(11)))
    print(f"toy: 8-mode 2D mixture, S={steps}, {toy.x_init.shape[0]} samples, "
          f"energy distance to the target")
    print(f"clean deterministic DDIM      {clean: .4f}")
    for label, resample in [("fixed spatial eps_W", False), ("eps_W resampled per step", True)]:
        res = sweep(toy, steps, level, named, num_mappings=6, resample_per_step=resample)
        print(f"\nequivalent weight-noise std {level}, {label}, mean over 6 mappings")
        for name, (mean, _) in res.items():
            print(f"  {name:<14} {mean: .4f}")


if __name__ == "__main__":
    main()
