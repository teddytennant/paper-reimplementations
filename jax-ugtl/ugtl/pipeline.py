"""Algorithm 4, the trajectory-learning framework, instantiated as UGTL.

The three stages are trajectory construction (Algorithm 5), trajectory modeling
(the conditional diffusion model), and candidate generation (Algorithm 6).
``run_ugtl`` runs them in that order. Passing ``constructor="random"`` swaps in
the incoherent constructor of Section 4.2 and leaves the other two stages
untouched, which is the controlled comparison Section 5.3.2 runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from ugtl.diffusion import train_diffusion
from ugtl.generation import generate_candidates, target_score
from ugtl.surrogate import train_ensemble
from ugtl.trajectory import construct_random_trajectories, construct_trajectories

__all__ = ["Normalizer", "UGTLConfig", "UGTLResult", "run_ugtl"]


@dataclass
class Normalizer:
    """Affine map between the design box [lo, hi] and the [-1, 1] box the
    diffusion model works in."""

    lo: np.ndarray
    hi: np.ndarray

    def forward(self, x):
        lo, hi = jnp.asarray(self.lo), jnp.asarray(self.hi)
        return 2.0 * (jnp.asarray(x) - lo) / (hi - lo) - 1.0

    def inverse(self, z):
        lo, hi = jnp.asarray(self.lo), jnp.asarray(self.hi)
        return lo + (jnp.asarray(z) + 1.0) * 0.5 * (hi - lo)


@dataclass
class UGTLConfig:
    """Hyperparameters, named after the symbols in Algorithms 5 and 6.

    The defaults are scaled down from Section 5.1 so a full run finishes on a
    CPU in seconds. The paper uses H = 64, K = 20, N_traj = 4000, M_ens = 10,
    T_diff = 200, N_gen = 1000, M_pool = 2048 and Q = 128.
    """

    horizon: int = 32
    n_traj: int = 400
    p_init: float = 20.0
    K: int = 20
    xi: float = 0.05
    rho: float = 0.5
    c_temp: float = 0.5
    n_ensemble: int = 5
    ensemble_steps: int = 800
    ensemble_width: int = 128
    T_diff: int = 100
    diffusion_steps: int = 3000
    diffusion_width: int = 256
    p_drop: float = 0.25
    lambda_tar: float = 1.3
    s_cfg: float = 1.5
    H_ctx: int = 16
    n_gen: int = 256
    M_pool: int = 512
    Q: int = 16


@dataclass
class UGTLResult:
    candidates: np.ndarray
    trajectories: object
    surrogate: object
    model: object
    y_tar: float = field(default=0.0)


def run_ugtl(key, task, config: UGTLConfig | None = None, constructor: str = "ugtl") -> UGTLResult:
    """Run the three stages of Algorithm 4 on an offline task.

    ``task`` needs ``X``, ``y``, ``lo`` and ``hi``. The oracle is never touched
    here; it is only for scoring what comes back.
    """
    config = config or UGTLConfig()
    k_sur, k_traj, k_diff, k_gen = jax.random.split(key, 4)
    X = np.asarray(task.X)

    # The paper writes y_tar = lambda_tar max y and xi = 0.05, both of which read
    # the raw Design-Bench scores, which are positive and O(1). Negated Branin is
    # neither, so the pipeline min-max scales the observed scores to [0, 1] first.
    # The map is affine and increasing, so it changes nothing about which designs
    # rank where; it just makes lambda_tar > 1 an extrapolation and xi a fraction
    # of the observed range in either sign convention.
    y_raw = np.asarray(task.y)
    span = float(y_raw.max() - y_raw.min()) or 1.0
    y = ((y_raw - y_raw.min()) / span).astype(np.float32)

    surrogate = train_ensemble(
        k_sur,
        X,
        y,
        n_members=config.n_ensemble,
        width=config.ensemble_width,
        steps=config.ensemble_steps,
    )

    # Stage 1, trajectory construction.
    if constructor == "ugtl":
        trajectories = construct_trajectories(
            k_traj,
            X,
            y,
            surrogate,
            horizon=config.horizon,
            n_traj=config.n_traj,
            p_init=config.p_init,
            K=config.K,
            xi=config.xi,
            rho=config.rho,
            c_temp=config.c_temp,
        )
    elif constructor == "random":
        trajectories = construct_random_trajectories(
            k_traj,
            X,
            y,
            horizon=config.horizon,
            n_traj=config.n_traj,
            p_init=config.p_init,
            rho=config.rho,
        )
    else:
        raise ValueError(f"unknown constructor {constructor!r}")

    # Stage 2, trajectory modeling.
    normalizer = Normalizer(task.lo, task.hi)
    model = train_diffusion(
        k_diff,
        normalizer.forward(trajectories.designs),
        jnp.asarray(trajectories.scores[:, -1]),
        T=config.T_diff,
        steps=config.diffusion_steps,
        width=config.diffusion_width,
        p_drop=config.p_drop,
        y_scale=(0.0, 1.0),
    )

    # Stage 3, candidate generation.
    y_tar = target_score(y, config.lambda_tar)
    candidates = generate_candidates(
        k_gen,
        model,
        trajectories,
        surrogate,
        normalizer,
        y_tar,
        Q=config.Q,
        n_gen=config.n_gen,
        H_ctx=config.H_ctx,
        M_pool=config.M_pool,
        s_cfg=config.s_cfg,
    )
    return UGTLResult(candidates, trajectories, surrogate, model, y_tar)
