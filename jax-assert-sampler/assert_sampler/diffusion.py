"""Forward diffusion schedule, timestep subsequence, and the analytic score of a 2D
Gaussian mixture.

Notation follows ASSERT (arXiv:2609.00955): alpha_t = 1 - beta_t and
alpha_bar_t = prod_{r=1}^{t} alpha_r, Section 3.2.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class Schedule(NamedTuple):
    """Discrete-time forward process over ``num_train_steps`` timesteps."""

    betas: jnp.ndarray  # (T,)
    alphas_cumprod: jnp.ndarray  # (T,), alpha_bar_t for t = 0 .. T-1

    @property
    def num_train_steps(self) -> int:
        return int(self.betas.shape[0])

    def alpha_bar(self, t: jnp.ndarray) -> jnp.ndarray:
        """alpha_bar at integer timestep ``t``; ``t = -1`` maps to 1.0.

        The final DDIM transition targets a "step before the first one", where the
        forward process has not yet added anything, so alpha_bar = 1.
        """
        t = jnp.asarray(t)
        return jnp.where(t < 0, 1.0, self.alphas_cumprod[jnp.maximum(t, 0)])


def linear_schedule(num_train_steps: int = 1000, beta_start: float = 1e-4,
                    beta_end: float = 0.02) -> Schedule:
    """The DDPM linear beta schedule, which DDIM subsamples at inference time."""
    betas = jnp.linspace(beta_start, beta_end, num_train_steps)
    return Schedule(betas=betas, alphas_cumprod=jnp.cumprod(1.0 - betas))


def uniform_subsequence(num_train_steps: int, num_steps: int) -> jnp.ndarray:
    """DDIM's uniform subsequence, Section 4.1.

    An S-step run subsamples the training horizon T as {floor(j T / S)} for
    j = 0 .. S-1 and traverses it in descending order. Index i = 0 is the early,
    high-noise step.
    """
    j = jnp.arange(num_steps)
    ts = jnp.floor(j * num_train_steps / num_steps).astype(jnp.int32)
    return ts[::-1]


class GaussianMixture(NamedTuple):
    """Isotropic 2D mixture. Its noised marginals stay Gaussian mixtures, so the
    score of q_t is available in closed form."""

    means: jnp.ndarray  # (K, D)
    weights: jnp.ndarray  # (K,)
    std: float

    def sample(self, key, n: int) -> jnp.ndarray:
        k_comp, k_noise = jax.random.split(key)
        idx = jax.random.categorical(k_comp, jnp.log(self.weights), shape=(n,))
        return self.means[idx] + self.std * jax.random.normal(
            k_noise, (n, self.means.shape[1]))


def default_mixture() -> GaussianMixture:
    """Eight modes on a ring of radius 2.5, std 0.25. Well separated, so a sampler
    that drifts off the ring or collapses modes is easy to detect."""
    angles = jnp.arange(8) * (2.0 * jnp.pi / 8)
    means = 2.5 * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
    return GaussianMixture(means=means, weights=jnp.full((8,), 1.0 / 8), std=0.25)


def marginal_log_prob(mix: GaussianMixture, x: jnp.ndarray,
                      alpha_bar: jnp.ndarray) -> jnp.ndarray:
    """log q_t(x) where x_t = sqrt(alpha_bar) x_0 + sqrt(1 - alpha_bar) eps."""
    d = x.shape[-1]
    var = alpha_bar * mix.std ** 2 + (1.0 - alpha_bar)
    centers = jnp.sqrt(alpha_bar) * mix.means  # (K, D)
    sq = jnp.sum((x[..., None, :] - centers) ** 2, axis=-1)  # (..., K)
    logits = (jnp.log(mix.weights) - 0.5 * sq / var
              - 0.5 * d * jnp.log(2.0 * jnp.pi * var))
    return jax.scipy.special.logsumexp(logits, axis=-1)


def analytic_eps(mix: GaussianMixture, x: jnp.ndarray,
                 alpha_bar: jnp.ndarray) -> jnp.ndarray:
    """Bayes-optimal epsilon prediction, eps*(x, t) = -sqrt(1 - alpha_bar) grad log q_t(x).

    This is the target the learned denoiser regresses onto, and the exact
    ``eps_theta`` used by the tests that check sampler correctness without any
    network error in the way.
    """
    score = jax.grad(lambda z: marginal_log_prob(mix, z, alpha_bar).sum())(x)
    return -jnp.sqrt(1.0 - alpha_bar) * score
