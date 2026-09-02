"""The negated Branin task from the paper's motivating example.

Section 5 and Figure 1 build the offline dataset by sampling 50,000 points
uniformly from [-5, 10] x [0, 15] and keeping only the bottom 60% by objective
value, which is what makes the setting offline: the high-value region is not in
the data. The oracle is analytic, so every returned candidate can be scored.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["branin", "negated_branin", "BRANIN_LO", "BRANIN_HI", "OfflineTask", "branin_task"]

BRANIN_LO = np.array([-5.0, 0.0], dtype=np.float32)
BRANIN_HI = np.array([10.0, 15.0], dtype=np.float32)


def branin(x: jnp.ndarray) -> jnp.ndarray:
    """The standard Branin function, minimum 0.397887 at three points."""
    x = jnp.atleast_2d(x)
    a, b, c, r, s, t = 1.0, 5.1 / (4 * jnp.pi**2), 5.0 / jnp.pi, 6.0, 10.0, 1.0 / (8 * jnp.pi)
    x1, x2 = x[..., 0], x[..., 1]
    return a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * jnp.cos(x1) + s


def negated_branin(x: jnp.ndarray) -> jnp.ndarray:
    """The maximization objective, so the three Branin minima become the optima."""
    return -branin(x)


@dataclass
class OfflineTask:
    """An offline dataset D together with the oracle used only for evaluation.

    ``y_lo`` and ``y_hi`` are the extrema of the full sample before the top
    scores were removed. Normalizing against them is the Design-Bench convention
    the paper follows, so a normalized score above ``best_offline`` means the
    method left the data behind.
    """

    X: np.ndarray
    y: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    oracle: callable
    y_lo: float
    y_hi: float

    @property
    def best_offline(self) -> float:
        return float(self.normalize(self.y.max()))

    def normalize(self, y):
        return (np.asarray(y) - self.y_lo) / (self.y_hi - self.y_lo)

    def evaluate(self, X) -> np.ndarray:
        """Normalized oracle scores for a batch of candidate designs."""
        return self.normalize(np.asarray(self.oracle(jnp.asarray(X))))


def branin_task(key, n_samples: int = 50_000, keep_fraction: float = 0.6) -> OfflineTask:
    """Sample the offline dataset of Figure 1: uniform draws, bottom fraction kept."""
    X = jax.random.uniform(
        key, (n_samples, 2), minval=jnp.asarray(BRANIN_LO), maxval=jnp.asarray(BRANIN_HI)
    )
    y = np.asarray(negated_branin(X))
    X = np.asarray(X)
    cut = int(np.ceil(keep_fraction * n_samples))
    keep = np.argsort(y, kind="stable")[:cut]
    return OfflineTask(
        X=X[keep].astype(np.float32),
        y=y[keep].astype(np.float32),
        lo=BRANIN_LO,
        hi=BRANIN_HI,
        oracle=negated_branin,
        y_lo=float(y.min()),
        y_hi=float(y.max()),
    )
