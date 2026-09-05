"""Feedback AWGN channel and the per-phase power constraint.

Section III-A. Over T phases the transmitter sends x_t in R^n, the receiver gets

    y_t = x_t + n_t,            n_t ~ N(0, sigma_ff^2 I_n)        (10)

and returns y_t to the transmitter over a passive noisy feedback link,

    y~_t = y_t + z_t,           z_t ~ N(0, sigma_fb^2 I_n)        (11)

for t in [T-1]. The transmitter never sees y_T; there is nothing left to do with
it. Each phase is subject to

    (1/n) E[ ||x_t||^2 ] <= P                                     (9)

with P = 1 in all of the paper's experiments. The inner rate is n/M with
M = n T, since the paper uses the bit-based rather than sub-block-based feedback
protocol.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


def snr_db_to_sigma(snr_db: float) -> float:
    """The paper defines the channel SNR as 1/sigma^2, so sigma = 10^(-snr/20)."""
    return float(10.0 ** (-snr_db / 20.0))


@dataclass(frozen=True)
class ChannelConfig:
    sigma_ff: float
    sigma_fb: float
    power: float = 1.0

    @classmethod
    def from_snr_db(
        cls, ff_db: float, fb_db: float = 20.0, power: float = 1.0
    ) -> "ChannelConfig":
        return cls(snr_db_to_sigma(ff_db), snr_db_to_sigma(fb_db), power)

    @property
    def sigma_eff(self) -> float:
        """sqrt(sigma_ff^2 + sigma_fb^2), the noise the encoder sees on y~ (eq. 18)."""
        return float((self.sigma_ff**2 + self.sigma_fb**2) ** 0.5)


def normalize_power(x: jax.Array, power: float = 1.0) -> jax.Array:
    """Scale a batch of parity messages so that mean(x^2) == power exactly.

    (9) is an expectation over the message and noise distributions, and the
    paper does not say how it is enforced. This is the usual choice in the
    feedback-coding literature (DeepCode, AttentionCode, GBAF all do some
    version of it): divide the whole phase-t batch by its own RMS. It is a
    single positive scalar, so it cannot change any sign, which matters because
    the first-phase message x_1 must keep the sign of 2c-1 for the soft syndrome
    to mean anything (eq. 16).

    Note the scale is a batch statistic, so at evaluation time the realised
    power depends on the batch. With a batch of a few hundred that is a fraction
    of a percent, and the alternative, a running average, would make the tests
    depend on evaluation order.
    """
    mean_sq = jnp.mean(jnp.square(x))
    return x * jnp.sqrt(power / jnp.maximum(mean_sq, 1e-12))


def average_power(x: jax.Array) -> jax.Array:
    """(1/n) E[||x||^2], estimated from a batch. The left side of (9)."""
    return jnp.mean(jnp.square(x))


def forward(x: jax.Array, sigma_ff: float, key: jax.Array) -> jax.Array:
    """Equation (10): y_t = x_t + n_t."""
    return x + sigma_ff * jax.random.normal(key, x.shape, dtype=x.dtype)


def feedback(y: jax.Array, sigma_fb: float, key: jax.Array) -> jax.Array:
    """Equation (11): y~_t = y_t + z_t, the passive noisy feedback."""
    return y + sigma_fb * jax.random.normal(key, y.shape, dtype=y.dtype)
