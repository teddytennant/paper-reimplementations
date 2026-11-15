"""Local quantization-error score for one block."""

import jax.numpy as jnp

from lea_ptq.quant import symmetric_quantize


def local_score(samples, bits, delta=0.05):
    """Cantelli upper score of per-sample linear quantization error.

    samples has length T. Each sample is a sequence of (W, x) pairs, one
    pair per linear layer in the block. x is a vector. For each sample,

        P_t = sum over layers of l2_norm((W - Q_b(W)) @ x)

    T < 2 raises ValueError. mu is the mean of P. sigma is the sample
    standard deviation with ddof=1. The local score is

        s_loc = mu + sigma * sqrt((1 - delta) / delta)

    delta must lie in (0, 1). The default delta=0.05 is a user risk level,
    not a table constant.

    Returns a dict with keys mu, sigma, and s_loc.
    """
    if not (0.0 < float(delta) < 1.0):
        raise ValueError("delta must be in (0, 1)")

    samples = list(samples)
    if len(samples) < 2:
        raise ValueError("T must be at least 2")

    powers = []
    for sample in samples:
        total = jnp.asarray(0.0)
        for weight, activation in sample:
            weight = jnp.asarray(weight)
            activation = jnp.asarray(activation)
            residual = (weight - symmetric_quantize(weight, bits)) @ activation
            total = total + jnp.linalg.norm(residual)
        powers.append(total)

    observed = jnp.stack([jnp.asarray(p) for p in powers])
    mu = jnp.mean(observed)
    sigma = jnp.std(observed, ddof=1)
    s_loc = mu + sigma * jnp.sqrt((1.0 - float(delta)) / float(delta))
    return {"mu": mu, "sigma": sigma, "s_loc": s_loc}
