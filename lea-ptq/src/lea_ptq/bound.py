"""One-sided error bounds used by the local score."""

import math


def cantelli_bound(mu, sigma, delta):
    """Return mu + sigma * sqrt((1 - delta) / delta).

    sigma < 0 or delta outside (0, 1) raises ValueError.
    """
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    if not (0.0 < float(delta) < 1.0):
        raise ValueError("delta must be in (0, 1)")
    return float(mu) + float(sigma) * math.sqrt((1.0 - float(delta)) / float(delta))


def layer_bound(mu_x, sig_x, mu_y, sig_y, cov, A_prev, B_prev, delta):
    """Propagate a Cantelli bound through one affine combination of errors.

    m = mu_x * A_prev + mu_y * B_prev
    v = A_prev**2 * sig_x**2 + B_prev**2 * sig_y**2 + 2 * A_prev * B_prev * cov
    returns m + sqrt(max(v, 0) * (1 - delta) / delta)
    """
    if not (0.0 < float(delta) < 1.0):
        raise ValueError("delta must be in (0, 1)")
    mean = float(mu_x) * float(A_prev) + float(mu_y) * float(B_prev)
    variance = (
        float(A_prev) ** 2 * float(sig_x) ** 2
        + float(B_prev) ** 2 * float(sig_y) ** 2
        + 2.0 * float(A_prev) * float(B_prev) * float(cov)
    )
    return mean + math.sqrt(max(variance, 0.0) * (1.0 - float(delta)) / float(delta))


def eps_total(delta_L, pairs):
    """Return delta_L + sum of (delta_l + delta_tilde_l) over pairs."""
    total = float(delta_L)
    for delta_l, delta_tilde_l in pairs:
        total += float(delta_l) + float(delta_tilde_l)
    return total
