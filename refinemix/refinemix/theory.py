"""Equation 3 and Theorem 1 on a toy enumerable domain.

Domains are disjoint sets of short sequences. The ideal RefineMix denoiser is

    x_theta(x0 | xt) = p(x0 | xt)    if t > tmax
                     = m(x0 | xt)    if t <= tmax
    m = lam * q + (1 - lam) * p

and the overlap is

    C(t) = E_{x ~ q(x_t)} [1 - c(x)],    c(x) = lam * q(x) / m(x).

Theorem 1: a RefineMix sampler started from pure noise (alpha(1) = 0, so
x_1 is drawn from pi and does not depend on x_0) generates a source sequence
with probability

    P(x0_hat in A) = (lam / (1 - lam)) * C(tmax)
                   <= (lam / (1 - lam)) * C(t')    for every t' >= tmax.

The reverse chain is enumerated exactly for the masked kernel. The schedule
used here is alpha_ideal(t) = 1 - t, not the eps-truncated experimental
schedule. Lemma 5 says C(t) is nondecreasing; `overlap_grid` evaluates it.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass


def alpha_ideal(t) -> float:
    """Idealized keep probability with alpha(1) = 0."""
    return 1.0 - float(t)


@dataclass
class ToyDomain:
    target: list[tuple]
    source: list[tuple]
    mask_id: int

    def __post_init__(self):
        self.target = [tuple(int(tok) for tok in seq) for seq in self.target]
        self.source = [tuple(int(tok) for tok in seq) for seq in self.source]
        if not self.target or not self.source:
            raise ValueError("both domains need at least one sequence")
        if set(self.target) & set(self.source):
            raise ValueError("source and target supports must be disjoint")
        lengths = {len(seq) for seq in self.target + self.source}
        if lengths != {len(self.target[0])}:
            raise ValueError("every sequence must have the same length")
        self.length = len(self.target[0])
        self.p = {seq: 1.0 / len(self.target) for seq in self.target}
        self.q = {seq: 1.0 / len(self.source) for seq in self.source}
        tokens = {self.mask_id}
        for seq in self.target + self.source:
            tokens.update(seq)
        self.alphabet = sorted(tokens)
        if any(self.mask_id in seq for seq in self.target + self.source):
            raise ValueError("clean sequences must not contain the mask id")


def token_forward_prob(zt: int, z0: int, alpha: float, mask_id: int) -> float:
    """Masked equation 1 for one token."""
    if z0 == mask_id:
        return 1.0 if zt == mask_id else 0.0
    if zt == z0:
        return float(alpha)
    if zt == mask_id:
        return 1.0 - float(alpha)
    return 0.0


def sequence_forward_prob(xt, x0, alpha: float, mask_id: int) -> float:
    prob = 1.0
    for zt, z0 in zip(xt, x0):
        prob *= token_forward_prob(int(zt), int(z0), alpha, mask_id)
        if prob == 0.0:
            return 0.0
    return prob


def noisy_marginal(xt, clean: dict, alpha: float, mask_id: int) -> float:
    total = 0.0
    for x0, prior in clean.items():
        lik = sequence_forward_prob(xt, x0, alpha, mask_id)
        if lik:
            total += lik * prior
    return total


def mix_clean(p_clean: dict, q_clean: dict, lam: float) -> dict:
    keys = set(p_clean) | set(q_clean)
    return {
        x: lam * q_clean.get(x, 0.0) + (1.0 - lam) * p_clean.get(x, 0.0) for x in keys
    }


def classifier_c(xt, t: float, lam: float, domain: ToyDomain) -> float | None:
    """c(xt) = lam * q(xt) / m(xt) at noise level t. None if m(xt) = 0."""
    alpha = alpha_ideal(t)
    q = noisy_marginal(xt, domain.q, alpha, domain.mask_id)
    p = noisy_marginal(xt, domain.p, alpha, domain.mask_id)
    m = lam * q + (1.0 - lam) * p
    if m == 0.0:
        return None
    return lam * q / m


def overlap_C(domain: ToyDomain, t: float, lam: float) -> float:
    """C(t) = E_{x ~ q(x_t)}[1 - c(x)] by enumerating noisy sequences."""
    alpha = alpha_ideal(t)
    total = 0.0
    for xt in itertools.product(domain.alphabet, repeat=domain.length):
        q = noisy_marginal(xt, domain.q, alpha, domain.mask_id)
        if q == 0.0:
            continue
        p = noisy_marginal(xt, domain.p, alpha, domain.mask_id)
        m = lam * q + (1.0 - lam) * p
        c = lam * q / m
        total += q * (1.0 - c)
    return total


def overlap_grid(domain: ToyDomain, lam: float, times) -> list[float]:
    return [overlap_C(domain, float(t), lam) for t in times]


def _token_posterior(zt: int, z0: int, alpha_s: float, alpha_t: float, mask_id: int) -> dict:
    """Equation 12 for one token, as a sparse distribution."""
    if zt != mask_id:
        return {zt: 1.0}
    denom = 1.0 - alpha_t
    if z0 == mask_id:
        return {mask_id: 1.0}
    p_mask = (1.0 - alpha_s) / denom
    p_clean = (alpha_s - alpha_t) / denom
    out = {}
    if p_mask > 0.0:
        out[mask_id] = p_mask
    if p_clean > 0.0:
        out[z0] = out.get(z0, 0.0) + p_clean
    return out


def _expand_posterior(xt, x0, alpha_s: float, alpha_t: float, mask_id: int) -> dict:
    pos = [
        _token_posterior(int(zt), int(z0), alpha_s, alpha_t, mask_id)
        for zt, z0 in zip(xt, x0)
    ]
    out = {}
    supports = [list(d.keys()) for d in pos]
    for choices in itertools.product(*supports):
        prob = 1.0
        for tok, dist in zip(choices, pos):
            prob *= dist[tok]
        out[choices] = out.get(choices, 0.0) + prob
    return out


def _denoiser(xt, t: float, tmax: float, lam: float, domain: ToyDomain) -> dict:
    """Equation 3: target posterior above tmax, mixture posterior at and below."""
    # t == tmax uses the mixture (equation 3). A tiny tolerance keeps a grid
    # point that is numerically tmax on the mixture side of the cutoff.
    clean = domain.p if t > tmax + 1e-12 else mix_clean(domain.p, domain.q, lam)
    alpha = alpha_ideal(t)
    likes = {}
    for x0, prior in clean.items():
        if prior <= 0.0:
            continue
        lik = sequence_forward_prob(xt, x0, alpha, domain.mask_id)
        if lik > 0.0:
            likes[x0] = lik * prior
    z = sum(likes.values())
    if z <= 0.0:
        return {}
    return {x0: value / z for x0, value in likes.items()}


def time_grid(tmax: float, n: int = 11) -> list[float]:
    """Increasing grid on [0, 1] that contains 0, tmax, and 1."""
    if n < 2:
        raise ValueError("n must be at least 2")
    raw = [i / (n - 1) for i in range(n)]
    times = []
    snapped = False
    for t in raw:
        if abs(t - tmax) <= 1e-12:
            times.append(float(tmax))
            snapped = True
        else:
            times.append(float(t))
    if not snapped:
        times.append(float(tmax))
    return sorted(set(times))


def enumerate_terminal(
    domain: ToyDomain,
    lam: float,
    tmax: float,
    times: list[float] | None = None,
) -> dict:
    """Exact terminal distribution of the masked RefineMix reverse chain.

    Starts at the all-mask sequence, which is pi when alpha(1) = 0.
    """
    if times is None:
        times = time_grid(tmax)
    if times[0] != 0.0 or times[-1] != 1.0:
        raise ValueError("time grid must start at 0 and end at 1")
    if not any(abs(t - tmax) <= 1e-12 for t in times):
        raise ValueError("tmax must lie on the time grid")
    x1 = tuple([domain.mask_id] * domain.length)
    mu = {x1: 1.0}
    for i in range(len(times) - 1, 0, -1):
        t = times[i]
        s = times[i - 1]
        a_t = alpha_ideal(t)
        a_s = alpha_ideal(s)
        new_mu: dict = {}
        for xt, pxt in mu.items():
            if pxt < 1e-18:
                continue
            denoiser = _denoiser(xt, t, tmax, lam, domain)
            for x0, px0 in denoiser.items():
                posterior = _expand_posterior(xt, x0, a_s, a_t, domain.mask_id)
                mass = pxt * px0
                for xs, pxs in posterior.items():
                    new_mu[xs] = new_mu.get(xs, 0.0) + mass * pxs
        mu = new_mu
    return mu


def source_probability(mu: dict, domain: ToyDomain) -> float:
    source = set(domain.source)
    return sum(prob for seq, prob in mu.items() if seq in source)


def theorem1_rhs(lam: float, c_value: float) -> float:
    """(lam / (1 - lam)) * C(t)."""
    return (lam / (1.0 - lam)) * c_value
