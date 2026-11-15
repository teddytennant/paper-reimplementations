"""Movement-aware aggregation: Fixed-Share, Combine, DM-multiscale."""

from __future__ import annotations

import math

import numpy as np


def _gate_unconstrained(x: float, tau: float, Z: float) -> float:
    """Solution of 8 g' = (x/tau) g + Z with g(0) = 0, from alg:two-way-combiner."""
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    y = x / (4.0 * math.sqrt(tau))
    # g(x) = (Z/4) sqrt(pi tau) exp(y^2) erf(y), y = x / (4 sqrt(tau)).
    if y > 0.0 and y * y > 40.0:
        return math.inf
    if y < 0.0 and y * y > 40.0:
        return -math.inf
    pref = (Z / 4.0) * math.sqrt(math.pi * tau)
    try:
        return pref * math.exp(y * y) * math.erf(y)
    except OverflowError:
        return math.inf if x > 0.0 else -math.inf


def gate(x: float, tau: float, Z: float) -> float:
    """Pi_[0,1](g_{tau,Z}(x)) from alg:two-way-combiner."""
    g = _gate_unconstrained(x, tau, Z)
    if g <= 0.0:
        return 0.0
    if g >= 1.0:
        return 1.0
    return g


def solve_U(tau: float, Z: float) -> float:
    """U_{tau,Z} = g^{-1}(1), the upper edge of the state projection."""
    lo = 0.0
    hi = 1.0
    while _gate_unconstrained(hi, tau, Z) < 1.0:
        hi *= 2.0
        if hi > 1e7:
            return hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _gate_unconstrained(mid, tau, Z) < 1.0:
            lo = mid
        else:
            hi = mid
    return hi


class FixedShare:
    """alg:fs-scale. log is the natural logarithm, as in the MW derivation."""

    def __init__(self, n: int, tau: float, D: float, eta: float | None = None, weights=None):
        if n < 1:
            raise ValueError("n must be positive")
        if tau <= 0.0 or D <= 0.0:
            raise ValueError("tau and D must be positive")
        self.n = int(n)
        self.tau = float(tau)
        self.D = float(D)
        if eta is None:
            self.eta = math.sqrt(math.log(self.n * self.tau) / (self.D * self.tau))
        else:
            self.eta = float(eta)
        if weights is None:
            self.z = np.full(self.n, 1.0 / self.n, dtype=np.float64)
        else:
            w = np.asarray(weights, dtype=np.float64)
            if w.shape != (self.n,):
                raise ValueError("weights must have shape (n,)")
            self.z = w / w.sum()
        # Freeze when the scale is too short for the stated learning rate.
        self.active = self.tau >= 16.0 * self.D * math.log(self.n * self.tau)

    def weights(self) -> np.ndarray:
        return self.z.copy()

    def update(self, ell) -> np.ndarray:
        loss = np.asarray(ell, dtype=np.float64)
        if loss.shape != (self.n,):
            raise ValueError("ell must have shape (n,)")
        if not self.active:
            return self.z.copy()
        share = 1.0 / (self.n * self.tau)
        num = self.z * np.exp(-self.eta * loss) + share
        self.z = num / np.sum(num)
        return self.z.copy()


class Combine:
    """alg:two-way-combiner. P is the coarser aggregate, Q the finer learner."""

    def __init__(self, P, Q, tau: float, Z: float, D: float):
        if tau <= 0.0 or D <= 0.0 or Z <= 0.0:
            raise ValueError("tau, Z, and D must be positive")
        self.P = P
        self.Q = Q
        self.tau = float(tau)
        self.Z = float(Z)
        self.D = float(D)
        self.x = 0.0
        self.U = solve_U(self.tau, self.Z)
        self.active = self.tau >= 64.0 * self.D * math.log(1.0 / self.Z)

    def weights(self) -> np.ndarray:
        p = self.P.weights()
        q = self.Q.weights()
        if not self.active:
            return p
        alpha = gate(self.x, self.tau, self.Z)
        return alpha * p + (1.0 - alpha) * q

    def update(self, ell) -> np.ndarray:
        loss = np.asarray(ell, dtype=np.float64)
        p = self.P.weights()
        q = self.Q.weights()
        p_next = self.P.update(loss)
        q_next = self.Q.update(loss)
        # ||.||_TV in alg:two-way-combiner is the paper's TV distance, 0.5 * L1.
        tv_p = 0.5 * float(np.sum(np.abs(p_next - p)))
        tv_q = 0.5 * float(np.sum(np.abs(q_next - q)))
        loss_p = (float(np.dot(loss, p)) + self.D * tv_p) / 3.0
        loss_q = (float(np.dot(loss, q)) + self.D * tv_q) / 3.0
        b = (loss_p - loss_q) / math.sqrt(self.D)
        x_new = (1.0 - 1.0 / self.tau) * self.x + b
        self.x = min(self.U + 2.0, max(-2.0, x_new))
        return self.weights()


class DMMultiscale:
    """alg:dm-multiscale.

    The tex sets T = 2^J. If T is not a power of two, the scale schedule uses
    the next power of two and the learner is still updated for the true horizon.
    D is the value passed by the caller (alg:master uses lambda/M). The tex
    states the construction for D >= 1 and only cites a reduction for D < 1.
    """

    def __init__(self, T: int, n_experts: int, D: float):
        if T < 2:
            raise ValueError("alg:dm-multiscale needs T >= 2")
        if D <= 0.0:
            raise ValueError("D must be positive")
        self.T = int(T)
        self.D = float(D)
        J = int(math.ceil(math.log2(self.T)))
        self.T_dm = 2 ** J
        self.Z = 1.0 / (2.0 * self.T_dm * math.log(self.T_dm))
        self.scale_learners = []
        for u in range(J):
            tau = self.T_dm / (2 ** u)
            self.scale_learners.append(FixedShare(n_experts, tau, self.D))
        root = self.scale_learners[0]
        for u in range(1, J):
            tau = self.T_dm / (2 ** u)
            root = Combine(root, self.scale_learners[u], tau, self.Z, self.D)
        self.root = root

    def weights(self) -> np.ndarray:
        return self.root.weights()

    def update(self, ell) -> np.ndarray:
        return self.root.update(ell)
