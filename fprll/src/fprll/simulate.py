"""Restarted FPRLL experts and the density-mixture master."""

from __future__ import annotations

import numpy as np

from fprll.aggregate import DMMultiscale
from fprll.core import (
    GRID_N,
    density,
    grid_mean,
    indicator_cost,
    lazy_sample,
    make_grid,
    minimizer,
    sample_index,
    total_variation,
)


def dyadic_scales(T: int) -> list[int]:
    """H in {1, 2, 4, ..., 2^{floor(log2 T)}}."""
    if T < 1:
        raise ValueError("T must be positive")
    scales = []
    H = 1
    while H <= T:
        scales.append(H)
        H *= 2
    return scales


class FPRLL:
    """One follow-the-perturbed-regularized-leader state on [-1, 1]^d.

    Unperturbed potential after the losses observed so far:
    F(x) = <cum_grad, x> + (1/gamma) r(x),
    r(x) = (sigma/2) ||x||^2 - sum_i log(1 - x_i^2).
    A fresh action is the minimizer of F(x) - <Z, x> with Z ~ Lap(mu),
    mu = laplace_scale, independent coordinates. The density of Z is
    (2*mu)^{-d} exp(-||Z||_1 / mu).
    """

    def __init__(
        self,
        gamma: float,
        sigma: float = 1.0,
        d: int = 1,
        n_grid: int = GRID_N,
        laplace_scale: float = 1.0,
    ):
        if gamma <= 0.0:
            raise ValueError("gamma must be positive")
        if sigma < 0.0:
            raise ValueError("sigma must be nonnegative")
        if d < 1:
            raise ValueError("d must be positive")
        mu = float(laplace_scale)
        if not np.isfinite(mu) or mu <= 0.0:
            raise ValueError("laplace_scale must be positive")
        self.gamma = float(gamma)
        self.sigma = float(sigma)
        self.d = int(d)
        self.n_grid = int(n_grid)
        self.laplace_scale = mu
        self.cum_grad = np.zeros(self.d, dtype=np.float64)
        self.centers, self.dx = make_grid(self.n_grid)
        self.prev_density = None

    def observe(self, g) -> None:
        self.cum_grad = self.cum_grad + np.asarray(g, dtype=np.float64).reshape(self.d)

    def reset(self) -> None:
        self.cum_grad = np.zeros(self.d, dtype=np.float64)

    def density_grid(self) -> np.ndarray:
        if self.d != 1:
            raise ValueError("grid density is implemented for d=1")
        return density(
            self.centers,
            float(self.cum_grad[0]),
            self.gamma,
            self.sigma,
            laplace_scale=self.laplace_scale,
        )

    def mean_action(self, n_samples: int = 2000, seed: int = 0, laplace_scale: float = 1.0):
        """Monte Carlo mean of the perturbed-leader minimizer.

        Draws i.i.d. Laplace perturbations with scale laplace_scale (the
        paper's mu). Default laplace_scale is 1. Pass this expert's
        laplace_scale when it is not 1.
        """
        mu = float(laplace_scale)
        if not np.isfinite(mu) or mu <= 0.0:
            raise ValueError("laplace_scale must be positive")
        rng = np.random.default_rng(seed)
        Z = rng.laplace(0.0, mu, size=(int(n_samples), self.d))
        xs = minimizer(self.cum_grad, Z, self.gamma, self.sigma)
        mean = np.mean(xs, axis=0)
        if self.d == 1:
            return float(mean[0])
        return mean

    def score(self, g, lam: float):
        """Surrogate E_Q[f] + lam * TV(Q, Q_prev), with c_1 = 0.

        Uses the current density (losses observed before this round). Stores
        that density for the next TV. Does not fold g into cum_grad.
        """
        if self.d != 1:
            raise ValueError("grid surrogate is implemented for d=1")
        q = np.asarray(self.density_grid(), dtype=np.float64)
        mean = grid_mean(q, self.centers, self.dx)
        loss = float(np.asarray(g, dtype=np.float64).reshape(-1)[0] * mean)
        if self.prev_density is None:
            tv = 0.0
        else:
            tv = total_variation(q, self.prev_density, self.dx)
        self.prev_density = q
        return loss + float(lam) * tv, q


class RestartedFPRLL(FPRLL):
    """Fresh FPRLL copy every block of length H.

    After H observations the cumulative gradient is cleared, so the next
    density is the prior. A standalone rollout draws a new action at that
    boundary instead of coupling to the previous copy.
    """

    def __init__(
        self,
        horizon: int,
        gamma: float,
        sigma: float = 1.0,
        d: int = 1,
        n_grid: int = GRID_N,
        laplace_scale: float = 1.0,
    ):
        if horizon < 1:
            raise ValueError("horizon must be positive")
        super().__init__(gamma, sigma, d, n_grid, laplace_scale=laplace_scale)
        self.H = int(horizon)
        self.seen = 0

    def observe(self, g) -> None:
        self.seen += 1
        if self.seen >= self.H:
            self.reset()
            self.seen = 0
        else:
            super().observe(g)


def _shift_losses(surrogates, grad_bound: float, lam: float):
    """Map signed surrogates into [0, 1] as ell = g/M.

    Linear losses on the box lie in [-B, B] with B = ||g||_1. A surrogate
    E[f] + lam * TV then lies in [-B, B + lam]. alg:master assumes a
    nonnegative loss and sets ell = g/M in [0, 1]. Shifting by B and taking
    M = 2B + lam is that reduction for signed losses. D = lam/M.
    """
    B = float(grad_bound)
    M = 2.0 * B + float(lam)
    if M <= 0.0:
        raise ValueError("M must be positive")
    shifted = np.asarray(surrogates, dtype=np.float64) + B
    ell = np.clip(shifted / M, 0.0, 1.0)
    D = float(lam) / M
    return ell, M, D


def rollout_scale(
    gradients,
    horizon: int,
    gamma: float,
    sigma: float = 1.0,
    seed: int = 0,
    n_grid: int = GRID_N,
    fresh_at_restart: bool = True,
    laplace_scale: float = 1.0,
):
    """Play one restarted expert and count grid-binned switches.

    Inside a block the action follows alg:lazy-sample. At the first round of
    each later block, a fresh copy draws a new grid sample (main-alg line 1
    of the new copy) when fresh_at_restart is true.
    """
    g_seq = np.asarray(gradients, dtype=np.float64).reshape(-1)
    T = int(g_seq.shape[0])
    expert = RestartedFPRLL(
        horizon, gamma, sigma, d=1, n_grid=n_grid, laplace_scale=laplace_scale
    )
    rng = np.random.default_rng(seed)
    q = expert.density_grid()
    idx = sample_index(q, rng)
    switches = 0
    loss = 0.0
    actions = [expert.centers[idx]]
    for t in range(T):
        loss += float(g_seq[t] * expert.centers[idx])
        q_now = expert.density_grid()
        expert.observe(g_seq[t])
        if t == T - 1:
            break
        q_next = expert.density_grid()
        block_start = ((t + 1) % expert.H) == 0
        if fresh_at_restart and block_start:
            new_idx = sample_index(q_next, rng)
        else:
            new_idx, _kept = lazy_sample(idx, q_now, q_next, rng)
        if indicator_cost(expert.centers[idx], expert.centers[new_idx], expert.n_grid):
            switches += 1
        idx = new_idx
        actions.append(expert.centers[idx])
    return {
        "switches": switches,
        "loss": loss,
        "actions": np.asarray(actions, dtype=np.float64),
        "horizon": int(horizon),
    }


def run(
    gradients,
    *,
    gamma: float,
    sigma: float = 1.0,
    lam: float = 1.0,
    comparators=None,
    seed: int = 0,
    n_grid: int = GRID_N,
    grad_bound: float | None = None,
    fresh_at_restart: bool = True,
    laplace_scale: float = 1.0,
):
    """alg:master on d=1 linear losses, with one restarted expert per dyadic H.

    The master is alg:dm-multiscale. Switching cost passed in is lam/M.
    Each scale is also rolled out on its own so its indicator-augmented cost
    can be compared with the mixture.
    """
    if lam <= 0.0:
        raise ValueError("lam must be positive")
    g_seq = np.asarray(gradients, dtype=np.float64)
    if g_seq.ndim != 1:
        raise ValueError("run implements the grid master for d=1 gradients of shape (T,)")
    T = int(g_seq.shape[0])
    if T < 2:
        raise ValueError("T must be at least 2")
    scales = dyadic_scales(T)
    experts = [
        RestartedFPRLL(H, gamma, sigma, d=1, n_grid=n_grid, laplace_scale=laplace_scale)
        for H in scales
    ]
    centers = experts[0].centers
    n_bins = experts[0].n_grid
    B = float(np.max(np.abs(g_seq)) if grad_bound is None else grad_bound)
    # D is fixed from the bound; the per-round shift uses the same B.
    _ell0, M, D = _shift_losses(np.zeros(len(scales)), B, lam)
    master = DMMultiscale(T, len(scales), D)
    rng = np.random.default_rng(seed)
    scale_rngs = [np.random.default_rng(seed + 1000 + i) for i in range(len(scales))]

    v = master.weights()
    qs = [np.asarray(e.density_grid(), dtype=np.float64) for e in experts]
    P = _mix(v, qs)
    idx = sample_index(P, rng)
    scale_idx = [sample_index(qs[i], scale_rngs[i]) for i in range(len(scales))]

    mix_loss = 0.0
    mix_switches = 0
    scale_loss = np.zeros(len(scales), dtype=np.float64)
    scale_switches = np.zeros(len(scales), dtype=np.float64)
    comp_loss = 0.0
    weight_rows = []
    u_seq = None if comparators is None else np.asarray(comparators, dtype=np.float64).reshape(-1)
    if u_seq is not None and u_seq.shape[0] != T:
        raise ValueError("comparators must have length T")

    for t in range(T):
        g_t = float(g_seq[t])
        x = float(centers[idx])
        mix_loss += g_t * x
        for i in range(len(scales)):
            scale_loss[i] += g_t * float(centers[scale_idx[i]])
        if u_seq is not None:
            comp_loss += g_t * float(u_seq[t])

        surrogates = []
        qs = []
        for e in experts:
            s, q = e.score(g_t, lam)
            surrogates.append(s)
            qs.append(q)
        ell, _M, _D = _shift_losses(surrogates, B, lam)
        for e in experts:
            e.observe(g_t)
        v_next = master.update(ell)
        weight_rows.append(v_next.copy())
        qs_next = [np.asarray(e.density_grid(), dtype=np.float64) for e in experts]

        if t == T - 1:
            v = v_next
            break

        P = _mix(v, qs)
        P_next = _mix(v_next, qs_next)
        new_idx, _kept = lazy_sample(idx, P, P_next, rng)
        if indicator_cost(centers[idx], centers[new_idx], n_bins):
            mix_switches += 1
        idx = new_idx

        for i, H in enumerate(scales):
            block_start = ((t + 1) % H) == 0
            if fresh_at_restart and block_start:
                new_s = sample_index(qs_next[i], scale_rngs[i])
            else:
                new_s, _kept_s = lazy_sample(scale_idx[i], qs[i], qs_next[i], scale_rngs[i])
            if indicator_cost(centers[scale_idx[i]], centers[new_s], n_bins):
                scale_switches[i] += 1
            scale_idx[i] = new_s
        v = v_next

    mix_cost = mix_loss + float(lam) * mix_switches
    scale_cost = scale_loss + float(lam) * scale_switches
    return {
        "scales": scales,
        "mixture_loss": mix_loss,
        "mixture_switches": mix_switches,
        "mixture_cost": mix_cost,
        "mixture_regret": mix_cost - comp_loss,
        "scale_loss": scale_loss,
        "scale_switches": scale_switches,
        "scale_cost": scale_cost,
        "scale_regret": scale_cost - comp_loss,
        "comparator_loss": comp_loss,
        "weights": np.vstack(weight_rows),
        "M": M,
        "D": D,
        "grad_bound": B,
    }


def _mix(weights, densities) -> np.ndarray:
    v = np.asarray(weights, dtype=np.float64)
    total = float(v.sum())
    if total <= 0.0:
        v = np.full(len(densities), 1.0 / len(densities))
    else:
        v = v / total
    acc = np.zeros_like(densities[0], dtype=np.float64)
    for w, q in zip(v, densities):
        acc += float(w) * np.asarray(q, dtype=np.float64)
    return acc
