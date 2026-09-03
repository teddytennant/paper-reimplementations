"""Appendix A: signal-noise model for TSE-EMA against VLE-EMA.

The model (Eqs. 12 and 13) writes each observed per-epoch loss as a slow signal
plus zero-mean noise:

    y_t^val   = mu_t^val   + eps_t^val,    Var(eps_t^val)   = sigma_{t,val}^2
    y_t^train = mu_t^train + eps_t^train,  Var(eps_t^train) = sigma_{t,train}^2

Both estimators are the normalized EMA of Eqs. 14 and 17, so for weights
w_t = gamma^(T-t),

    E[S]   = sum_t w_t mu_t / sum_t w_t                       (Eqs. 15, 18)
    Var(S) = sum_t w_t^2 sigma_t^2 / (sum_t w_t)^2            (Eqs. 16, 19)

and the variance gap is Eq. 20,

    Var(S_TSE) - Var(S_VLE)
        = sum_t gamma^(2(T-t)) (sigma_{t,train}^2 - sigma_{t,val}^2)
          / (sum_t gamma^(T-t))^2.

Bias is taken against the quantity the appendix says is of interest, "final
generalization performance", i.e. mu_T^val.

The appendix reports a Monte Carlo check at M = 1000 replications giving bias
about -0.01 for VLE-EMA against -0.045 for TSE-EMA, and variance 0.0026 against
0.0041. It does not print the mu_t and sigma_t trajectories it used, so
:func:`paper_scenario` is a reconstruction: a validation curve that drops fast
and then drifts up as the network starts to overfit, a training curve that
keeps falling to a floor below the final validation loss, validation noise that
is large early and decays, and training minibatch noise that is nearly flat.
Those are the two conditions the appendix names, and with the paper's
gamma = 0.78 and T = 15 they land on the reported four numbers.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .estimators import GAMMA, CUTOFF_EPOCHS, ema_weights


@dataclasses.dataclass(frozen=True)
class SignalNoiseModel:
    """Trajectories mu_t and noise scales sigma_t for both loss signals."""

    mu_val: np.ndarray
    mu_train: np.ndarray
    sigma_val: np.ndarray
    sigma_train: np.ndarray
    gamma: float = GAMMA

    def __post_init__(self) -> None:
        n = self.mu_val.size
        for name in ("mu_train", "sigma_val", "sigma_train"):
            if getattr(self, name).size != n:
                raise ValueError(f"{name} must have length {n}")

    @property
    def n_epochs(self) -> int:
        return int(self.mu_val.size)

    @property
    def target(self) -> float:
        """mu_T^val, the final generalization loss the surrogate should track."""
        return float(self.mu_val[-1])


def paper_scenario(
    n_epochs: int = CUTOFF_EPOCHS, gamma: float = GAMMA
) -> SignalNoiseModel:
    """A concrete instance of Eqs. 12-13 matching the appendix's MC numbers.

    Validation loss falls from 1.32, bottoms out near epoch 7 and then drifts
    up; training loss falls monotonically to a floor 0.062 under the final
    validation loss, which is the "training loss underestimates generalization
    loss" gap the appendix appeals to. Validation noise starts at 0.295 and
    decays at 6% per epoch, training noise starts at 0.228 and decays at 2%, so
    late epochs, the ones the EMA actually weights, are quieter on validation.
    """
    k = np.arange(n_epochs, dtype=np.float64)
    mu_val = 0.9 * np.exp(-0.8 * k) + 0.42 + 0.00776 * k
    final_val = float(mu_val[-1])
    mu_train = 1.1 * np.exp(-0.8 * k) + (final_val - 0.062)
    sigma_val = 0.295 * np.exp(-0.06 * k)
    sigma_train = 0.228 * np.exp(-0.02 * k)
    return SignalNoiseModel(mu_val, mu_train, sigma_val, sigma_train, gamma)


def analytic_mean(mu: np.ndarray, gamma: float = GAMMA) -> float:
    """Eqs. 15 and 18."""
    w = ema_weights(mu.size, gamma)
    return float((w @ mu) / w.sum())


def analytic_variance(sigma: np.ndarray, gamma: float = GAMMA) -> float:
    """Eqs. 16 and 19."""
    w = ema_weights(sigma.size, gamma)
    return float(((w**2) @ (sigma**2)) / w.sum() ** 2)


def variance_gap(model: SignalNoiseModel) -> float:
    """Eq. 20: Var(S_TSE) - Var(S_VLE), positive when VLE-EMA is the calmer one."""
    w = ema_weights(model.n_epochs, model.gamma)
    num = (w**2) @ (model.sigma_train**2 - model.sigma_val**2)
    return float(num / w.sum() ** 2)


@dataclasses.dataclass(frozen=True)
class Moments:
    bias_vle: float
    bias_tse: float
    var_vle: float
    var_tse: float


def analytic_moments(model: SignalNoiseModel) -> Moments:
    """Closed-form bias against mu_T^val and variance for both estimators."""
    target = model.target
    return Moments(
        bias_vle=analytic_mean(model.mu_val, model.gamma) - target,
        bias_tse=analytic_mean(model.mu_train, model.gamma) - target,
        var_vle=analytic_variance(model.sigma_val, model.gamma),
        var_tse=analytic_variance(model.sigma_train, model.gamma),
    )


def monte_carlo_moments(
    model: SignalNoiseModel, n_replications: int = 1000, seed: int = 0
) -> Moments:
    """The appendix's Monte Carlo check, M = 1000 replications by default.

    Draws Gaussian noise around mu_t with scale sigma_t, forms both normalized
    EMAs per replication, and reports the sample bias and sample variance.
    Runs in float64; the numbers being checked are order 1e-3.
    """
    rng = np.random.default_rng(seed)
    T = model.n_epochs
    w = ema_weights(T, model.gamma, normalize=True)
    eps_val = rng.standard_normal((n_replications, T)) * model.sigma_val
    eps_train = rng.standard_normal((n_replications, T)) * model.sigma_train
    s_vle = (model.mu_val + eps_val) @ w
    s_tse = (model.mu_train + eps_train) @ w
    target = model.target
    return Moments(
        bias_vle=float(s_vle.mean() - target),
        bias_tse=float(s_tse.mean() - target),
        var_vle=float(s_vle.var(ddof=1)),
        var_tse=float(s_tse.var(ddof=1)),
    )
