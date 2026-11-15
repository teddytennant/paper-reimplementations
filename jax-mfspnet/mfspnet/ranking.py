"""Rank correlation and the proxy-ranking comparison of Sec. 6.4 / Appendix E.

Table 6 scores each proxy by Spearman's rho and Kendall's tau against the final
validation accuracy of 50 sampled architectures:

    VLE-EMA                     rho 0.87   tau 0.71
    TSE-EMA                     rho 0.63   tau 0.49
    low-epoch (10-epoch)        rho 0.58   tau 0.42

Spearman and Kendall are implemented here so the package does not lean on
scipy for its own results; the tests cross-check both against scipy.

:func:`synthetic_curves` generates learning curves whose final accuracy is
known by construction. It encodes the paper's stated premise rather than
proving it: validation loss late in training tracks final generalization,
training loss tracks optimization speed and only partly tracks generalization,
and single-epoch readings are noisy. Given that premise the ordering of the
three proxies in Table 6 should fall out, and that is what the test checks.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .estimators import GAMMA, CUTOFF_EPOCHS, low_epoch_proxy, tse_ema, vle_ema


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared, 1-based."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(x.size, dtype=np.float64)
    sx = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(a, b) -> float:
    """Pearson correlation of the average ranks."""
    ra, rb = _rankdata(a), _rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra @ ra) * (rb @ rb))
    if denom == 0.0:
        return 0.0
    return float((ra @ rb) / denom)


def kendall_tau(a, b) -> float:
    """Kendall's tau-b, so ties in either argument are handled."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = a.size
    da = a[:, None] - a[None, :]
    db = b[:, None] - b[None, :]
    iu = np.triu_indices(n, 1)
    sa, sb = np.sign(da[iu]), np.sign(db[iu])
    concordant = float((sa * sb).sum())
    n0 = 0.5 * n * (n - 1)
    ties_a = float((sa == 0).sum())
    ties_b = float((sb == 0).sum())
    denom = np.sqrt((n0 - ties_a) * (n0 - ties_b))
    if denom == 0.0:
        return 0.0
    return concordant / denom


@dataclasses.dataclass(frozen=True)
class CurveSet:
    """Sampled architectures with their curves and their known final quality."""

    val_losses: np.ndarray  # (n_arch, T)
    train_losses: np.ndarray  # (n_arch, T)
    final_accuracy: np.ndarray  # (n_arch,)


def synthetic_curves(
    n_arch: int = 50,
    n_epochs: int = CUTOFF_EPOCHS,
    seed: int = 0,
    train_noise: float = 0.08,
    val_noise: float = 0.18,
    optimization_speed_spread: float = 0.50,
) -> CurveSet:
    """Learning curves for ``n_arch`` architectures with known final accuracy.

    Each architecture gets a latent generalization quality ``q`` and an
    independent optimization speed ``s``. Final accuracy is a monotone function
    of ``q`` alone. The validation curve converges toward a floor set by ``q``;
    the training curve converges toward a floor set by ``q`` and ``s`` together,
    so training loss confounds "learns fast" with "generalizes well". That is
    the confound Sec. 4.3 raises against TSE-EMA. Per-epoch validation noise is
    inflated early and settles down, as in the Appendix A model, which is what
    penalizes a single-epoch reading relative to the weighted average.

    ``n_arch = 50`` and ``n_epochs = 15`` are the paper's Sec. 6.4 protocol. The
    three constants above are not in the paper; they set how strong the speed
    confound and the per-epoch noise are, and they were picked so the three
    proxies land near their Table 6 values.
    """
    rng = np.random.default_rng(seed)
    k = np.arange(n_epochs, dtype=np.float64)

    quality = rng.uniform(0.0, 1.0, size=n_arch)
    speed = rng.uniform(0.0, 1.0, size=n_arch)

    val_floor = 0.95 - 0.45 * quality
    train_floor = val_floor - 0.10 - optimization_speed_spread * speed
    decay = 0.35 + 0.45 * speed

    val_signal = val_floor[:, None] + 1.3 * np.exp(-decay[:, None] * k[None, :])
    train_signal = train_floor[:, None] + 1.5 * np.exp(-decay[:, None] * k[None, :])

    val_scale = val_noise * (1.0 + 2.0 * np.exp(-0.35 * k))
    train_scale = train_noise * np.ones_like(k)

    val_losses = val_signal + rng.standard_normal((n_arch, n_epochs)) * val_scale
    train_losses = train_signal + rng.standard_normal((n_arch, n_epochs)) * train_scale

    final_accuracy = 0.55 + 0.40 * quality
    return CurveSet(val_losses, train_losses, final_accuracy)


def proxy_scores(curves: CurveSet, gamma: float = GAMMA, low_epoch: int = 10):
    """Score every architecture with each of the three Table 6 proxies.

    All three are losses, so they are negated before correlating against
    accuracy; a good proxy gives a low value to a good architecture.
    """
    return {
        "vle_ema": np.array([vle_ema(v, gamma) for v in curves.val_losses]),
        "tse_ema": np.array([tse_ema(t, gamma) for t in curves.train_losses]),
        "low_epoch": np.array(
            [low_epoch_proxy(v, low_epoch) for v in curves.val_losses]
        ),
    }


def rank_correlations(curves: CurveSet, gamma: float = GAMMA, low_epoch: int = 10):
    """Table 6: Spearman rho and Kendall tau per proxy against final accuracy."""
    scores = proxy_scores(curves, gamma, low_epoch)
    acc = curves.final_accuracy
    return {
        name: {
            "spearman": spearman(-s, acc),
            "kendall": kendall_tau(-s, acc),
        }
        for name, s in scores.items()
    }
