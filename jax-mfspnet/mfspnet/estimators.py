"""TSE, TSE-EMA and VLE-EMA (paper Sec. 4.3, Eqs. 6, 7 and 8).

Eq. 6, the training speed estimator of Ru et al.:

    TSE = sum_{t=1..T} (1/B) sum_{i=1..B} loss(f(x_i), y_i)

Eq. 7, its exponential moving average variant:

    TSE-EMA = sum_{t=1..T} gamma^(T-t) * mean-batch training loss at epoch t

Eq. 8, the proposed estimator, which is Eq. 7 with the training loss swapped
for the validation loss (the substitution l -> l_v spelled out in Sec. 4.3):

    VLE-EMA = sum_{t=1..T} gamma^(T-t) * mean-batch validation loss at epoch t

gamma = 0.78 (Table 2), validation cut-off T = 15 epochs (Table 2).

One wrinkle worth naming. Eqs. 7 and 8 in the body are unnormalized sums, while
Eqs. 14 and 17 in Appendix A divide by sum_t gamma^(T-t) and call the same
objects S_VLE and S_TSE. The two differ by a constant factor for fixed T, so
they rank architectures identically, but the Appendix A bias and variance
expressions only hold for the normalized form. ``normalize`` picks which one
you get; the search uses the body form and Appendix A uses the normalized one.
"""

from __future__ import annotations

import numpy as np

GAMMA = 0.78
CUTOFF_EPOCHS = 15


def ema_weights(n_epochs: int, gamma: float = GAMMA, normalize: bool = False) -> np.ndarray:
    """The weights gamma^(T-t) for t = 1..T, newest epoch weighted 1."""
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    t = np.arange(1, n_epochs + 1)
    w = gamma ** (n_epochs - t)
    if normalize:
        w = w / w.sum()
    return w


def _epoch_means(losses: np.ndarray) -> np.ndarray:
    """Collapse per-minibatch losses to the inner mean of Eqs. 6-8.

    Accepts either (T,) epoch means already, or (T, B) per-minibatch losses.
    """
    losses = np.asarray(losses, dtype=np.float64)
    if losses.ndim == 1:
        return losses
    if losses.ndim == 2:
        return losses.mean(axis=1)
    raise ValueError("losses must be (T,) or (T, B)")


def tse(train_losses) -> float:
    """Eq. 6. Uniform weights over the T observed epochs."""
    return float(_epoch_means(train_losses).sum())


def tse_ema(train_losses, gamma: float = GAMMA, normalize: bool = False) -> float:
    """Eq. 7 (``normalize=False``) or Eq. 17 (``normalize=True``)."""
    m = _epoch_means(train_losses)
    return float(ema_weights(m.size, gamma, normalize) @ m)


def vle_ema(val_losses, gamma: float = GAMMA, normalize: bool = False) -> float:
    """Eq. 8 (``normalize=False``) or Eq. 14 (``normalize=True``).

    Same weighting as Eq. 7, applied to validation losses. Lower is better, so
    this is used directly as the PSO fitness in Alg. 3, line 10.
    """
    m = _epoch_means(val_losses)
    return float(ema_weights(m.size, gamma, normalize) @ m)


def low_epoch_proxy(val_losses, epoch: int = 10) -> float:
    """The "low-epoch (10-epoch) training" baseline of Table 6.

    A single loss reading at ``epoch``, with no averaging over the trajectory.
    Table 6 labels the row "training" but Appendix E's text calls the same
    baseline "low-epoch validation", so this reads the validation trajectory:
    train for 10 epochs, look at validation once. ``epoch`` is 1-based to match
    the paper's t index.
    """
    m = _epoch_means(val_losses)
    if not 1 <= epoch <= m.size:
        raise ValueError(f"epoch {epoch} outside 1..{m.size}")
    return float(m[epoch - 1])
