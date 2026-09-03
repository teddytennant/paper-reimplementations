"""TSE, TSE-EMA, VLE-EMA (paper Eqs. 6, 7, 8) and the Table 2 constants."""

import numpy as np
import pytest

from mfspnet.estimators import (
    CUTOFF_EPOCHS,
    GAMMA,
    ema_weights,
    low_epoch_proxy,
    tse,
    tse_ema,
    vle_ema,
)


def test_table_2_constants():
    assert GAMMA == 0.78
    assert CUTOFF_EPOCHS == 15


def test_tse_is_the_plain_sum_over_epochs():
    """Eq. 6 with per-minibatch losses: mean inside, sum outside."""
    losses = np.array([[1.0, 3.0], [0.5, 0.5], [0.2, 0.4]])
    assert tse(losses) == pytest.approx(2.0 + 0.5 + 0.3)


def test_ema_weights_put_gamma_to_the_zero_on_the_last_epoch():
    """Eq. 7: weight gamma^(T-t), so t = T gets 1 and t = 1 gets gamma^(T-1)."""
    w = ema_weights(5, 0.5)
    assert w.tolist() == [0.0625, 0.125, 0.25, 0.5, 1.0]
    assert ema_weights(5, 0.5, normalize=True).sum() == pytest.approx(1.0)


def test_tse_ema_equals_the_explicit_sum():
    losses = np.array([1.0, 0.8, 0.6, 0.5])
    expected = sum(GAMMA ** (4 - t) * losses[t - 1] for t in range(1, 5))
    assert tse_ema(losses) == pytest.approx(expected)


def test_vle_ema_is_tse_ema_with_the_loss_signal_swapped():
    """Sec. 4.3 states the only change is l -> l_v, so on identical input the
    two functions must agree exactly."""
    x = np.array([1.2, 0.9, 0.7, 0.55, 0.5])
    assert vle_ema(x) == pytest.approx(tse_ema(x))


def test_vle_ema_weights_late_epochs_more():
    early_drop = np.array([2.0, 0.4, 0.4, 0.4, 0.4])
    late_drop = np.array([0.4, 0.4, 0.4, 0.4, 2.0])
    assert vle_ema(early_drop) < vle_ema(late_drop)
    # while TSE, which is unweighted, cannot tell them apart
    assert tse(early_drop) == pytest.approx(tse(late_drop))


def test_normalization_is_a_positive_rescale_so_ranking_is_unchanged():
    """Eqs. 7/8 are unnormalized, Eqs. 14/17 are normalized. Same order."""
    rng = np.random.default_rng(0)
    curves = rng.uniform(0.2, 2.0, size=(30, CUTOFF_EPOCHS))
    raw = np.array([vle_ema(c) for c in curves])
    norm = np.array([vle_ema(c, normalize=True) for c in curves])
    assert np.array_equal(np.argsort(raw), np.argsort(norm))
    ratio = raw / norm
    assert np.allclose(ratio, ratio[0])


def test_gamma_one_makes_the_ema_the_plain_sum():
    losses = np.array([1.0, 0.8, 0.6])
    assert tse_ema(losses, gamma=1.0) == pytest.approx(tse(losses))


def test_minibatch_and_epoch_mean_input_agree():
    per_batch = np.array([[1.0, 3.0], [0.5, 0.5], [0.2, 0.4]])
    assert vle_ema(per_batch) == pytest.approx(vle_ema(per_batch.mean(axis=1)))


def test_low_epoch_proxy_is_one_reading():
    losses = np.arange(15, dtype=float)
    assert low_epoch_proxy(losses, 10) == 9.0
    with pytest.raises(ValueError):
        low_epoch_proxy(losses, 16)


def test_gamma_out_of_range_is_rejected():
    with pytest.raises(ValueError):
        ema_weights(5, 0.0)
    with pytest.raises(ValueError):
        ema_weights(5, 1.5)
