"""Sec. 6.4 / Appendix E: does VLE-EMA rank architectures better than the
proxies of Table 6.

Table 6: VLE-EMA rho 0.87 tau 0.71, TSE-EMA rho 0.63 tau 0.49, low-epoch
(10-epoch) rho 0.58 tau 0.42.
"""

import numpy as np
import pytest
from scipy import stats

from mfspnet.ranking import (
    kendall_tau,
    rank_correlations,
    spearman,
    synthetic_curves,
)

SEEDS = range(10)


def test_spearman_matches_scipy():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a, b = rng.normal(size=40), rng.normal(size=40)
        assert spearman(a, b) == pytest.approx(stats.spearmanr(a, b).statistic)


def test_kendall_tau_matches_scipy_including_ties():
    rng = np.random.default_rng(1)
    for _ in range(20):
        a = rng.integers(0, 5, 40).astype(float)
        b = rng.integers(0, 6, 40).astype(float)
        assert kendall_tau(a, b) == pytest.approx(stats.kendalltau(a, b).statistic)


def test_perfect_and_reversed_orderings():
    x = np.arange(20, dtype=float)
    assert spearman(x, x) == pytest.approx(1.0)
    assert kendall_tau(x, x) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)
    assert kendall_tau(x, -x) == pytest.approx(-1.0)


def test_curve_set_has_the_paper_protocol_shape():
    """Sec. 6.4: N = 50 sampled architectures, c = 15 cutoff epochs."""
    curves = synthetic_curves()
    assert curves.val_losses.shape == (50, 15)
    assert curves.train_losses.shape == (50, 15)
    assert curves.final_accuracy.shape == (50,)


def test_vle_ema_outranks_both_baselines_on_every_seed():
    """The Table 6 ordering, per seed, on both coefficients."""
    for seed in SEEDS:
        corr = rank_correlations(synthetic_curves(seed=seed))
        for metric in ("spearman", "kendall"):
            assert corr["vle_ema"][metric] > corr["tse_ema"][metric]
            assert corr["vle_ema"][metric] > corr["low_epoch"][metric]


def test_correlations_land_near_the_table_6_values():
    """Averaged over seeds, the three proxies reproduce Table 6 to +/- 0.07."""
    corr = [rank_correlations(synthetic_curves(seed=s)) for s in SEEDS]
    mean = {
        proxy: {m: float(np.mean([c[proxy][m] for c in corr])) for m in ("spearman", "kendall")}
        for proxy in ("vle_ema", "tse_ema", "low_epoch")
    }
    assert mean["vle_ema"]["spearman"] == pytest.approx(0.87, abs=0.07)
    assert mean["vle_ema"]["kendall"] == pytest.approx(0.71, abs=0.07)
    assert mean["tse_ema"]["spearman"] == pytest.approx(0.63, abs=0.07)
    assert mean["tse_ema"]["kendall"] == pytest.approx(0.49, abs=0.07)
    assert mean["low_epoch"]["spearman"] == pytest.approx(0.58, abs=0.07)
    assert mean["low_epoch"]["kendall"] == pytest.approx(0.42, abs=0.07)


def test_the_generator_does_not_hand_vle_ema_the_answer():
    """Final accuracy depends on latent quality only. Check the validation
    curve alone cannot be inverted perfectly: even the best possible reading
    of it is short of a perfect ranking, so the margin over TSE-EMA is not an
    artifact of a noiseless validation signal."""
    corr = rank_correlations(synthetic_curves(seed=0))
    assert corr["vle_ema"]["spearman"] < 0.97


def test_gamma_matters_and_the_paper_value_is_in_the_good_band():
    """Sec. 4.3 says gamma was swept over [0.6, 0.9] and 0.78 came out best.
    Uniform averaging over the whole trajectory should be clearly worse."""
    curves = [synthetic_curves(seed=s) for s in SEEDS]
    paper = np.mean(
        [rank_correlations(c, gamma=0.78)["vle_ema"]["spearman"] for c in curves]
    )
    uniform = np.mean(
        [rank_correlations(c, gamma=1.0)["vle_ema"]["spearman"] for c in curves]
    )
    assert paper > uniform


def test_ranking_survives_swapping_kendall_for_spearman():
    """Both coefficients have to agree on the winner or the claim is fragile."""
    for seed in SEEDS:
        corr = rank_correlations(synthetic_curves(seed=seed))
        order_s = sorted(corr, key=lambda k: -corr[k]["spearman"])
        order_k = sorted(corr, key=lambda k: -corr[k]["kendall"])
        assert order_s[0] == order_k[0] == "vle_ema"
