import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import entropy

from cadmp.drift import drift_profile, kl_divergence, select_mask_layer, smooth


def _dists(rng, layers, tokens):
    x = rng.random((layers, tokens)) + 0.05
    return jnp.asarray(x / x.sum(axis=-1, keepdims=True))


def test_kl_matches_scipy_entropy():
    rng = np.random.default_rng(0)
    for _ in range(20):
        p, q = _dists(rng, 2, 40)
        got = float(kl_divergence(p, q, eps=0.0))
        assert got == pytest.approx(float(entropy(np.asarray(p), np.asarray(q))), rel=1e-12)


def test_kl_of_a_distribution_with_itself_is_zero():
    rng = np.random.default_rng(1)
    p = _dists(rng, 1, 64)[0]
    assert float(kl_divergence(p, p, eps=0.0)) == pytest.approx(0.0, abs=1e-15)
    assert float(kl_divergence(p, p)) == pytest.approx(0.0, abs=1e-12)


def test_kl_is_asymmetric():
    p = jnp.asarray([0.7, 0.2, 0.1])
    q = jnp.asarray([0.2, 0.3, 0.5])
    forward = float(kl_divergence(p, q, eps=0.0))
    backward = float(kl_divergence(q, p, eps=0.0))
    assert forward != pytest.approx(backward, rel=1e-6)
    assert min(forward, backward) > 0.0


def test_drift_uses_the_paper_direction_current_layer_first():
    """K_i is D_KL(A_i || A_{i-1}), not the reverse.

    The two orderings differ numerically, so matching one oracle and failing the
    other pins the direction down rather than just checking a magnitude.
    """
    rng = np.random.default_rng(2)
    dists = _dists(rng, 8, 50)
    got = np.asarray(drift_profile(dists, eps=0.0))

    paper = np.array(
        [entropy(np.asarray(dists[i]), np.asarray(dists[i - 1])) for i in range(1, 8)]
    )
    flipped = np.array(
        [entropy(np.asarray(dists[i - 1]), np.asarray(dists[i])) for i in range(1, 8)]
    )

    np.testing.assert_allclose(got, paper, rtol=1e-12)
    assert not np.allclose(got, flipped, rtol=1e-3)


def test_drift_profile_has_l_minus_one_entries():
    rng = np.random.default_rng(3)
    assert drift_profile(_dists(rng, 32, 576)).shape == (31,)


def test_drift_profile_needs_two_layers():
    with pytest.raises(ValueError, match="at least 2 layers"):
        drift_profile(jnp.ones((1, 5)) / 5)
    with pytest.raises(ValueError, match=r"\[L, N\]"):
        drift_profile(jnp.ones((5,)))


def test_a_constant_attention_sequence_has_zero_drift():
    dists = jnp.tile(jnp.asarray([0.5, 0.3, 0.2]), (6, 1))
    np.testing.assert_allclose(np.asarray(drift_profile(dists)), 0.0, atol=1e-12)


def test_zero_in_the_reference_stays_finite_only_with_smoothing():
    p = jnp.asarray([0.5, 0.5, 0.0])
    q = jnp.asarray([0.0, 0.5, 0.5])
    assert np.isinf(float(kl_divergence(p, q, eps=0.0)))
    smoothed = float(kl_divergence(p, q, eps=1e-9))
    assert np.isfinite(smoothed)
    assert smoothed < np.log(3 / 1e-9)


def test_smoothing_keeps_a_distribution_normalized_and_positive():
    rng = np.random.default_rng(4)
    p = _dists(rng, 1, 30)[0].at[3].set(0.0)
    p = p / p.sum()
    s = smooth(p, 1e-6)
    assert float(s.sum()) == pytest.approx(1.0, rel=1e-12)
    assert float(s.min()) >= 1e-6 / 30 * (1 - 1e-12)


def test_smoothing_barely_moves_a_well_separated_pair():
    rng = np.random.default_rng(5)
    p, q = _dists(rng, 2, 100)
    exact = float(kl_divergence(p, q, eps=0.0))
    assert float(kl_divergence(p, q, eps=1e-9)) == pytest.approx(exact, abs=1e-7)


def test_mask_layer_is_the_layer_before_the_peak():
    """drift[j] is K_{j+2}, so l_mask = argmax(drift) as a 0-based layer index."""
    drift = jnp.asarray([0.1, 0.2, 0.9, 0.3, 0.15])  # peak at j*=2, i.e. K_4
    j = select_mask_layer(drift)
    assert j == 2
    # paper's 1-based statement: i* = 4, l_mask = 3, which is 0-based index 2
    assert j + 2 == 4
    assert j + 1 == 3


def test_mask_layer_at_the_first_transition():
    """A peak at K_2 selects layer 1, the first layer, 0-based 0."""
    drift = jnp.asarray([5.0, 0.1, 0.1, 0.1])
    assert select_mask_layer(drift) == 0


def test_mask_layer_at_the_last_transition_never_picks_the_final_layer():
    """A peak at K_L selects layer L-1, so index L-2, one short of the end."""
    num_layers = 32
    drift = jnp.zeros(num_layers - 1).at[-1].set(1.0)
    j = select_mask_layer(drift)
    assert j == num_layers - 2 == 30
    assert j < num_layers - 1


def test_mask_layer_breaks_ties_low():
    assert select_mask_layer(jnp.asarray([1.0, 1.0, 0.5])) == 0


def test_mask_layer_on_a_real_profile_agrees_with_numpy():
    rng = np.random.default_rng(6)
    drift = drift_profile(_dists(rng, 32, 200))
    assert select_mask_layer(drift) == int(np.argmax(np.asarray(drift)))
