import numpy as np

from refinemix.posterior import (
    bayes_posterior_probs,
    general_posterior_probs,
    masked_posterior_probs,
)
from refinemix.schedule import alpha


def _eq1(z_from, alpha_t, pi):
    probs = (1.0 - alpha_t) * np.array(pi, dtype=np.float64)
    probs[z_from] += alpha_t
    return probs


def _direct_bayes(zt, z0, alpha_s, alpha_t, pi):
    """p(z_s | z_t, z_0) from equation 1, including the transition alpha_t/alpha_s."""
    pi = np.asarray(pi, dtype=np.float64)
    alpha_ts = alpha_t / alpha_s
    p_zs = _eq1(z0, alpha_s, pi)
    p_zt = _eq1(z0, alpha_t, pi)[zt]
    joint = np.zeros(pi.shape[0], dtype=np.float64)
    for zs in range(pi.shape[0]):
        joint[zs] = _eq1(zs, alpha_ts, pi)[zt] * p_zs[zs]
    return joint / p_zt


def test_eq12_matches_direct_bayes_on_vocab_4():
    vocab = 4
    mask_id = 3
    pi = np.zeros(vocab, dtype=np.float64)
    pi[mask_id] = 1.0
    z0 = 1
    pairs = [(0.2, 0.6), (0.35, 0.85)]
    current = {
        "unmasked": z0,  # the only unmasked token with positive forward mass
        "masked": mask_id,
    }
    for s, t in pairs:
        alpha_s = float(alpha(s))
        alpha_t = float(alpha(t))
        assert alpha_s > alpha_t
        for name, zt in current.items():
            bayes = _direct_bayes(zt, z0, alpha_s, alpha_t, pi)
            eq12 = np.asarray(
                masked_posterior_probs(zt, z0, alpha_s, alpha_t, mask_id, vocab),
                dtype=np.float64,
            )
            eq11 = np.asarray(
                general_posterior_probs(zt, z0, alpha_s, alpha_t, pi),
                dtype=np.float64,
            )
            library_bayes = np.asarray(
                bayes_posterior_probs(zt, z0, alpha_s, alpha_t, pi),
                dtype=np.float64,
            )
            assert np.allclose(eq12, bayes, atol=1e-6), (name, s, t, eq12, bayes)
            assert np.allclose(eq11, bayes, atol=1e-6), (name, s, t)
            assert np.allclose(library_bayes, bayes, atol=1e-6)
            assert abs(eq12.sum() - 1.0) < 1e-6
            if name == "unmasked":
                assert abs(eq12[zt] - 1.0) < 1e-6
            else:
                # Point mass is not on the mask; mass splits between mask and z0.
                assert eq12[mask_id] > 0.0
                assert eq12[z0] > 0.0
                assert abs(eq12[mask_id] + eq12[z0] - 1.0) < 1e-6


def test_eq11_matches_direct_bayes_for_uniform_kernel():
    vocab = 4
    pi = np.full(vocab, 1.0 / vocab)
    z0 = 1
    for s, t in [(0.2, 0.6), (0.35, 0.85)]:
        alpha_s = float(alpha(s))
        alpha_t = float(alpha(t))
        for zt in (z0, 2):
            bayes = _direct_bayes(zt, z0, alpha_s, alpha_t, pi)
            eq11 = np.asarray(general_posterior_probs(zt, z0, alpha_s, alpha_t, pi))
            library_bayes = np.asarray(bayes_posterior_probs(zt, z0, alpha_s, alpha_t, pi))
            assert np.allclose(eq11, bayes, atol=1e-6)
            assert np.allclose(library_bayes, bayes, atol=1e-6)
            assert abs(float(eq11.sum()) - 1.0) < 1e-6
