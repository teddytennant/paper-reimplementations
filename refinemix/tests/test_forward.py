import numpy as np
import jax.numpy as jnp

from refinemix.kernels import token_forward_probs
from refinemix.schedule import alpha


def test_masked_keep_rate_matches_closed_form():
    vocab = 6
    mask_id = 5
    z0 = jnp.array([0, 1, 2, 3, 4])
    t = 0.3
    probs = np.asarray(
        token_forward_probs(z0, t, vocab_size=vocab, kernel="masked", mask_id=mask_id)
    )
    a = float(alpha(t))
    for i, token in enumerate(np.asarray(z0)):
        explicit = np.zeros(vocab)
        explicit[token] = a
        explicit[mask_id] = 1.0 - a
        assert np.allclose(probs[i], explicit, atol=1e-6)
        assert abs(probs[i, token] - a) < 1e-6
        assert abs(probs[i].sum() - 1.0) < 1e-6


def test_uniform_keep_rate_and_resample_distribution():
    # N non-mask tokens, no absorbing state. Keep branch has weight alpha_t;
    # the resample distribution is uniform on those N tokens.
    n = 5
    z0 = jnp.arange(n)
    t = 0.4
    probs = np.asarray(
        token_forward_probs(z0, t, vocab_size=n, kernel="uniform", mask_id=None)
    )
    a = float(alpha(t))
    pi = np.full(n, 1.0 / n)
    for i, token in enumerate(range(n)):
        explicit = (1.0 - a) * pi
        explicit[token] += a
        assert np.allclose(probs[i], explicit, atol=1e-6)
        # Recover the keep-branch weight: excess mass over the uniform floor.
        floor = (1.0 - a) / n
        keep = probs[i, token] - floor
        assert abs(keep - a) < 1e-6
        off = np.delete(probs[i], token)
        assert np.allclose(off, floor, atol=1e-6)
        assert np.max(off) - np.min(off) < 1e-6
