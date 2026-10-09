import jax
import jax.numpy as jnp
import numpy as np

from refinemix.loss import masked_nelbo, subs_logits
from refinemix.schedule import nelbo_weight


def _log_softmax_subs(row, mask_id):
    row = np.array(row, dtype=np.float64).copy()
    row[mask_id] = -np.inf
    finite = np.isfinite(row)
    shift = np.max(row[finite])
    exp = np.exp(row - shift)
    exp[~finite] = 0.0
    return row - shift - np.log(np.sum(exp))


def test_eq13_matches_hand_expanded_sum_and_sign():
    mask_id = 3
    z0 = jnp.array([[0, 1, 2]], dtype=jnp.int32)
    zt = jnp.array([[0, mask_id, mask_id]], dtype=jnp.int32)
    logits = jnp.array(
        [
            [
                [3.0, 1.0, -2.0, 4.0],  # unmasked position; must not enter the sum
                [0.0, 0.0, 0.0, 9.0],
                [0.0, 1.0, 0.0, 9.0],
            ]
        ],
        dtype=jnp.float32,
    )
    t = 0.25
    loss = masked_nelbo(logits, z0, zt, jnp.asarray(t), mask_id, reduction="none")

    lp_pos1 = _log_softmax_subs([0.0, 0.0, 0.0, 9.0], mask_id)[1]
    lp_pos2 = _log_softmax_subs([0.0, 1.0, 0.0, 9.0], mask_id)[2]
    weight = float(nelbo_weight(t))
    assert weight < 0.0
    assert abs(weight - (-1.0 / t)) < 1e-5
    hand = weight * (lp_pos1 + lp_pos2)
    assert lp_pos1 < 0.0 and lp_pos2 < 0.0
    assert hand > 0.0
    assert abs(float(loss[0]) - hand) < 1e-5

    # Unmasked position contributes 0: changing it does not change the loss.
    logits_changed = logits.at[0, 0].set(jnp.array([8.0, -3.0, 2.0, 1.0]))
    z0_changed = z0.at[0, 0].set(2)
    loss_changed = masked_nelbo(
        logits_changed, z0_changed, zt, jnp.asarray(t), mask_id, reduction="none"
    )
    assert np.allclose(np.asarray(loss), np.asarray(loss_changed), atol=0.0)

    constrained = subs_logits(logits, mask_id)
    assert np.all(np.isneginf(np.asarray(constrained[..., mask_id])))
    probs = np.asarray(jax.nn.softmax(constrained, axis=-1))
    assert np.all(probs[..., mask_id] == 0.0)
    assert np.allclose(probs.sum(axis=-1), 1.0, atol=1e-6)
