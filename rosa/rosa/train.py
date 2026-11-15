"""Algorithm 1 step: rotate S_e, mask bottom-rho, AdamW on M_t only.

Prefix activations are not cached. Every step runs the full forward pass
and stops the backward pass at the input of G_k.
"""

from __future__ import annotations

import jax.numpy as jnp
import optax

from rosa.mask import active_parameter_mask
from rosa.model import loss_and_grads
from rosa.schedule import active_set


def adam_moments(opt_state):
    """Return (mu, nu, count) from an optax AdamW state."""

    adam_state = opt_state[0]
    return adam_state.mu, adam_state.nu, adam_state.count


class RosaTrainer:
    """Rotational sparse adaptation on a stack of per-layer parameters.

    Optimizer slots exist only for the current bottom-rho mask of S_e.
    Leaving a group deletes its moments. A group that becomes active again,
    including a revisit of G_1, starts from Adam init (mu = 0, nu = 0,
    count = 0) rather than from stored moments. Slots that stay in the mask,
    typically part of the anchor, keep their moments.
    """

    def __init__(
        self,
        params,
        schedule,
        rho=0.2,
        delta_t=600,
        learning_rate=1e-2,
        weight_decay=0.01,
        b1=0.9,
        b2=0.95,
        eps=1e-8,
    ):
        if delta_t < 1:
            raise ValueError("delta_t must be at least 1")
        if not 0 <= rho <= 1:
            raise ValueError("rho must lie in [0, 1]")
        self.params = params
        self.schedule = schedule
        self.rho = float(rho)
        self.delta_t = int(delta_t)
        self.tx = optax.adamw(
            learning_rate,
            b1=b1,
            b2=b2,
            eps=eps,
            weight_decay=weight_decay,
        )
        self.t = 0
        self.epoch = None
        self.plan = None
        self.mask = frozenset()
        self.opt_states = {}
        self._mask_built_at = None

    @property
    def active(self):
        if self.plan is None:
            return None
        return self.plan.active

    @property
    def l_start(self):
        if self.plan is None:
            return None
        return self.plan.l_start

    def begin_epoch(self, epoch: int):
        """Freeze parameters outside S_e, drop their states, cold-start newcomers."""

        plan = active_set(self.schedule, epoch)
        self.epoch = epoch
        self.plan = plan
        self._discard_states_outside(plan.active)
        self._refresh_mask()
        self._mask_built_at = self.t
        return plan

    def step(self, x, y) -> float:
        """One truncated step. AdamW writes only M_t."""

        if self.plan is None:
            raise RuntimeError("call begin_epoch before step")
        if self.t % self.delta_t == 0 and self._mask_built_at != self.t:
            self._refresh_mask()
            self._mask_built_at = self.t

        loss, grads = loss_and_grads(
            self.params,
            x,
            y,
            self.schedule.layers,
            self.l_start,
        )
        self._adamw_on_mask(grads)
        self.t += 1
        return float(loss)

    def _discard_states_outside(self, active_ids):
        active = set(active_ids)
        self.opt_states = {
            slot: state
            for slot, state in self.opt_states.items()
            if slot[0] in active
        }

    def _refresh_mask(self):
        slots = active_parameter_mask(self.params, self.plan.active, self.rho)
        new_mask = frozenset(slots)
        kept = {slot: self.opt_states[slot] for slot in new_mask if slot in self.opt_states}
        for slot in slots:
            if slot not in kept:
                value = _read_slot(self.params, slot)
                kept[slot] = self.tx.init(value)
        self.opt_states = kept
        self.mask = new_mask

    def _adamw_on_mask(self, grads):
        if not self.mask:
            return
        grouped = {}
        new_states = {}
        for slot in self.mask:
            value = _read_slot(self.params, slot)
            grad = _read_slot(grads, slot)
            updates, new_state = self.tx.update(grad, self.opt_states[slot], value)
            new_value = optax.apply_updates(value, updates)
            layer_id, name, index = slot
            grouped.setdefault((layer_id, name), []).append((index, new_value))
            new_states[slot] = new_state
        self.params = _write_grouped(self.params, grouped)
        self.opt_states = new_states


def _read_slot(tree, slot):
    layer_id, name, index = slot
    flat = tree[layer_id][name].reshape((-1,))
    return flat[index]


def _write_grouped(params, grouped):
    updated = {layer_id: dict(layer) for layer_id, layer in params.items()}
    for (layer_id, name), pairs in grouped.items():
        array = params[layer_id][name]
        flat = array.reshape((-1,))
        indices = jnp.asarray([index for index, _ in pairs], dtype=jnp.int32)
        values = jnp.stack([value for _, value in pairs])
        flat = flat.at[indices].set(values)
        updated[layer_id][name] = flat.reshape(array.shape)
    return updated
