"""One MeqMuon step and pytree helpers. State is a single momentum buffer."""

import jax
import jax.numpy as jnp

from meqmuon.normalize import (
    column_normalize,
    global_rms_normalize,
    one_sided_normalize,
    row_normalize,
)
from meqmuon.orthogonalize import newton_schulz


def _tree_map(fn, *trees):
    if hasattr(jax.tree, "map"):
        return jax.tree.map(fn, *trees)
    return jax.tree_util.tree_map(fn, *trees)


def meqmuon_step(
    param,
    grad,
    momentum,
    lr,
    kind,
    *,
    mu=0.95,
    rho=0.2,
    weight_decay=0.1,
    nesterov=False,
    ns_steps=5,
    delta=1e-7,
):
    """Apply one MeqMuon update.

    kind is "hidden", "matrix", or "vector". Hidden weights are orthogonalized
    then normalized on the more imbalanced side. Matrix weights get row RMS
    normalization followed by column RMS normalization. Vectors get one global
    RMS normalization.
    """
    param = jnp.asarray(param)
    grad = jnp.asarray(grad)
    momentum = jnp.asarray(momentum)

    new_momentum = mu * momentum + grad

    if kind == "hidden":
        if param.ndim != 2:
            raise ValueError(f"kind 'hidden' requires ndim 2, got {param.ndim}")
        # After the buffer update, Nesterov orthogonalizes G + mu * B_new.
        lookahead = grad + mu * new_momentum
        h = jnp.where(nesterov, lookahead, new_momentum)
        direction = one_sided_normalize(newton_schulz(h, steps=ns_steps), delta=delta)
    elif kind == "matrix":
        if param.ndim != 2:
            raise ValueError(f"kind 'matrix' requires ndim 2, got {param.ndim}")
        direction = column_normalize(row_normalize(new_momentum, delta=delta), delta=delta)
    elif kind == "vector":
        if param.ndim != 1:
            raise ValueError(f"kind 'vector' requires ndim 1, got {param.ndim}")
        direction = global_rms_normalize(new_momentum)
    else:
        raise ValueError(f"unknown kind {kind!r}, expected 'hidden', 'matrix', or 'vector'")

    new_param = (1 - lr * weight_decay) * param - rho * lr * direction
    return new_param, new_momentum


def init_momentum(params):
    """Zeros matching params. This is the only optimizer state."""
    return _tree_map(jnp.zeros_like, params)


def step_pytree(params, grads, momentum, kinds, lr, **hyperparameters):
    """Step a pytree of parameters. kinds is a matching pytree of kind strings."""

    def _one(param, grad, mom, kind):
        return meqmuon_step(param, grad, mom, lr, kind, **hyperparameters)

    if isinstance(params, dict):
        new_params = {}
        new_momentum = {}
        for key in params:
            new_params[key], new_momentum[key] = _one(
                params[key], grads[key], momentum[key], kinds[key]
            )
        return new_params, new_momentum

    return _tree_map(_one, params, grads, momentum, kinds)
