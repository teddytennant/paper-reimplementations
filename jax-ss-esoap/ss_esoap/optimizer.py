"""SS-ESOAP and a SOAP baseline as optax gradient transformations.

Reference: Wang, Toftrup, Loeschcke, Wang, Anandkumar, "SS-ESOAP: Self-Scaled
Adaptive Preconditioning for Physics-Informed Learning", arXiv:2608.29448v1.
Algorithm 2 (Appendix B) is the version implemented here; Algorithm 1 in the body
is a condensed form of the same thing.

SS-ESOAP is SOAP plus three changes, and the two constructors below differ only in
those three flags, so the delta is readable as a diff:

  1. Self-scaling. A scalar ``tau_t`` from the secant condition, equation (3),
     rescales the eigenspace step by ``tau_t^{-1/2}``.
  2. Adaptive basis trigger. The eigenbasis is recomputed when the off-diagonal
     ratio (4) exceeds ``tau_trigger``, not on a fixed period.
  3. Variance-state downscaling. On a triggered basis change the second moment is
     multiplied by ``gamma < 1`` and keeps its coordinatewise shape, instead of
     being reprojected into the new basis (Section 3.2).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from . import _core
from ._core import Config, SSESOAPState

__all__ = ["scale_by_ss_esoap", "ss_esoap", "soap", "SSESOAPState"]


def scale_by_ss_esoap(cfg: Config) -> optax.GradientTransformation:
    """The preconditioner alone, without the learning rate or the sign flip.

    ``update`` requires ``params``. Step 4 of Algorithm 2 needs the true
    parameter displacement ``S_t = W_{t-1} - W_{t-2}``, and ``tau_t`` is not
    invariant to how that displacement is scaled: the numerator of (3) is linear
    in ``S`` and the denominator is quadratic, so ``tau`` picks up a factor of
    ``1 / eta``. Reading the displacement off ``params`` rather than off our own
    emitted updates keeps this correct no matter what else is in the chain.
    """

    def init_fn(params):
        leaves = jax.tree.map(
            lambda p: _core.init_leaf(p, cfg.eps, cfg.max_precond_dim), params
        )
        return SSESOAPState(count=jnp.zeros((), jnp.int32), leaves=leaves)

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError(
                "ss_esoap needs params: the secant factor tau_t in equation (3) "
                "is computed from W_{t-1} - W_{t-2}."
            )
        count = state.count
        grads, treedef = jax.tree.flatten(updates)
        leaves = treedef.flatten_up_to(state.leaves)
        ps = treedef.flatten_up_to(params)
        pairs = [
            _core.update_leaf(g, s_, p, count, cfg)
            for g, s_, p in zip(grads, leaves, ps)
        ]
        new_updates = jax.tree.unflatten(treedef, [u for u, _ in pairs])
        new_leaves = treedef.unflatten([s_ for _, s_ in pairs])
        return new_updates, SSESOAPState(count=count + 1, leaves=new_leaves)

    return optax.GradientTransformation(init_fn, update_fn)


def _build(learning_rate, weight_decay: float, cfg: Config):
    steps = [scale_by_ss_esoap(cfg)]
    if weight_decay:
        # Algorithm 2 step 5 decays the parameters, W_t <- (1 - eta lambda) W_{t-1}
        # - eta U_t, which is decoupled decay folded into the same learning rate.
        steps.append(optax.add_decayed_weights(weight_decay))
    steps.append(optax.scale_by_learning_rate(learning_rate))
    return optax.chain(*steps)


def ss_esoap(
    learning_rate,
    b1: float = 0.9,
    b2: float = 0.95,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    tau_min: float = 0.1,
    check_interval: int = 1,
    trigger: float = 0.2,
    warmup_steps: int = 0,
    gamma: float | None = None,
    self_scale: bool = True,
    adaptive_basis: bool = True,
    basis_updates: bool = True,
    variance_transition: str = "downscale",
    max_precond_dim: int = 8192,
) -> optax.GradientTransformation:
    """SS-ESOAP, Algorithm 2.

    Defaults follow Table 6 where the paper gives a number: ``b1 = 0.9``,
    ``b2 = 0.95``, ``eps = 1e-8``, ``check_interval = 1``, ``trigger = 0.2``, and
    the adaptive gamma ladder of Section 3.2. Two knobs in the Algorithm 2
    signature never get a value anywhere in the paper, so the defaults here are
    ours: ``tau_min = 0.1`` and ``warmup_steps = 0``. Neither appears in Table 6
    or in the ablations. ``tau_min`` is load-bearing, because ``tau <= 1`` means
    self-scaling can only ever grow the step, by ``tau^{-1/2}``, and 0.1 caps
    that growth at about 3.2x. Section 3.1 reports that ``tau = 1`` holds on most
    iterations. That is not what happens here. ``L`` and ``R`` are both gradient
    second moments and they multiply in the denominator of (3), so ``c_t / a_t``
    goes as the fifth power of the gradient magnitude; once training gets
    accurate it falls through the floor and stays there. See
    ``tests/test_scale_invariance.py``.

    Args:
      gamma: fixed variance downscaling factor. ``None`` uses the ladder,
        0.25 / 0.5 / 0.75 by the size of the off-diagonal ratio.
      self_scale: step 4. Off gives plain SOAP-style scaling.
      adaptive_basis: step 2. Off falls back to a fixed period.
      basis_updates: freeze ``Q_L = Q_R = I`` when False, which reduces the whole
        transformation to Adam.
      variance_transition: "downscale" (the paper's), "reproject" (what it argues
        against), "reset", or "keep".
    """
    cfg = Config(
        b1=b1, b2=b2, eps=eps, tau_min=tau_min, check_interval=check_interval,
        trigger=trigger, warmup_steps=warmup_steps, self_scale=self_scale,
        adaptive_basis=adaptive_basis, variance_transition=variance_transition,
        gamma=gamma, basis_updates=basis_updates,
        max_precond_dim=max_precond_dim,
    )
    return _build(learning_rate, weight_decay, cfg)


def soap(
    learning_rate,
    b1: float = 0.9,
    b2: float = 0.95,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    precondition_frequency: int = 10,
    max_precond_dim: int = 8192,
) -> optax.GradientTransformation:
    """SOAP baseline: Adam in the Shampoo eigenbasis, Section 2.3.

    Fixed-period basis refresh, and on refresh both the momentum and the second
    moment are carried over by the similarity transform
    ``V <- (Q_new^T Q_old) V (Q_old^T Q_new)``. That is the behaviour Section 2.4
    describes and objects to, so it is what the baseline does. The reference SOAP
    implementation differs in details this does not try to match.
    """
    cfg = Config(
        b1=b1, b2=b2, eps=eps, self_scale=False, adaptive_basis=False,
        precondition_frequency=precondition_frequency,
        variance_transition="reproject", max_precond_dim=max_precond_dim,
    )
    return _build(learning_rate, weight_decay, cfg)
