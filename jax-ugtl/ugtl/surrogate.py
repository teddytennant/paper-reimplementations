"""The surrogate ensemble of Eq. (23) and the proxy gradient it supplies.

Line 1 of Algorithm 5 trains M_ens independently initialized models on the
offline dataset and derives two quantities from them: the ensemble mean mu_phi,
which is the proxy score used everywhere downstream, and the ensemble dispersion
sigma_phi, which sets the sampling temperature in Eq. (24).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import optax

from ugtl.nn import apply_mlp, init_mlp

__all__ = ["EnsembleSurrogate", "train_ensemble"]


@dataclass
class EnsembleSurrogate:
    """M_ens MLPs stacked along a leading axis, plus the input/output scaling.

    ``mean`` and ``dispersion`` are mu_phi and sigma_phi of Eq. (23). Both are
    reported in the units of the training targets, so mu_phi is directly
    comparable with the observed scores y.
    """

    params: list
    x_mean: jnp.ndarray
    x_std: jnp.ndarray
    y_mean: jnp.ndarray
    y_std: jnp.ndarray

    def _members(self, x: jnp.ndarray) -> jnp.ndarray:
        """Every member's prediction for a batch, shaped (M_ens, batch)."""
        xn = (x - self.x_mean) / self.x_std
        out = jax.vmap(lambda p: apply_mlp(p, xn)[..., 0])(self.params)
        return out * self.y_std + self.y_mean

    def members(self, x: jnp.ndarray) -> jnp.ndarray:
        return self._members(jnp.atleast_2d(x))

    def mean(self, x: jnp.ndarray) -> jnp.ndarray:
        """mu_phi, the first half of Eq. (23)."""
        return jnp.mean(self._members(jnp.atleast_2d(x)), axis=0)

    def dispersion(self, x: jnp.ndarray) -> jnp.ndarray:
        """sigma_phi, the second half of Eq. (23).

        The 1 / M_ens normalization is the paper's, not the unbiased 1 / (M - 1).
        """
        preds = self._members(jnp.atleast_2d(x))
        return jnp.sqrt(jnp.mean((preds - jnp.mean(preds, axis=0)) ** 2, axis=0))

    def proxy_gradient(self, x: jnp.ndarray) -> jnp.ndarray:
        """grad_x mu_phi(x), the directional signal Algorithm 5 steers with."""
        single = lambda z: jnp.mean(self._members(z[None, :]))
        return jax.vmap(jax.grad(single))(jnp.atleast_2d(x))


@partial(jax.jit, static_argnums=(4, 5, 6, 7, 8))
def _fit(member_keys, train_key, X, y, n_members, width, depth, steps, batch_size, lr):
    n, d = X.shape
    sizes = [d] + [width] * depth + [1]
    params = jax.vmap(lambda k: init_mlp(k, sizes))(member_keys)
    opt = optax.adam(lr)
    state = opt.init(params)

    def loss_fn(params, idx):
        def member_loss(p, i):
            pred = apply_mlp(p, X[i])[..., 0]
            return jnp.mean((pred - y[i]) ** 2)

        return jnp.mean(jax.vmap(member_loss)(params, idx))

    def step(carry, k):
        params, state = carry
        idx = jax.random.randint(k, (n_members, batch_size), 0, n)
        loss, grads = jax.value_and_grad(loss_fn)(params, idx)
        updates, state = opt.update(grads, state, params)
        return (optax.apply_updates(params, updates), state), loss

    keys = jax.random.split(train_key, steps)
    (params, _), losses = jax.lax.scan(step, (params, state), keys)
    return params, losses


def train_ensemble(
    key: jax.Array,
    X: jnp.ndarray,
    y: jnp.ndarray,
    n_members: int = 5,
    width: int = 128,
    depth: int = 2,
    steps: int = 600,
    batch_size: int = 128,
    lr: float = 1e-3,
) -> EnsembleSurrogate:
    """Line 1 of Algorithm 5.

    Members differ through their initialization and through independently drawn
    minibatches, which is what gives sigma_phi anything to measure. The paper
    uses M_ens = 10 two-hidden-layer MLPs of width 1024 trained for 5000 steps;
    the defaults here are smaller so a run finishes on a CPU.
    """
    X = jnp.asarray(X, dtype=jnp.float32)
    y = jnp.asarray(y, dtype=jnp.float32)
    x_mean, x_std = X.mean(axis=0), X.std(axis=0) + 1e-6
    y_mean, y_std = y.mean(), y.std() + 1e-6
    member_keys, train_key = jax.random.split(key, 2)
    member_keys = jax.random.split(member_keys, n_members)
    params, _ = _fit(
        member_keys,
        train_key,
        (X - x_mean) / x_std,
        (y - y_mean) / y_std,
        n_members,
        width,
        depth,
        steps,
        batch_size,
        lr,
    )
    return EnsembleSurrogate(params, x_mean, x_std, y_mean, y_std)
