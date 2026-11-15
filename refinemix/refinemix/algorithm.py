"""Algorithm 1 data selection and a single training step.

    draw t
    if t <= tmax: x0 ~ lambda * source + (1 - lambda) * target
    else:         x0 ~ target
    noise x0 to x_t
    gradient step on equation 13 at level t

`select_domain` returns "source" or "target".
P(source | t <= tmax) = lam, and P(source | t > tmax) = 0.

The training step defaults to the experimental time sampler t ~ Uniform(eps, 1).
Pass `time_mode="algorithm"` for Uniform(0, 1), or pass `time_sampler` to use
whatever sampler the caller supplies. The selection rule is applied to those
times unchanged.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from refinemix.kernels import noise_tokens
from refinemix.loss import masked_nelbo
from refinemix.schedule import EPS, sample_time

SOURCE = "source"
TARGET = "target"


def domain_indicator(t, tmax, lam, key):
    """1 if the row is drawn from the source, 0 if from the target.

    Jittable. `t` may be a scalar or a batch of times. One key draws every row.
    """
    t = jnp.asarray(t)
    scalar = t.ndim == 0
    t_b = t.reshape((1,)) if scalar else t
    u = jax.random.uniform(key, t_b.shape)
    source = (t_b <= tmax) & (u < lam)
    ind = source.astype(jnp.int32)
    if scalar:
        return ind[0]
    return ind


def select_domain(t, tmax, lam, key):
    """Return "source" or "target" (a string, or an array of strings)."""
    ind = domain_indicator(t, tmax, lam, key)
    names = np.where(np.asarray(ind) == 1, SOURCE, TARGET)
    if np.asarray(t).ndim == 0:
        return str(names.reshape(()))
    return names


def _draw_times(key, batch, time_sampler, time_mode, eps):
    if time_sampler is not None:
        return jnp.asarray(time_sampler(key, (batch,)), dtype=jnp.float32)
    return sample_time(key, (batch,), mode=time_mode, eps=eps)


def training_step(
    params,
    target_batch,
    source_batch,
    key,
    tmax,
    lam,
    apply_fn,
    mask_id: int,
    optimizer=None,
    opt_state=None,
    time_sampler=None,
    time_mode: str = "experimental",
    eps: float = EPS,
    ideal: bool = False,
):
    """One Algorithm 1 step.

    Returns a dict with `loss`, integer `domain` (1 = source, 0 = target),
    `domain_name` ("source" / "target"), the times `t`, the selected clean
    batch `x0`, and the noised batch `zt`. If `optimizer` and `opt_state`
    are given, also returns updated `params` and `opt_state`.
    """
    target_batch = jnp.asarray(target_batch, dtype=jnp.int32)
    source_batch = jnp.asarray(source_batch, dtype=jnp.int32)
    if target_batch.shape != source_batch.shape:
        raise ValueError(
            f"batch shape mismatch: target {target_batch.shape} vs source {source_batch.shape}"
        )
    k_t, k_dom, k_noise = jax.random.split(key, 3)
    batch = target_batch.shape[0]
    t = _draw_times(k_t, batch, time_sampler, time_mode, eps)
    domain = domain_indicator(t, tmax, lam, k_dom)
    x0 = jnp.where(domain[:, None] == 1, source_batch, target_batch)
    zt = noise_tokens(
        x0, t, k_noise, kernel="masked", mask_id=mask_id, eps=eps, ideal=ideal
    )

    def loss_fn(p):
        logits = apply_fn(p, zt, t)
        return masked_nelbo(logits, x0, zt, t, mask_id, eps=eps, ideal=ideal)

    updated_params = params
    updated_state = opt_state
    if optimizer is not None and opt_state is not None:
        import optax

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, updated_state = optimizer.update(grads, opt_state, params)
        updated_params = optax.apply_updates(params, updates)
    else:
        loss = loss_fn(params)

    return {
        "loss": loss,
        "domain": domain,
        "domain_name": np.where(np.asarray(domain) == 1, SOURCE, TARGET),
        "t": t,
        "x0": x0,
        "zt": zt,
        "params": updated_params,
        "opt_state": updated_state,
    }


def make_train_step(
    apply_fn,
    optimizer,
    mask_id: int,
    *,
    time_mode: str = "experimental",
    eps: float = EPS,
    ideal: bool = False,
):
    """Jitted Algorithm 1 update. Returns params, opt_state, loss, domain, t."""
    import optax

    @jax.jit
    def step(params, opt_state, target_batch, source_batch, key, tmax, lam):
        k_t, k_dom, k_noise = jax.random.split(key, 3)
        batch = target_batch.shape[0]
        t = sample_time(k_t, (batch,), mode=time_mode, eps=eps)
        domain = domain_indicator(t, tmax, lam, k_dom)
        x0 = jnp.where(domain[:, None] == 1, source_batch, target_batch)
        zt = noise_tokens(
            x0,
            t,
            k_noise,
            kernel="masked",
            mask_id=mask_id,
            eps=eps,
            ideal=ideal,
        )

        def loss_fn(p):
            logits = apply_fn(p, zt, t)
            return masked_nelbo(logits, x0, zt, t, mask_id, eps=eps, ideal=ideal)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, domain, t

    return step
