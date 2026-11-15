"""A few gradient steps of Suan on a tiny categorical policy."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from suan.losses import DEFAULT_BETA, DEFAULT_TAU, suan_loss
from suan.policy import log_prob


def main() -> None:
    n_prompts, n_actions, batch = 4, 5, 8
    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    logits = jax.random.normal(k1, (n_prompts, n_actions))
    ref_logits = jax.random.normal(k2, (n_prompts, n_actions))
    x = jax.random.randint(k3, (batch,), 0, n_prompts)
    y_plus = jax.random.randint(k4, (batch,), 0, n_actions)
    y_minus = (y_plus + 1) % n_actions
    ref_plus = log_prob(ref_logits, x, y_plus)
    ref_minus = log_prob(ref_logits, x, y_minus)

    def objective(theta):
        return jnp.mean(
            suan_loss(
                log_prob(theta, x, y_plus),
                log_prob(theta, x, y_minus),
                ref_plus,
                ref_minus,
                tau=DEFAULT_TAU,
                beta=DEFAULT_BETA,
            )
        )

    def margin(theta):
        return jnp.mean(log_prob(theta, x, y_plus) - log_prob(theta, x, y_minus))

    grad_fn = jax.jit(jax.grad(objective))
    obj_fn = jax.jit(objective)
    print(f"step 0  L_Suan={float(obj_fn(logits)):.6f}  margin={float(margin(logits)):.6f}")
    for step in range(1, 6):
        logits = logits - 0.4 * grad_fn(logits)
        print(f"step {step}  L_Suan={float(obj_fn(logits)):.6f}  margin={float(margin(logits)):.6f}")


if __name__ == "__main__":
    main()
