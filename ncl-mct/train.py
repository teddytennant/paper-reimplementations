"""Short known-law fit and a nonreactive linear-diffusion rollout."""

import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from ncl_mct import (
    LocalConstitutive,
    known_law_loss,
    linear_diffusion_velocity,
    rollout,
)


def main():
    key = jax.random.PRNGKey(0)
    k_model, k_data = jax.random.split(key)
    n = 32
    dx = 1.0 / n
    D = 1.0
    rho = 0.4 + jax.random.uniform(k_data, (8, n))
    model = LocalConstitutive(k_model)
    opt = optax.adam(1e-2)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, opt_state, batch):
        loss, grads = eqx.filter_value_and_grad(known_law_loss)(model, batch, D, dx)
        updates, opt_state = opt.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    loss0 = known_law_loss(model, rho, D, dx)
    for _ in range(40):
        model, opt_state, _loss = step(model, opt_state, rho)
    loss_f = known_law_loss(model, rho, D, dx)
    print(f"initial known-law loss {float(loss0):.6e}")
    print(f"final known-law loss {float(loss_f):.6e}")
    if not (float(loss_f) < float(loss0)):
        return 1

    n_roll = 48
    dx_roll = 1.0 / n_roll
    dt = 1e-3
    D_roll = 0.05
    x = jnp.arange(n_roll) * dx_roll
    rho0 = 1.0 + 0.2 * jnp.sin(2.0 * jnp.pi * x)

    def velocity_fn(density):
        return linear_diffusion_velocity(density, D_roll, dx_roll)

    traj = rollout(rho0, velocity_fn, dt, dx_roll, steps=20)
    mass0 = jnp.sum(rho0) * dx_roll
    mass_f = jnp.sum(traj[-1]) * dx_roll
    rel = float(jnp.abs(mass_f - mass0) / jnp.abs(mass0))
    print(f"mass relative error {rel:.6e}")
    if bool(jnp.isnan(jnp.max(traj))) or rel > 0.05:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
