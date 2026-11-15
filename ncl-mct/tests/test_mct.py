"""CPU checks for the 1D periodic MCT step and known-law supervision."""

import numpy as np
import pytest
import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from ncl_mct import (
    LocalConstitutive,
    curl_penalty,
    known_law_loss,
    known_law_targets,
    mct_step,
    reaction_step,
    rollout,
    semilag_mass,
    ssprk2_compression,
)
from ncl_mct.core import periodic_central_diff


def test_constant_density_zero_velocity():
    n = 32
    dx = 1.0 / n
    dt = 0.1
    rho = jnp.ones((n,))
    I = jnp.ones((n,))
    M = rho
    def velocity(_rho):
        return jnp.zeros_like(_rho)

    rho2, I2, M2 = mct_step(rho, I, M, velocity, dt, dx)
    assert bool(jnp.allclose(rho2, rho, atol=1e-6))
    assert bool(jnp.allclose(I2, I, atol=1e-6))
    assert bool(jnp.allclose(M2, M, atol=1e-6))


def test_uniform_density_constant_velocity():
    n = 32
    dx = 1.0
    dt = 0.25
    rho0 = jnp.ones((n,))

    def velocity(_rho):
        return jnp.ones_like(_rho)

    traj = rollout(rho0, velocity, dt, dx, steps=4)
    assert traj.shape == (5, n)
    assert bool(jnp.allclose(traj, 1.0, atol=1e-5))
    mass0 = jnp.sum(traj[0]) * dx
    mass_f = jnp.sum(traj[-1]) * dx
    assert float(jnp.abs(mass_f - mass0)) < 1e-4


def test_ssprk2_zero_velocity_leaves_bump():
    n = 32
    dx = 0.1
    dt = 0.05
    x = jnp.arange(n) * dx
    center = x[n // 2]
    I = 1.0 + 0.4 * jnp.exp(-((x - center) ** 2) / 0.02)
    v = jnp.zeros((n,))
    I2 = ssprk2_compression(I, v, dt, dx)
    assert bool(jnp.allclose(I2, I, atol=1e-6))


def test_known_law_targets_sine():
    n = 64
    dx = 1.0 / n
    x = jnp.arange(n) * dx
    rho = 1.5 + 0.4 * jnp.sin(2.0 * jnp.pi * x)
    D = 1.0
    xi, f = known_law_targets(rho, D, dx)
    assert bool(jnp.allclose(xi, 1.0 / rho, atol=1e-6))

    rho_np = np.asarray(rho, dtype=np.float64)
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    drho = np.fft.ifft(1j * k * np.fft.fft(rho_np)).real
    f_spec = -D * drho
    assert np.max(np.abs(np.asarray(f, dtype=np.float64) - f_spec)) < 0.05


def test_rescaling_identifiability():
    n = 32
    dx = 1.0 / n
    x = jnp.arange(n) * dx
    rho = 1.2 + 0.3 * jnp.sin(2.0 * jnp.pi * x)
    xi, f = known_law_targets(rho, D=1.0, dx=dx)
    xi2 = 2.0 * xi
    f2 = f / 2.0
    velocity_gap = jnp.mean((xi2 * f2 - xi * f) ** 2)
    law_gap = jnp.mean((jnp.log(xi2) - jnp.log(xi)) ** 2 + (f2 - f) ** 2)
    assert float(velocity_gap) < 1e-8
    assert float(law_gap) > 0.1


def test_curl_penalty_1d_and_2d():
    f1 = jnp.sin(jnp.linspace(0.0, 1.0, 16, endpoint=False))
    assert float(curl_penalty(f1)) == 0.0

    n = 32
    ys = jnp.arange(n)
    xs = jnp.arange(n)
    yy, xx = jnp.meshgrid(ys, xs, indexing="ij")
    fx = jnp.sin(2.0 * jnp.pi * yy / n)
    fy = jnp.sin(2.0 * jnp.pi * xx / n)
    f = jnp.stack([fx, fy], axis=0)
    assert float(curl_penalty(f)) > 0.0

    phi = jnp.sin(2.0 * jnp.pi * xx / n) * jnp.cos(2.0 * jnp.pi * yy / n)
    fx_grad = periodic_central_diff(phi, spacing=1.0, axis=-1)
    fy_grad = periodic_central_diff(phi, spacing=1.0, axis=-2)
    f_grad = jnp.stack([fx_grad, fy_grad], axis=0)
    assert float(curl_penalty(f_grad)) < 1e-6


def test_known_law_short_train():
    n = 32
    dx = 1.0
    D = 1.0
    key = jax.random.PRNGKey(0)
    k_model, k_data = jax.random.split(key)
    rho = 0.5 + jax.random.uniform(k_data, (8, n))
    model = LocalConstitutive(k_model)
    loss0 = known_law_loss(model, rho, D, dx)

    opt = optax.adam(1e-2)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, opt_state, rho):
        loss, grads = eqx.filter_value_and_grad(known_law_loss)(model, rho, D, dx)
        updates, opt_state = opt.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    for _ in range(30):
        model, opt_state, _loss = step(model, opt_state, rho)
    loss_f = known_law_loss(model, rho, D, dx)
    assert float(loss_f) < float(loss0), (float(loss0), float(loss_f))
    assert float(loss_f) < 0.8 * float(loss0), (float(loss0), float(loss_f))


def test_dt_nonpositive_raises():
    n = 8
    rho = jnp.ones((n,))

    def velocity(_rho):
        return jnp.zeros_like(_rho)

    with pytest.raises(ValueError):
        mct_step(rho, None, None, velocity, dt=0.0, dx=1.0)
    with pytest.raises(ValueError):
        mct_step(rho, None, None, velocity, dt=-0.1, dx=1.0)
    with pytest.raises(ValueError):
        rollout(rho, velocity, dt=-1.0, dx=1.0, steps=1)


def test_reaction_midpoint_matches_equation_19():
    M = jnp.array([1.0, 0.5, 2.0])
    I = jnp.array([1.5, 2.0, 0.25])
    h = 0.4

    def rate_fn(mass, compression):
        return 0.3 * mass * compression

    got = reaction_step(M, I, rate_fn, h)
    m = jnp.log(M)
    k1 = rate_fn(jnp.exp(m), I)
    k2 = rate_fn(jnp.exp(m + (h / 2.0) * k1), I)
    expected = jnp.exp(m + h * k2)
    assert bool(jnp.allclose(got, expected, atol=1e-6))
    assert bool(jnp.allclose(reaction_step(M, I, None, h), M, atol=1e-6))


def test_semilag_integer_shift_and_zero_velocity():
    M = jnp.array([1.0, 2.0, 3.0, 4.0])
    v = jnp.ones_like(M)
    shifted = semilag_mass(M, v, dt=2.0, dx=1.0)
    assert bool(jnp.allclose(shifted, jnp.roll(M, 2), atol=1e-5))
    held = semilag_mass(M, jnp.zeros_like(M), dt=0.3, dx=0.25)
    assert bool(jnp.allclose(held, M, atol=1e-6))


def test_known_law_rejects_nonpositive_density():
    with pytest.raises(ValueError):
        known_law_targets(jnp.array([1.0, 0.0, 1.2]), D=1.0, dx=1.0)
    with pytest.raises(ValueError):
        known_law_targets(jnp.array([1.0, -0.2, 1.2]), D=1.0, dx=1.0)
    with pytest.raises(ValueError):
        ssprk2_compression(jnp.ones((4,)), jnp.zeros((4,)), dt=0.1, dx=0.0)
