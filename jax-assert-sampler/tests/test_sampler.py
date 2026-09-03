"""Equations 3-7, plus the sampler's agreement with the target distribution when it
is driven by the exact score."""

import jax
import jax.numpy as jnp
import pytest

from assert_sampler import metrics, sampler, schedules
from assert_sampler.diffusion import default_mixture, linear_schedule, uniform_subsequence


@pytest.fixture(scope="module")
def sched():
    return linear_schedule()


def test_uniform_subsequence_matches_section_4_1():
    T, S = 1000, 50
    ts = uniform_subsequence(T, S)
    want = [int(j * T // S) for j in range(S)][::-1]
    assert list(map(int, ts)) == want
    assert int(ts[-1]) == 0
    assert bool(jnp.all(jnp.diff(ts) < 0))


def test_alpha_bar_before_the_first_step_is_one(sched):
    assert float(sched.alpha_bar(jnp.array(-1))) == 1.0


def test_predict_x0_inverts_the_forward_process(sched):
    key = jax.random.key(0)
    k0, ke = jax.random.split(key)
    x0 = jax.random.normal(k0, (64, 2))
    eps = jax.random.normal(ke, (64, 2))
    ab = sched.alphas_cumprod[300]
    xt = jnp.sqrt(ab) * x0 + jnp.sqrt(1.0 - ab) * eps
    assert jnp.allclose(sampler.predict_x0(xt, eps, ab), x0, atol=1e-4)


def test_eta_zero_is_the_deterministic_ddim_update(sched):
    """Equation 5 at eta = 0 must equal the closed-form deterministic update."""
    key = jax.random.key(1)
    kx, ke, kz = jax.random.split(key, 3)
    x = jax.random.normal(kx, (128, 2))
    eps = jax.random.normal(ke, (128, 2))
    z = jax.random.normal(kz, (128, 2))
    ab_t, ab_s = sched.alphas_cumprod[600], sched.alphas_cumprod[580]
    got = sampler.ddim_step(x, eps, ab_t, ab_s, 0.0, z)
    want = (jnp.sqrt(ab_s) * (x - jnp.sqrt(1.0 - ab_t) * eps) / jnp.sqrt(ab_t)
            + jnp.sqrt(1.0 - ab_s) * eps)
    assert jnp.allclose(got, want, atol=1e-6)
    # and z must not enter at all.
    other = sampler.ddim_step(x, eps, ab_t, ab_s, 0.0, 3.0 * z + 1.0)
    assert jnp.array_equal(got, other)


def test_sigma_matches_the_ddpm_posterior_at_eta_one(sched):
    """Equation 4 with adjacent timesteps collapses to sqrt(beta_t (1-ab_s)/(1-ab_t)),
    the DDPM posterior std, which is what eta = 1 is supposed to mean."""
    for t in [1, 50, 400, 999]:
        ab_t = sched.alphas_cumprod[t]
        ab_s = sched.alphas_cumprod[t - 1]
        got = sampler.sigma_t_to_s(1.0, ab_t, ab_s)
        want = jnp.sqrt(sched.betas[t] * (1.0 - ab_s) / (1.0 - ab_t))
        assert float(got) == pytest.approx(float(want), rel=2e-4)


def test_sigma_is_linear_in_eta_and_vanishes_at_zero(sched):
    ab_t, ab_s = sched.alphas_cumprod[700], sched.alphas_cumprod[650]
    assert float(sampler.sigma_t_to_s(0.0, ab_t, ab_s)) == 0.0
    a = sampler.sigma_t_to_s(0.3, ab_t, ab_s)
    b = sampler.sigma_t_to_s(0.6, ab_t, ab_s)
    assert float(b) == pytest.approx(2.0 * float(a), rel=1e-6)


def test_step_coefficient_equation_6(sched):
    """delta x_s = c_i delta eps. The update is affine in eps, so this is exact."""
    key = jax.random.key(2)
    kx, ke, kd = jax.random.split(key, 3)
    x = jax.random.normal(kx, (32, 2))
    eps = jax.random.normal(ke, (32, 2))
    d = 1e-2 * jax.random.normal(kd, (32, 2))
    ab_t, ab_s = sched.alphas_cumprod[900], sched.alphas_cumprod[860]
    for eta in [0.0, 0.5, 1.0]:
        z = jnp.zeros_like(x)
        lo = sampler.ddim_step(x, eps, ab_t, ab_s, eta, z)
        hi = sampler.ddim_step(x, eps + d, ab_t, ab_s, eta, z)
        c = sampler.step_coefficient(ab_t, ab_s, eta)
        assert jnp.allclose(hi - lo, c * d, atol=1e-6)


def test_x0_sensitivity_equation_7_grows_toward_high_noise(sched):
    key = jax.random.key(3)
    kx, ke, kd = jax.random.split(key, 3)
    x = jax.random.normal(kx, (32, 2))
    eps = jax.random.normal(ke, (32, 2))
    d = 1e-2 * jax.random.normal(kd, (32, 2))
    for t in [10, 300, 700, 990]:
        ab = sched.alphas_cumprod[t]
        lo = sampler.predict_x0(x, eps, ab)
        hi = sampler.predict_x0(x, eps + d, ab)
        got = jnp.linalg.norm(hi - lo)
        want = sampler.x0_sensitivity(ab) * jnp.linalg.norm(d)
        assert float(got) == pytest.approx(float(want), rel=1e-3)
    ts = jnp.array([10, 300, 700, 990])
    sens = sampler.x0_sensitivity(sched.alphas_cumprod[ts])
    assert bool(jnp.all(jnp.diff(sens) > 0))


def test_sample_rejects_a_mismatched_eta_schedule(sched):
    with pytest.raises(ValueError):
        sampler.sample(jax.random.key(0), None, sched, 10, jnp.zeros(9),
                       jnp.zeros((4, 2)), eps_fn=lambda p, x, t, ab, k: x)


@pytest.fixture(scope="module")
def analytic_case(sched):
    mix = default_mixture()
    n = 1500
    k_init, k_ref = jax.random.split(jax.random.key(5))
    x_init = jax.random.normal(k_init, (n, 2))
    reference = mix.sample(k_ref, n)
    return mix, x_init, reference, sampler.analytic_eps_fn(mix)


@pytest.mark.parametrize("eta_name", ["deterministic", "ddpm_limit", "assert"])
def test_exact_score_reproduces_the_target_distribution(sched, analytic_case, eta_name):
    """With eps* in place of a network, every eta schedule in the DDIM family targets
    the same marginals, so all three must land on the mixture.

    The null for these statistics, two independent draws from the mixture itself at
    this sample size, is about 0.005 in energy distance and 0.15 in moment error.
    """
    mix, x_init, reference, eps_fn = analytic_case
    s = 100
    etas = {"deterministic": schedules.constant_schedule(s, 0.0),
            "ddpm_limit": schedules.constant_schedule(s, 1.0),
            "assert": schedules.assert_schedule(s)}[eta_name]
    out = sampler.sample(jax.random.key(7), None, sched, s, etas, x_init,
                         eps_fn=eps_fn).sample
    assert float(metrics.energy_distance(out, reference)) < 0.02
    assert float(metrics.moment_error(out, reference)) < 0.3


def test_deterministic_sampler_is_reproducible_and_key_independent(sched, analytic_case):
    mix, x_init, reference, eps_fn = analytic_case
    s = 20
    etas = schedules.constant_schedule(s, 0.0)
    a = sampler.sample(jax.random.key(0), None, sched, s, etas, x_init, eps_fn=eps_fn).sample
    b = sampler.sample(jax.random.key(99), None, sched, s, etas, x_init, eps_fn=eps_fn).sample
    assert jnp.array_equal(a, b)


def test_stochastic_sampler_depends_on_its_key(sched, analytic_case):
    mix, x_init, reference, eps_fn = analytic_case
    s = 20
    etas = schedules.assert_schedule(s)
    a = sampler.sample(jax.random.key(0), None, sched, s, etas, x_init, eps_fn=eps_fn).sample
    b = sampler.sample(jax.random.key(99), None, sched, s, etas, x_init, eps_fn=eps_fn).sample
    assert not jnp.allclose(a, b)


def test_trajectory_shape_and_final_state(sched, analytic_case):
    mix, x_init, reference, eps_fn = analytic_case
    s = 10
    tr = sampler.sample(jax.random.key(0), None, sched, s, schedules.assert_schedule(s),
                        x_init[:16], eps_fn=eps_fn)
    assert tr.x.shape == (s + 1, 16, 2)
    assert jnp.array_equal(tr.sample, tr.x[-1])
