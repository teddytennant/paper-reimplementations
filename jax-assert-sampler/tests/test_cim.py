"""Equations 1-2, and the persistence property they are wrapped in."""

import jax
import jax.numpy as jnp
import pytest

from assert_sampler import cim, denoiser, schedules
from assert_sampler.diffusion import linear_schedule
from assert_sampler.sampler import sample


@pytest.fixture(scope="module")
def params():
    return denoiser.init_params(jax.random.key(0), width=32, depth=3)


def test_weight_noise_variance_follows_equation_2(params):
    """sigma_W^2 = alpha_W |W| + b_W, checked against the empirical variance of many
    draws of the same entry."""
    noise = cim.CIMNoise(alpha_w=3e-3, b_w=1e-4)
    w = params[0]["w"]
    draw = jax.jit(lambda k: cim.sample_mapping(k, params, noise)[0]["w"])
    draws = jax.vmap(draw)(jax.random.split(jax.random.key(0), 4000))
    empirical = jnp.var(draws, axis=0)
    predicted = noise.alpha_w * jnp.abs(w) + noise.b_w
    ratio = empirical / predicted
    assert float(jnp.mean(ratio)) == pytest.approx(1.0, abs=0.03)
    assert float(jnp.max(jnp.abs(ratio - 1.0))) < 0.2
    assert float(jnp.mean(draws)) == pytest.approx(0.0, abs=2e-4)


def test_variance_grows_with_weight_magnitude(params):
    noise = cim.CIMNoise(alpha_w=1e-2, b_w=0.0)
    w = jnp.array([[0.01, 1.0]])
    tiny = cim.weight_sigma(w, noise)
    assert float(tiny[0, 1]) == pytest.approx(10.0 * float(tiny[0, 0]), rel=1e-5)


@pytest.mark.parametrize("level", [0.01, 0.03, 0.08])
@pytest.mark.parametrize("fraction", [0.0, 0.5, 1.0])
def test_calibrate_hits_the_requested_equivalent_level(params, level, fraction):
    noise = cim.calibrate(params, level, magnitude_fraction=fraction)
    assert cim.equivalent_level(params, noise) == pytest.approx(level, rel=1e-4)


def test_perturb_leaves_biases_digital(params):
    noise = cim.calibrate(params, 0.05)
    mapping = cim.sample_mapping(jax.random.key(1), params, noise)
    perturbed = cim.perturb(params, mapping)
    for p, q, m in zip(params, perturbed, mapping):
        assert jnp.array_equal(p["b"], q["b"])
        assert jnp.array_equal(q["w"], p["w"] + m["w"])
    assert not jnp.allclose(params[0]["w"], perturbed[0]["w"])


def test_layers_get_independently_sampled_maps(params):
    noise = cim.calibrate(params, 0.05)
    a = cim.sample_mapping(jax.random.key(1), params, noise)
    b = cim.sample_mapping(jax.random.key(2), params, noise)
    assert not jnp.allclose(a[0]["w"], b[0]["w"])
    flat0 = a[0]["w"].ravel()[:64]
    flat1 = a[1]["w"].ravel()[:64]
    corr = float(jnp.corrcoef(flat0, flat1)[0, 1])
    assert abs(corr) < 0.35


@pytest.fixture(scope="module")
def toy_run(params):
    sched = linear_schedule()
    x = jax.random.normal(jax.random.key(4), (64, 2))
    return sched, x


def _det(sched, params, x, noise, mapping, key, steps=12):
    etas = schedules.constant_schedule(steps, 0.0)
    return sample(key, params, sched, steps, etas, x, noise=noise, mapping=mapping).sample


def test_fixed_mapping_is_held_across_every_denoising_step(params, toy_run):
    """Section 3.1: the weight-perturbation map is fixed for a deployment and all of
    its steps. With eta = 0 and sigma_Y = 0 nothing else in the sampler is random,
    so changing the sampler key must not change a single output value."""
    sched, x = toy_run
    noise = cim.calibrate(params, 0.05)
    mapping = cim.sample_mapping(jax.random.key(1), params, noise)
    a = _det(sched, params, x, noise, mapping, jax.random.key(0))
    b = _det(sched, params, x, noise, mapping, jax.random.key(12345))
    assert jnp.array_equal(a, b)


def test_resampled_mapping_redraws_every_step(params, toy_run):
    """The control. With resample_per_step the map is drawn from the sampler key, so
    a different key gives a different trajectory even at eta = 0."""
    sched, x = toy_run
    noise = cim.calibrate(params, 0.05, resample_per_step=True)
    mapping = cim.sample_mapping(jax.random.key(1), params, noise)
    a = _det(sched, params, x, noise, mapping, jax.random.key(0))
    b = _det(sched, params, x, noise, mapping, jax.random.key(12345))
    assert not jnp.allclose(a, b)
    # and it ignores the mapping it was handed.
    other = cim.sample_mapping(jax.random.key(77), params, noise)
    c = _det(sched, params, x, noise, other, jax.random.key(0))
    assert jnp.array_equal(a, c)


def test_output_noise_is_redrawn_per_mvm(params, toy_run):
    """eps_Y in Equation 1 is redrawn for every MVM, so it must depend on the key even
    with a fixed mapping and eta = 0."""
    sched, x = toy_run
    noise = cim.CIMNoise(sigma_y=0.05)
    mapping = cim.zero_mapping(params)
    a = _det(sched, params, x, noise, mapping, jax.random.key(0))
    b = _det(sched, params, x, noise, mapping, jax.random.key(12345))
    assert not jnp.allclose(a, b)


def test_clean_noise_is_a_no_op(params, toy_run):
    sched, x = toy_run
    clean = cim.CIMNoise()
    assert clean.is_clean
    a = _det(sched, params, x, clean, None, jax.random.key(0))
    b = _det(sched, params, x, clean, cim.zero_mapping(params), jax.random.key(9))
    assert jnp.allclose(a, b, atol=1e-6)


def test_active_mask_switches_the_mapping_off(params, toy_run):
    """a_i in Equation 8. All zeros must reproduce clean inference exactly."""
    sched, x = toy_run
    steps = 12
    noise = cim.calibrate(params, 0.05)
    mapping = cim.sample_mapping(jax.random.key(1), params, noise)
    etas = schedules.constant_schedule(steps, 0.0)
    off = sample(jax.random.key(0), params, sched, steps, etas, x, noise=noise,
                 mapping=mapping, active=jnp.zeros(steps)).sample
    clean = sample(jax.random.key(0), params, sched, steps, etas, x).sample
    assert jnp.allclose(off, clean, atol=1e-6)
    on = sample(jax.random.key(0), params, sched, steps, etas, x, noise=noise,
                mapping=mapping, active=jnp.ones(steps)).sample
    assert not jnp.allclose(on, clean, atol=1e-4)
