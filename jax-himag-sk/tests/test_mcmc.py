"""Occupation measure vs exact Boltzmann; field sign; all finite."""

import jax
import jax.numpy as jnp
import numpy as np

from himag.mcmc import run_chain
from himag.sk import boltzmann_probs, hamiltonian, sample_goe
from himag.slice import enumerate_slice, n_plus, random_slice_spin


def _state_index(samples, states):
    """Map each ±1 vector to its row in ``states``."""
    samples = np.asarray(samples)
    states = np.asarray(states)
    # pack ±1 as bits
    bits_s = (samples > 0).astype(np.int64)
    bits_t = (states > 0).astype(np.int64)
    pow2 = 1 << np.arange(samples.shape[1])
    key_s = bits_s @ pow2
    key_t = bits_t @ pow2
    lookup = {int(k): i for i, k in enumerate(key_t)}
    return np.array([lookup[int(k)] for k in key_s])


def test_boltzmann_occupation_d6_k2():
    d, k, beta = 6, 2, 0.45
    key = jax.random.PRNGKey(20)
    kJ, kh, kx, kc = jax.random.split(key, 4)
    J = sample_goe(kJ, d)
    h = jax.random.normal(kh, (d,)) * 0.4
    states = enumerate_slice(d, k)
    pi = np.asarray(boltzmann_probs(states, J, h, beta))
    x0 = random_slice_spin(kx, d, k)
    samples = run_chain(kc, x0, J, h, beta, n_steps=25000)
    assert int(np.min(n_plus(samples))) == k
    assert int(np.max(n_plus(samples))) == k
    idx = _state_index(samples[2000:], states)
    n = len(idx)
    counts = np.bincount(idx, minlength=len(pi)).astype(np.float64)
    emp = counts / counts.sum()
    tv = 0.5 * np.abs(emp - pi).sum()
    assert tv < 0.18, f"TV={tv}"
    E = np.asarray(hamiltonian(states, J, h))
    exact_mean = float(pi @ E)
    emp_mean = float(np.mean(np.asarray(hamiltonian(samples[2000:], J, h))))
    se = float(np.std(np.asarray(hamiltonian(samples[2000:], J, h))) / np.sqrt(n))
    assert abs(emp_mean - exact_mean) < max(0.15, 8 * se)
    assert np.isfinite(emp_mean) and np.isfinite(exact_mean)


def test_field_sign_flips_aligned_magnetization():
    # Homogeneous h is constant on a slice; use a nonuniform field.
    d, k, beta = 7, 3, 1.2
    key = jax.random.PRNGKey(21)
    kJ, kx, k1, k2 = jax.random.split(key, 4)
    J = sample_goe(kJ, d)
    h = jax.random.normal(jax.random.PRNGKey(22), (d,))
    x0 = random_slice_spin(kx, d, k)
    s_pos = run_chain(k1, x0, J, h, beta, n_steps=8000)
    s_neg = run_chain(k2, x0, J, -h, beta, n_steps=8000)
    m_pos = float(np.mean(np.asarray(s_pos[1000:]) @ np.asarray(h)))
    m_neg = float(np.mean(np.asarray(s_neg[1000:]) @ np.asarray(h)))
    assert np.isfinite(m_pos) and np.isfinite(m_neg)
    # Flipping h should reverse the sign of E[⟨x, h⟩] (or at least reverse order).
    assert m_pos > m_neg
    assert m_pos > 0
    assert m_neg < 0


def test_samples_finite():
    d, k = 5, 2
    J = sample_goe(jax.random.PRNGKey(23), d)
    h = jnp.zeros((d,))
    x0 = random_slice_spin(jax.random.PRNGKey(24), d, k)
    xs = run_chain(jax.random.PRNGKey(25), x0, J, h, 0.3, n_steps=20)
    assert bool(jnp.all(jnp.isfinite(xs)))
    assert bool(jnp.all(jnp.isfinite(hamiltonian(xs, J, h))))
