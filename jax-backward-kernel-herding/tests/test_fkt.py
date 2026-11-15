"""Dyadic bits, FKT coreset size, and Gaussian square-root kernel tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from bkh.fkt import (
    coreset_size_from_bits,
    dyadic_bits,
    flexible_kernel_thinning,
    reconstructed_p,
)
from bkh.kernels import gaussian_kernel, gaussian_sqrt_kernel


@pytest.mark.parametrize("p", [0.3, 0.5, 0.75])
def test_dyadic_bits_match_greedy_expansion(p):
    tau = 1e-3
    bits, resid = dyadic_bits(p, tau=tau)
    rec = reconstructed_p(bits)
    assert abs(p - rec) <= tau + 1e-12
    assert abs(p - rec - resid) <= 1e-12
    # Greedy check: each bit is 1 iff the residual before it is >= 1/2^i.
    r = float(p)
    for i, b in enumerate(bits, start=1):
        thresh = 2.0 ** (-i)
        expect = 1 if r >= thresh else 0
        assert b == expect
        r = r - b * thresh
    assert r < tau or len(bits) >= 40


@pytest.mark.parametrize(
    "n,p",
    [(16, 0.5), (16, 0.75), (16, 0.25), (32, 0.3), (32, 0.5), (32, 0.75)],
)
def test_fkt_coreset_size_matches_bit_expansion(n, p):
    key = jax.random.PRNGKey(7)
    x = jax.random.normal(key, (n, 2), dtype=jnp.float64)
    tau = 1e-3
    coreset, bits = flexible_kernel_thinning(x, p=p, tau=tau, sigma=1.0, key=key)
    expect = coreset_size_from_bits(n, bits)
    assert int(coreset.size) == expect
    assert len(set(int(i) for i in coreset)) == int(coreset.size)


def test_gaussian_sqrt_kernel_quadrature_1d():
    sigma = 1.0
    # Quadrature grid for integral k_rt(x, z) k_rt(y, z) dz.
    z = jnp.linspace(-12.0, 12.0, 2401, dtype=jnp.float64)[:, None]
    dz = float(z[1, 0] - z[0, 0])
    pairs = [(0.0, 0.0), (0.0, 0.4), (-0.7, 1.1), (1.5, 1.5)]
    for a, b in pairs:
        x = jnp.asarray([[a]], dtype=jnp.float64)
        y = jnp.asarray([[b]], dtype=jnp.float64)
        krt_xz = gaussian_sqrt_kernel(x, z, sigma=sigma)[0]
        krt_yz = gaussian_sqrt_kernel(y, z, sigma=sigma)[0]
        integral = float(jnp.sum(krt_xz * krt_yz) * dz)
        kxy = float(gaussian_kernel(x, y, sigma=sigma)[0, 0])
        rel = abs(integral - kxy) / max(abs(kxy), 1e-12)
        assert rel < 0.03, f"pair {(a, b)} rel err {rel} integral {integral} k {kxy}"
