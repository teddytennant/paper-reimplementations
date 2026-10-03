"""Classical checks for arXiv:2609.40277. No quantum circuits."""

from __future__ import annotations

import math
from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

import nuct

_IDENTITY_ATOL = 1e-12


def _max_abs(a, b) -> float:
    return float(jnp.max(jnp.abs(jnp.asarray(a) - jnp.asarray(b))))


def test_package_uses_float64():
    x = nuct.equispaced_nodes(4)
    assert x.dtype == jnp.float64
    assert nuct.nudft_matrix(nuct.nuct_times(x)).dtype == jnp.complex128


def test_definition_11_nodes_and_coefficients():
    N = 8
    k = jnp.arange(N, dtype=jnp.float64)
    expected = -1.0 + 2.0 * k / N
    assert _max_abs(nuct.equispaced_nodes(N), expected) == 0.0

    f = jnp.asarray(
        [0.3, -1.25, 0.5, 2.0, -0.125, 0.75, -2.5, 1.0], dtype=jnp.float64
    )
    direct = nuct.unnormalized_coefficients(f)
    x = nuct.equispaced_nodes(N)
    for j in range(N):
        acc = 0.0
        for node in range(N):
            acc += float(f[node]) * float(nuct.chebyshev_T(j, x[node]))
        assert abs(float(direct[j]) - acc) < 1e-12

    via_matrix = nuct.coefficients_from_matrix(f)
    assert _max_abs(direct, via_matrix) < 1e-12
    C = nuct.chebyshev_matrix(N)
    assert C.dtype == jnp.float64
    assert _max_abs(direct, math.sqrt(N) * (C @ f)) < 1e-12
    # Eq. (3): (C_N)_jk = T_j(x_k) / sqrt(N)
    assert _max_abs(C, nuct.chebyshev_T(jnp.arange(N)[:, None], x[None, :]) / math.sqrt(N)) < 1e-15


def test_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        nuct.equispaced_nodes(6)
    with pytest.raises(ValueError):
        nuct.equispaced_nodes(2)


@pytest.mark.parametrize("N", [4, 8, 16])
def test_theorem_51_identity(N):
    t, t_minus = nuct.theorem_51_nodes(N)
    assert float(t[0]) == 0.5
    assert float(jnp.min(t)) > 0.0
    assert float(jnp.max(t)) == 0.5
    assert _max_abs(t_minus, jnp.mod(-t, 1.0)) == 0.0
    # t = 1/2 reflects to itself; every other negative node is 1 - t.
    assert float(t_minus[0]) == 0.5
    assert _max_abs(t_minus[1:], 1.0 - t[1:]) < 1e-15

    C = nuct.chebyshev_matrix(N)
    average = nuct.nuct_from_nufft(N)
    err = _max_abs(C, average)
    assert err < _IDENTITY_ATOL, err
    # The average of the two complex factors is the real cosine matrix.
    assert float(jnp.max(jnp.abs(jnp.imag(average)))) < _IDENTITY_ATOL

    # Endpoint column: T_j(-1) = (-1)^j, and F_t at t=1/2 has the same phase.
    column = ((-1.0) ** jnp.arange(N)) / math.sqrt(N)
    assert _max_abs(C[:, 0], column) < 1e-15


def test_unitary_dft_sign_and_fft():
    N = 8
    F = nuct.unitary_dft(N)
    # Negative exponent: F_11 = N^{-1/2} exp(-2π i / N) has negative imaginary part.
    assert float(F[1, 1].imag) < 0.0
    via_fft = jnp.fft.fft(jnp.eye(N, dtype=jnp.float64), axis=0, norm="ortho")
    assert _max_abs(F, via_fft) < 1e-12


def test_split_tie_convention_is_half_away():
    assert nuct.TIE_CONVENTION == "half_away_from_zero"
    # N=8, m=4, a=15, τ=15/16 = 1 - 1/(2N). Nτ = 7.5, a tie.
    # Half away from zero selects sunw=8, σ=0, z=-1. Half toward zero would
    # select sunw=7, z=+1. The test pins the away-from-zero choice.
    split = nuct.split_fixed_point(a=15, m=4, q=3)
    assert split.sunw == 8
    assert split.sigma == 0
    assert split.z_numerator == -1 and split.z_denominator == 1
    assert split.z == -1.0
    assert abs(split.z) <= 1.0
    assert split.reconstructed_N_tau() == split.N * split.tau_fraction()


def test_split_wraps_when_tau_is_within_half_bin_of_one():
    # Strictly inside (1 - 1/(2N), 1): not a tie. N=8, m=5, a=31, τ=31/32.
    # Distance to 1 is 1/32 < 1/16 = 1/(2N). Nτ = 7.75, sunw=8, σ=0, z=-1/2.
    inside = nuct.split_fixed_point(a=31, m=5, q=3)
    assert inside.sunw == 8
    assert inside.sigma == 0
    assert inside.z == -0.5
    assert inside.reconstructed_N_tau() == Fraction(8 * 31, 32)

    # Just outside the interval: Nτ = 7.25, sunw=7, σ=7.
    outside = nuct.split_fixed_point(a=29, m=5, q=3)
    assert outside.sunw == 7
    assert outside.sigma == 7
    assert outside.z == 0.5

    # τ=1 itself, allowed as a boundary, is an integer multiple: z=0, σ=0.
    endpoint = nuct.split_fixed_point(a=32, m=5, q=3)
    assert endpoint.sunw == 8
    assert endpoint.sigma == 0
    assert endpoint.z == 0.0


def test_split_integer_identity_and_phase():
    cases = []
    for q, m in [(3, 2), (3, 3), (3, 4), (3, 6), (4, 2), (5, 8), (2, 6)]:
        for a in range(0, (1 << m) + 1):
            cases.append((a, m, q))
    # A few larger nodes, still inside the float64 range.
    rng = np.random.default_rng(260940277)
    for q, m in [(6, 20), (8, 30)]:
        for a in rng.integers(0, (1 << m) + 1, size=40):
            cases.append((int(a), m, q))

    for a, m, q in cases:
        split = nuct.split_fixed_point(a, m, q)
        assert split.reconstructed_N_tau() == split.N * split.tau_fraction()
        assert 0 <= split.sigma < split.N
        assert abs(split.z) <= 1.0
        # Denominator records the paper's fractional width when there is one.
        if m > q:
            assert split.z_denominator == 1 << (m - q - 1)
        else:
            assert split.z == 0.0

    # Phase identity on a subset, every frequency. Integer a only: the split
    # never sees a float of τ.
    phase_cases = [
        (0, 6, 3),
        (1, 6, 3),
        (15, 4, 3),  # tie, wraps
        (31, 5, 3),  # strict wrap
        (29, 5, 3),
        (32, 5, 3),  # τ=1
        (7, 3, 3),  # m=q, z=0
        (3, 2, 4),  # m<q, z=0
        (1 << 19, 20, 6),
        (123456, 20, 6),
    ]
    for a, m, q in phase_cases:
        split = nuct.split_fixed_point(a, m, q)
        for k in range(split.N):
            left, right = nuct.split_phase_sides(k, split)
            err = abs(complex(left) - complex(right))
            assert err < 1e-12, (a, m, q, k, err)


def test_kernel_truncation_falls_with_rank():
    grid = jnp.linspace(-1.0, 1.0, 33)
    errors = []
    for K in (4, 8, 12):
        coeffs = nuct.bessel_coeff_table(K)
        # Odd parity is exactly zero; that is part of the formula, not a cutoff.
        for ell in range(K):
            for r in range(K):
                if (ell + r) % 2:
                    assert coeffs[ell, r] == 0
        exact = nuct.interaction_kernel(grid[None, :], grid[:, None])
        approx = nuct.truncated_kernel(grid, grid, coeffs)
        errors.append(float(jnp.max(jnp.abs(exact - approx))))
    assert errors[0] > errors[1] > errors[2]
    # Modest rank is already small. K=12 was ~1e-10 on this grid; allow margin.
    assert errors[1] < 1e-4
    assert errors[2] < 1e-8
    # Endpoints are included in the grid, so this is not an interior-only check.
    assert float(grid[0]) == -1.0 and float(grid[-1]) == 1.0


def test_frequency_rescaling_identity():
    N = 8
    for k in range(N):
        for z in (-1.0, -0.5, 0.0, 0.3, 1.0):
            left, right = nuct.frequency_phase_rescaling(k, z, N)
            assert abs(complex(left) - complex(right)) < 1e-12


@pytest.mark.parametrize("N", [8, 16])
def test_factorized_nudft_approaches_stored_nodes(N):
    stored = nuct.store_nuct_nodes(N, m=16)
    # Algebraic split on the stored integers, then the float phase identity.
    for split in stored["splits"]:
        assert split.reconstructed_N_tau() == split.N * split.tau_fraction()
    target = nuct.nudft_matrix(stored["tau"])
    entry_errors = []
    operator_errors = []
    for K in (4, 8, 12):
        approx = nuct.factorized_nudft(stored["z"], stored["sigma"], K)
        entry_errors.append(_max_abs(approx, target))
        operator_errors.append(float(jnp.linalg.norm(approx - target, ord=2)))
    assert entry_errors[0] > entry_errors[1] > entry_errors[2]
    assert operator_errors[0] > operator_errors[1] > operator_errors[2]
    assert entry_errors[2] < 1e-8
    assert operator_errors[2] < 1e-8
    # The truncation is matching stored τ, not merely sitting near exact t.
    # Node error at m=16 is orders of magnitude above the K=12 residual.
    node_gap = nuct.nudft_operator_distance(stored["t"], stored["tau"])
    assert node_gap > 1e-6
    assert operator_errors[2] < node_gap * 1e-3


def test_factorized_nudft_is_the_stated_product():
    N, K = 8, 6
    stored = nuct.store_nuct_nodes(N, m=12)
    coeffs = nuct.bessel_coeff_table(K)
    F = nuct.unitary_dft(N)
    M = nuct.embedding_matrix(stored["sigma"], N)
    v = nuct.frequency_weights(N, K)
    u = nuct.node_amplitudes(stored["z"], coeffs)
    manual = jnp.zeros((N, N), dtype=jnp.complex128)
    for r in range(K):
        manual = manual + jnp.diag(v[r]) @ F @ M @ jnp.diag(u[r])
    got = nuct.factorized_nudft(stored["z"], stored["sigma"], K, coeffs=coeffs)
    assert _max_abs(manual, got) < 1e-12
    # One 1 per column, in the row the split named.
    assert int(jnp.sum(M)) == N
    for j, sigma_j in enumerate(stored["sigma"]):
        column = M[:, j]
        assert int(jnp.sum(column)) == 1
        assert float(column[int(sigma_j)]) == 1.0


def test_conditioning_free_at_the_endpoint():
    z = 1.0 - 1e-12
    ell = 5
    delta_phi = 1e-8
    record = nuct.output_rounding_chebyshev_error(z, ell, delta_phi)
    error = float(record["error"])
    angle_error = float(record["angle_error"])
    bound = float(record["bound"])
    assert bound == ell * delta_phi
    assert angle_error <= delta_phi + 1e-15
    assert error <= bound
    # Worst-case output perturbation of size exactly δ_φ, still no κ.
    worst = float(nuct.worst_case_angle_chebyshev_error(z, ell, delta_phi))
    assert worst <= bound + 1e-15

    kappa_scaled = float(nuct.kappa_scaled_input_error(z, delta_phi))
    # κ(1 - 1e-12) is about 7e5. The input-error factor is orders of
    # magnitude above both δ_φ and the angle-rounding Chebyshev error.
    assert kappa_scaled > 1e4 * delta_phi
    assert kappa_scaled > 1e3 * max(error, 1e-30)
    assert kappa_scaled / bound > 1e4

    # Same absolute input step, realized through arccos, is also far larger
    # than the output-rounding tolerance.
    realized = nuct.realized_input_angle_error(z, delta_phi)
    assert realized > 1e3 * delta_phi

    # The same bound holds at the other endpoint and does not grow with κ.
    z_neg = -1.0 + 1e-12
    neg = nuct.output_rounding_chebyshev_error(z_neg, ell, delta_phi)
    assert float(neg["error"]) <= ell * delta_phi
    assert float(nuct.kappa_scaled_input_error(z_neg, delta_phi)) > 1e4 * delta_phi

    # Interior: κ is 1, so the blow-up is an endpoint effect, not a constant.
    assert abs(float(nuct.arccos_kappa(0.0)) - 1.0) < 1e-12
    interior = nuct.output_rounding_chebyshev_error(0.0, ell, delta_phi)
    assert float(interior["error"]) <= ell * delta_phi
    assert float(nuct.kappa_scaled_input_error(0.0, delta_phi)) == pytest.approx(delta_phi)


def test_node_error_operator_bound_holds_and_is_not_vacuous():
    rng = np.random.default_rng(260940277)
    for N in (8, 16):
        t, _ = nuct.theorem_51_nodes(N)
        delta = rng.uniform(-1e-4, 1e-4, size=N)
        tau = jnp.clip(t + jnp.asarray(delta), 0.0, 1.0 - 1e-15)
        linf = float(jnp.max(jnp.abs(t - tau)))
        lhs = nuct.nudft_operator_distance(t, tau)
        rhs = nuct.node_error_operator_bound(N, linf)
        assert linf > 0.0
        assert lhs > 1e-6, lhs
        assert lhs <= rhs, (lhs, rhs)
        # The prefactor is the one in the brief, not a looser stand-in.
        expected = (2.0 * math.pi / math.sqrt(3.0)) * (N ** 1.5) * linf
        assert rhs == pytest.approx(expected)


def test_row_oracle_matches_embedding_and_norm():
    sigma = [0, 0, 1, 2]
    N = 4
    assert nuct.row_support(sigma, 0) == (0, 1)
    assert nuct.row_support(sigma, 1) == (2,)
    assert nuct.row_support(sigma, 2) == (3,)
    assert nuct.row_support(sigma, 3) == ()
    assert nuct.multiplicities(sigma, N) == (2, 1, 1, 0)
    assert nuct.max_multiplicity(sigma, N) == 2
    assert nuct.embedding_spectral_norm(sigma, N) == pytest.approx(math.sqrt(2.0))
    numerical = nuct.numerical_embedding_spectral_norm(sigma, N)
    assert numerical == pytest.approx(math.sqrt(2.0), abs=1e-12)
    M = nuct.embedding_matrix(sigma, N)
    for ell in range(N):
        cols = nuct.row_support(sigma, ell)
        assert [j for j in range(len(sigma)) if float(M[ell, j]) == 1.0] == list(cols)


@pytest.mark.parametrize("N", [16, 32])
def test_proposition_55_multiplicity_on_equispaced_nodes(N):
    q = nuct.nuct_order(N)
    # Two extra bits past the bin-size threshold, so a float ulp in arccos
    # cannot push the realized error out of the hypothesis.
    m = nuct.minimum_node_bits(q) + 2
    stored = nuct.store_nuct_nodes(N, m)
    assert nuct.collision_hypothesis_holds(stored["t"], stored["tau"], N), stored["linf_error"]
    # Paper's quantization promise, checked rather than assumed.
    assert stored["linf_error"] <= 2.0 ** (-m)

    sigma = stored["sigma"]
    d = nuct.multiplicities(sigma, N)
    d_tau = nuct.max_multiplicity(sigma, N)
    cap = min(nuct.MULTIPLICITY_CAP, N)
    assert d_tau <= cap, (N, m, d_tau, d)
    assert d_tau >= 1
    closed_form = nuct.embedding_spectral_norm(sigma, N)
    numerical = nuct.numerical_embedding_spectral_norm(sigma, N)
    assert closed_form == pytest.approx(math.sqrt(d_tau))
    assert numerical == pytest.approx(closed_form, abs=1e-12)

    # One 1 per column, and the oracle is that column's row.
    seen = []
    for ell in range(N):
        cols = nuct.row_support(sigma, ell)
        seen.extend(cols)
        if cols:
            assert list(cols) == list(range(cols[0], cols[-1] + 1))
        assert len(cols) == d[ell]
    # Rows are not visited in column order; the oracle must still partition the columns.
    assert sorted(seen) == list(range(N))
    assert len(seen) == N

    # Lemma 5.4, classical half: positive nodes do not wrap, σ is
    # non-increasing, and rows above N/2 are empty.
    sigma_list = [int(s) for s in list(sigma)]
    assert all(s <= N // 2 for s in sigma_list)
    assert sigma_list == sorted(sigma_list, reverse=True)
    for ell in range(N // 2 + 1, N):
        assert nuct.row_support(sigma, ell) == ()
    # k=0 is the exact node τ=1/2, so it occupies row N/2.
    assert int(sigma_list[0]) == N // 2
    assert 0 in nuct.row_support(sigma, N // 2)
