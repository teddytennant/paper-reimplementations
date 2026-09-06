"""The sieve MLE of eq (3.9) on synthetic data.

No rate is asserted here.  The paper's rates are asymptotic in n with
logarithmic factors and a constant nobody has pinned down, and n is small enough
that a fitted exponent would be noise.  What is asserted is the direction: the
median error over three seeds is smaller with eight times the data.  The medians
below were measured before being written down.
"""

import statistics

import jax
import jax.numpy as jnp
import pytest

from mkv_sieve import (
    OUPotential,
    fit_mle,
    ise_on_interval,
    kl_stationary,
    profile_a,
    sample_particles,
    solve_profile,
)
from mkv_sieve.mle import nll
from mkv_sieve.sieve import domain_bound, init_requ_params, make_xi
from mkv_sieve.truth import ENVELOPE, TRUTH

WHICH = 3
D = 2
KAPPA, K = ENVELOPE[WHICH]
XI0 = TRUTH[WHICH]
STEPS = 500
SEEDS = (0, 1, 2)


def setup():
    pot = OUPotential(D)
    prof0 = solve_profile(XI0, pot, KAPPA, K)
    return pot, prof0, domain_bound(pot, K), float(profile_a(prof0))


def one_fit(pot, prof0, U, n, seed, width=3):
    k_data, k_init = jax.random.split(jax.random.PRNGKey(seed))
    x = sample_particles(k_data, prof0, pot, n)
    params0 = init_requ_params(k_init, width=width, U=U)
    params, losses = fit_mle(
        params0, pot.V(x), pot, KAPPA, K, U=U, steps=STEPS, lr=0.02
    )
    return params, losses


def test_more_data_helps():
    """Median root-ISE on I over three seeds is lower at n = 4000 than at n = 500.

    I = [0.05 a_0, 0.9 a_0], the interior interval of Remark 3.6 and the grid the
    paper's Section 8 reports ISE on.  Measured medians over seeds 0, 1, 2:
    0.0818 at n = 500 and 0.0170 at n = 4000.  The per-seed spread is wide, which
    is why this is a median and not a single-seed threshold.
    """
    pot, prof0, U, a0 = setup()
    r_lo, r_hi = 0.05 * a0, 0.9 * a0

    def median_err(n):
        out = []
        for seed in SEEDS:
            params, _ = one_fit(pot, prof0, U, n, seed)
            xi_hat = make_xi(params, KAPPA, K, U=U)
            out.append(float(jnp.sqrt(ise_on_interval(xi_hat, XI0, r_lo, r_hi))))
        return statistics.median(out)

    small = median_err(500)
    large = median_err(4000)
    assert large < small
    assert large < 0.05


def test_kl_to_the_truth_shrinks_with_more_data():
    """KL(pi_0 || pi_hat), the loss Section 5 actually bounds, over the same seeds."""
    pot, prof0, U, _ = setup()

    def median_kl(n):
        out = []
        for seed in SEEDS:
            params, _ = one_fit(pot, prof0, U, n, seed)
            prof_hat = solve_profile(make_xi(params, KAPPA, K, U=U), pot, KAPPA, K)
            out.append(float(kl_stationary(prof0, prof_hat, pot)))
        return statistics.median(out)

    assert median_kl(4000) < median_kl(500)


def test_optimisation_makes_progress_and_stays_finite():
    pot, prof0, U, _ = setup()
    params, losses = one_fit(pot, prof0, U, 2000, 1)
    assert bool(jnp.all(jnp.isfinite(losses)))
    assert float(losses[-1]) < float(losses[0])


def test_gradient_of_the_likelihood_matches_finite_differences():
    """The forward map is differentiated end to end, root find included."""
    pot, prof0, U, _ = setup()
    x = sample_particles(jax.random.PRNGKey(3), prof0, pot, 400)
    energies = pot.V(x)
    params = init_requ_params(jax.random.PRNGKey(4), width=3, U=U)

    def loss(w):
        p = [(params[0][0], params[0][1]), (w, params[1][1])]
        return nll(p, energies, pot, KAPPA, K, U=U)

    w0 = params[1][0]
    g = jax.grad(loss)(w0)
    eps = 1e-6
    for j in range(w0.shape[1]):
        e = jnp.zeros_like(w0).at[0, j].set(eps)
        fd = (loss(w0 + e) - loss(w0 - e)) / (2 * eps)
        assert float(g[0, j]) == pytest.approx(float(fd), rel=1e-5, abs=1e-9)


def test_cli_runs(capsys):
    from mkv_sieve.fit import main

    main(["--n", "300", "--steps", "20", "--d", "2", "--xi", "2"])
    out = capsys.readouterr().out
    assert "KL(pi_0||pi_hat)" in out
    assert "nan" not in out.lower()
