"""CPU tests for (P1), the closed-form support map, and enumeration BnB."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from robust_sparse_mv.bnb import branch_and_bound_p1
from robust_sparse_mv.brute import brute_force_p1
from robust_sparse_mv.closed_form import (
    H_omega,
    Xi,
    assumption2_holds,
    dual_multiplier,
    f_beta,
    l0_norm,
    quadratic_risk,
    robust_constraint_value,
    singleton_ratios,
    support_objective,
    xi,
)
from robust_sparse_mv.problem import RobustSparseMV, synthetic_instance


def _problem(n: int = 6, **kwargs) -> RobustSparseMV:
    return synthetic_instance(n, **kwargs)


def test_closed_form_active_robust_constraint():
    problem = _problem(n=6, seed=1)
    omega = (0, 2, 4)
    w = xi(problem, omega)
    lhs = problem.r[jnp.array(omega)] @ w - problem.gamma * jnp.sqrt(w @ (problem.D[jnp.ix_(jnp.array(omega), jnp.array(omega))] @ w))
    rel = abs(float(lhs) - problem.r_bar) / problem.r_bar
    assert rel < 1e-5

    x = Xi(problem, omega)
    rel_full = abs(float(robust_constraint_value(problem, x)) - problem.r_bar) / problem.r_bar
    assert rel_full < 1e-5


def test_assumption2_implies_singleton_feasibility():
    problem = _problem(n=6, seed=2, gamma=0.04, margin=0.4)
    assert bool(assumption2_holds(problem))
    ratios = np.asarray(singleton_ratios(problem))
    assert np.min(ratios) > problem.gamma
    for i in range(problem.n):
        x = Xi(problem, (i,))
        lhs = float(robust_constraint_value(problem, x))
        assert lhs >= problem.r_bar - 1e-8 * (1.0 + problem.r_bar)


def test_h_omega_matches_definition_and_enters_xi():
    problem = _problem(n=5, seed=3)
    omega = (1, 3, 4)
    idx = jnp.array(omega)
    d_w = problem.D[jnp.ix_(idx, idx)]
    r_w = problem.r[idx]
    h = float(H_omega(problem.D, problem.r, omega))
    h_def = float(jnp.sqrt(r_w @ jnp.linalg.solve(d_w, r_w)))
    assert abs(h - h_def) < 1e-12
    w = xi(problem, omega)
    w_from_h = (problem.r_bar / (h * (h - problem.gamma))) * jnp.linalg.solve(d_w, r_w)
    np.testing.assert_allclose(np.asarray(w), np.asarray(w_from_h), rtol=1e-10, atol=1e-12)


def test_zero_padding_support_subset():
    problem = _problem(n=6, seed=4)
    omega = (0, 5)
    x = np.asarray(Xi(problem, omega))
    outside = [i for i in range(problem.n) if i not in omega]
    assert np.all(np.abs(x[outside]) == 0.0)
    assert np.any(np.abs(x[list(omega)]) > 0.0)


def test_brute_force_matches_bnb():
    problem = _problem(n=6, seed=5, beta=0.03)
    brute = brute_force_p1(problem)
    bnb = branch_and_bound_p1(problem, eps=1e-8)
    assert brute.n_xi_evals == (1 << problem.n) - 1
    assert bnb.f_beta <= brute.f_beta + 1e-6
    assert abs(bnb.f_beta - brute.f_beta) < 1e-6
    assert abs(float(f_beta(problem, bnb.x)) - bnb.f_beta) < 1e-6


def test_pruning_skips_right_subtree_and_stays_optimal():
    problem = _problem(n=6, seed=6, beta=0.15)
    brute = brute_force_p1(problem)
    bnb = branch_and_bound_p1(problem, extra_pruning=True, eps=1e-8)
    assert bnb.n_right_pruned >= 1
    assert bnb.n_xi_evals < (1 << problem.n) - 1
    assert abs(bnb.f_beta - brute.f_beta) < 1e-6

    constructed = False
    for line in bnb.history:
        if line.startswith("prune-right"):
            constructed = True
            break
    assert constructed


def test_dual_multiplier_formula():
    problem = _problem(n=5, seed=7)
    omega = (0, 1, 2)
    mu = float(dual_multiplier(problem, omega))
    h = float(H_omega(problem.D, problem.r, omega))
    mu_formula = 2.0 * problem.r_bar / (h - problem.gamma) ** 2
    assert abs(mu - mu_formula) < 1e-12


def test_f_beta_definition():
    problem = _problem(n=5, seed=8, beta=0.07)
    x = Xi(problem, (0, 2, 3))
    val = float(f_beta(problem, x))
    expected = float(quadratic_risk(x, problem.D) + problem.beta * l0_norm(x))
    assert abs(val - expected) < 1e-12
    w = xi(problem, (0, 2, 3))
    d_w = problem.D[jnp.ix_(jnp.array([0, 2, 3]), jnp.array([0, 2, 3]))]
    assert abs(float(support_objective(problem, (0, 2, 3))) - float(w @ (d_w @ w) + problem.beta * 3)) < 1e-12


def test_assumption1_positive_excess_target():
    problem = _problem(n=4, seed=9)
    assert problem.r_bar > 0.0
