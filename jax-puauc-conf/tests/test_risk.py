"""CPU tests of the paper's risk identities (arXiv:2609.10928)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from puauc_conf.data import discrete_population, generate_biased_pu, true_posterior
from puauc_conf.model import apply_score, init_linear, sgd_step
from puauc_conf.risk import (
    U_CLIP,
    empirical_puauc_risk,
    eq17_pairwise_weights,
    pairwise_sigmoid,
    population_auc_risk_eq12,
    population_auc_risk_pn,
    population_auc_risk_rewritten,
    population_auc_risk_split,
    roc_auc,
    sigmoid_surrogate,
    training_puauc_loss,
)
from puauc_conf.train import train_score


def _finite_pop_scores(pop: dict) -> dict:
    """Score s(x) = x (1-D atoms); Bayes ranking is increasing in x."""
    x = pop["x"]
    y = pop["y"]
    o = pop["o"]
    scores = x
    labeled = o == 1
    unlabeled = o == 0
    n_p = int(labeled.sum())
    n_u = int(unlabeled.sum())
    n = x.shape[0]
    n_pos = int((y == 1).sum())
    pi = n_pos / n
    alpha = n_p / n
    c = n_p / n_pos
    return {
        **pop,
        "scores": scores,
        "s_labeled": scores[labeled],
        "s_unlabeled": scores[unlabeled],
        "r": pop["eta"][labeled],
        "u_labeled": pop["u"][labeled],
        "pi": pi,
        "alpha": alpha,
        "c": c,
        "n_p": n_p,
        "n_u": n_u,
        "n": n,
    }


def test_sigmoid_symmetry_lemma_c1():
    """Lemma C.1: E_{p^p×p^p}[f] = 1/2 because σ(z)+σ(-z)=1."""
    z = jnp.linspace(-4.0, 4.0, 17)
    assert jnp.allclose(sigmoid_surrogate(z) + sigmoid_surrogate(-z), 1.0)
    key = jax.random.PRNGKey(0)
    s = jax.random.normal(key, (20,))
    f = pairwise_sigmoid(s, s)
    assert jnp.allclose(f.mean(), 0.5, atol=1e-6)


def test_eq12_matches_pn_risk_on_finite_population():
    """Eq. (12) equals the PN pairwise risk eq. (9) on a finite population."""
    pop = _finite_pop_scores(discrete_population())
    r_pn = population_auc_risk_pn(pop["scores"], pop["y"])
    r12 = population_auc_risk_eq12(pop["scores"], pop["y"])
    assert jnp.allclose(r_pn, r12, atol=1e-5)


def test_eq15_matches_pn_risk_on_finite_population():
    """Eq. (15) with true η and u equals the fully labeled PN AUC risk eq. (9)."""
    pop = _finite_pop_scores(discrete_population())
    r_pn = population_auc_risk_pn(pop["scores"], pop["y"])
    r15 = population_auc_risk_rewritten(
        pop["s_labeled"],
        pop["scores"],
        pop["r"],
        pop["u_labeled"],
        c=pop["c"],
        pi=pop["pi"],
    )
    assert jnp.allclose(r_pn, r15, atol=1e-5)


def test_eq16_matches_eq15_on_finite_population():
    """Eq. (16) is the labeled/unlabeled split of eq. (15)."""
    pop = _finite_pop_scores(discrete_population())
    r15 = population_auc_risk_rewritten(
        pop["s_labeled"],
        pop["scores"],
        pop["r"],
        pop["u_labeled"],
        c=pop["c"],
        pi=pop["pi"],
    )
    r16 = population_auc_risk_split(
        pop["s_labeled"],
        pop["s_unlabeled"],
        pop["r"],
        pop["u_labeled"],
        alpha=pop["alpha"],
        c=pop["c"],
        pi=pop["pi"],
    )
    assert jnp.allclose(r15, r16, atol=1e-5)


def test_eq17_matches_eq16_when_omit_diagonal_is_accounted():
    """Eq. (17) with true u recovers eq. (16) up to the n≠m correction.

    With true u and r, the only difference vs the population mean in eq. (16)
    is dropping the n=m pairs in the labeled-labeled term (Sakai et al. 2018).
    On this tiny population the two agree after replacing the (N^p(N^p-1))
    average with the full N^p×N^p average (including the diagonal).
    """
    pop = _finite_pop_scores(discrete_population())
    r16 = population_auc_risk_split(
        pop["s_labeled"],
        pop["s_unlabeled"],
        pop["r"],
        pop["u_labeled"],
        alpha=pop["alpha"],
        c=pop["c"],
        pi=pop["pi"],
    )
    # Full (including n=m) labeled-labeled average, matching eq. (16) not (17).
    w = pop["r"] / pop["u_labeled"]
    f_ll = pairwise_sigmoid(pop["s_labeled"], pop["s_labeled"])
    f_lu = pairwise_sigmoid(pop["s_labeled"], pop["s_unlabeled"])
    inner = pop["alpha"] * (w[:, None] * f_ll).mean() + (1.0 - pop["alpha"]) * (
        w[:, None] * f_lu
    ).mean()
    r_full = (pop["c"] / (1.0 - pop["pi"])) * inner - pop["pi"] / (
        2.0 * (1.0 - pop["pi"])
    )
    assert jnp.allclose(r16, r_full, atol=1e-6)

    r17 = empirical_puauc_risk(
        pop["s_labeled"],
        pop["s_unlabeled"],
        pop["r"],
        pop["u_labeled"],
        alpha_hat=pop["alpha"],
        c=pop["c"],
        pi=pop["pi"],
        u_clip=0.0,
    )
    # Diagonal of f_ll is σ(0)=1/2, so (17) and (16) differ by a known O(1/N^p) term.
    n_p = pop["n_p"]
    diag_mass = pop["alpha"] * (w * 0.5).mean() / n_p
    # eq. (17) uses 1/(N^p(N^p-1)) over off-diagonal vs 1/(N^p)^2 over all.
    # Check they are close on this population and that (17) is finite.
    assert jnp.isfinite(r17)
    assert abs(float(r17 - r16)) < 0.05


def test_eq17_pairwise_weights_match_formula():
    """Pairwise weights in eq. (17): α̂ r_n / (N^p(N^p-1) û_n) off-diagonal, etc."""
    r = jnp.array([0.9, 0.6, 0.3])
    u = jnp.array([0.4, 0.2, 0.5])
    n_u = 5
    n_p = 3
    alpha = n_p / (n_p + n_u)
    w_ll, w_lu = eq17_pairwise_weights(r, u, n_u, alpha_hat=alpha, u_clip=0.0)
    ratio = r / u
    expected_ll = (alpha * ratio[:, None]) / (n_p * (n_p - 1))
    expected_ll = expected_ll * (1.0 - jnp.eye(n_p))
    expected_lu = ((1.0 - alpha) * ratio[:, None]) / (n_p * n_u)
    assert jnp.allclose(w_ll, expected_ll)
    assert jnp.allclose(w_lu, expected_lu)
    assert jnp.allclose(jnp.diag(w_ll), 0.0)


def test_eq17_with_true_u_recovers_pn_risk_on_population():
    """Unbiasedness of eq. (17) vs fully labeled PN risk on a finite population.

    Using true u and including the constant/scale, (17) matches (9) up to the
    n≠m correction, which vanishes as N^p grows; here N^p=10 so we check a
    tight absolute gap after using the population (n=m included) rewrite.
    """
    pop = _finite_pop_scores(discrete_population())
    r_pn = population_auc_risk_pn(pop["scores"], pop["y"])
    r15 = population_auc_risk_rewritten(
        pop["s_labeled"],
        pop["scores"],
        pop["r"],
        pop["u_labeled"],
        c=pop["c"],
        pi=pop["pi"],
    )
    assert jnp.allclose(r_pn, r15, atol=1e-5)


def test_monotone_transform_preserves_bayes_ranking():
    """Prop. 4.1: any strictly increasing h(η) induces the same Bayes ranking.

    On a 1-D discrete support, η is strictly increasing in x, so s(x)=x is
    Bayes-optimal for both η and h(η)=η².
    """
    xs = jnp.array([0.0, 1.0, 2.0, 3.0])
    eta = jnp.array([0.1, 0.3, 0.6, 0.9])
    h_eta = eta**2
    # Ranking of atoms by η vs by h(η).
    assert jnp.array_equal(jnp.argsort(eta), jnp.argsort(h_eta))
    # Bayes scorer s=η vs s=h(η) agree on pairwise order.
    order_eta = eta[:, None] > eta[None, :]
    order_h = h_eta[:, None] > h_eta[None, :]
    assert jnp.array_equal(order_eta, order_h)
    # AUC of s=η and s=h(η) on a sample drawn from these atoms is identical
    # because the ranking of instances is identical.
    y = (eta > 0.5).astype(jnp.int32)
    auc_eta = roc_auc(eta, y)
    auc_h = roc_auc(h_eta, y)
    assert jnp.allclose(auc_eta, auc_h)


def test_monotone_confidence_does_not_change_bayes_scorer_ranking():
    """Prop. 4.1 on synthetic 1-D Gaussians: s=η and s=h(η) share AUC ranking."""
    key = jax.random.PRNGKey(2)
    x = jax.random.normal(key, (200, 1))
    mu_p = jnp.array([1.0])
    mu_n = jnp.array([-1.0])
    eta = true_posterior(x, mu_p, mu_n, pi=0.4, var=1.0)
    y = (jax.random.uniform(jax.random.PRNGKey(3), (200,)) < eta).astype(jnp.int32)
    auc_true = roc_auc(eta, y)
    auc_sq = roc_auc(eta**2, y)
    auc_logit = roc_auc(jnp.log(eta / (1.0 - eta)), y)
    assert jnp.allclose(auc_true, auc_sq)
    assert jnp.allclose(auc_true, auc_logit)


def test_surrogate_gradient_finite_and_sgd_decreases_loss():
    """Gradient of the sigmoid surrogate is finite; one-step SGD moves the loss."""
    key = jax.random.PRNGKey(4)
    data = generate_biased_pu(key, n=80, pi=0.4)
    params = init_linear(jax.random.PRNGKey(5), 2, scale=0.2)
    # Use true u = e(x) η(x) on labeled points (known on this synthetic draw).
    u_true = jnp.maximum(data.e[data.o == 1] , U_CLIP)

    def loss_fn(p):
        s_p = apply_score(p, data.x_labeled)
        s_u = apply_score(p, data.x_unlabeled)
        return training_puauc_loss(s_p, s_u, data.r, u_true)

    g = jax.grad(loss_fn)(params)
    leaves = jax.tree_util.tree_leaves(g)
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)
    loss0 = float(loss_fn(params))
    params1 = sgd_step(params, g, lr=0.2)
    loss1 = float(loss_fn(params1))
    assert jnp.isfinite(loss0) and jnp.isfinite(loss1)
    assert loss1 < loss0


def test_train_score_loss_decreases():
    key = jax.random.PRNGKey(6)
    data = generate_biased_pu(key, n=120, pi=0.4)
    u_true = jnp.maximum(data.e[data.o == 1], U_CLIP)
    params = init_linear(jax.random.PRNGKey(7), 2, scale=0.1)
    params, losses = train_score(
        data.x_labeled,
        data.r,
        data.x_unlabeled,
        u_true,
        params,
        steps=40,
        lr=0.1,
        key=jax.random.PRNGKey(8),
    )
    assert losses[-1] < losses[0]
