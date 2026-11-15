import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import math

import jax
import jax.numpy as jnp

from roc_vus import (
    cross_fit_vus,
    label_vus,
    mixed_volume,
    plugin_vus,
    population_vus_mc,
    population_vus_ustat,
    u_statistic_g,
)


def _one_hot_balanced(n_per, n_classes):
    rows = []
    for class_id in range(n_classes):
        eye = jnp.zeros(n_classes, dtype=jnp.float64).at[class_id].set(1.0)
        rows.append(jnp.tile(eye, (n_per, 1)))
    return jnp.concatenate(rows, axis=0)


def _mann_whitney_auc(scores_pos, scores_neg):
    diff = scores_pos[:, None] - scores_neg[None, :]
    return jnp.mean((diff > 0).astype(jnp.float64) + 0.5 * (diff == 0).astype(jnp.float64))


def test_theorem_222_k2_matches_half_max_product():
    cases = [
        (1.5, 0.5, 0.5, 1.5),
        (2.0, 0.0, 0.0, 2.0),
        (1.0, 3.0, 4.0, 2.0),
        (0.0, 1.0, 1.0, 0.0),
        (0.2, 1.7, 3.5, 0.4),
    ]
    for a, b, c, d in cases:
        got = float(mixed_volume(jnp.array([[a, b], [c, d]], dtype=jnp.float64)))
        expected = 0.5 * max(a * d, b * c)
        assert abs(got - expected) < 1e-12
    # Example 2.23: rectangle areas 9/4 and 1/4, mixed volume 9/8.
    paper = float(mixed_volume(jnp.array([[1.5, 0.5], [0.5, 1.5]])))
    assert abs(paper - 9.0 / 8.0) < 1e-12


def test_theorem_222_k3_hand_computed():
    # Products: 45, 48, 72, 84, 96, 105. Maximum 105, volume 105/6.
    edges = jnp.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    assert abs(float(mixed_volume(edges)) - 17.5) < 1e-12

    # Identity is the unique maximizer: 5*5*5 = 125, volume 125/6.
    diagonal = jnp.array(
        [
            [5.0, 1.0, 1.0],
            [1.0, 5.0, 1.0],
            [1.0, 1.0, 5.0],
        ]
    )
    assert abs(float(mixed_volume(diagonal)) - 125.0 / 6.0) < 1e-12


def test_formula1_perfect_and_chance_exact():
    class_1 = jnp.array([2.0, 0.0])
    class_2 = jnp.array([0.0, 2.0])
    perfect_tuples = jnp.stack(
        [
            jnp.stack([class_1, class_1]),
            jnp.stack([class_2, class_2]),
            jnp.stack([class_1, class_2]),
            jnp.stack([class_2, class_1]),
        ]
    )
    perfect = float(population_vus_mc(perfect_tuples))
    assert abs(perfect - 1.0) < 1e-8

    chance_tuples = jnp.ones((6, 2, 2), dtype=jnp.float64)
    chance = float(population_vus_mc(chance_tuples))
    assert abs(chance - 0.5) < 1e-8

    chance_rows = jnp.ones((12, 2), dtype=jnp.float64)
    chance_u = float(population_vus_ustat(chance_rows))
    assert abs(chance_u - 0.5) < 1e-8

    # Binary AUC on the same constructions.
    perfect_auc = float(_mann_whitney_auc(jnp.array([1.0, 1.0]), jnp.array([0.0, 0.0])))
    chance_auc = float(_mann_whitney_auc(jnp.ones(8), jnp.ones(8)))
    assert abs(perfect - perfect_auc) < 1e-8
    assert abs(chance - chance_auc) < 1e-8


def test_formula1_intermediate_auc_075():
    # Equal priors. Class 1 puts mass 0.75 on point a. Class 2 puts mass 0.75 on b.
    # q(a) = (1.5, 0.5), q(b) = (0.5, 1.5). Likelihood ratio is 3 at a and 1/3 at b.
    # Mann-Whitney AUC = 0.75 * 0.75 + 0.5 * 0.375 = 0.75.
    # Formula (1) on the four equally likely mixture pairs is also 0.75.
    q_a = jnp.array([1.5, 0.5])
    q_b = jnp.array([0.5, 1.5])
    enumerated = jnp.stack(
        [
            jnp.stack([q_a, q_a]),
            jnp.stack([q_b, q_b]),
            jnp.stack([q_a, q_b]),
            jnp.stack([q_b, q_a]),
        ]
    )
    population = float(population_vus_mc(enumerated))
    assert abs(population - 0.75) < 1e-12

    p_pos_gt = 0.75 * 0.75
    p_tie = 0.75 * 0.25 + 0.25 * 0.75
    auc = p_pos_gt + 0.5 * p_tie
    assert abs(auc - 0.75) < 1e-12
    assert abs(population - auc) < 1e-12

    key = jax.random.PRNGKey(7)
    key_mc, key_u = jax.random.split(key)
    m_tuples = 20000
    bits = jax.random.bernoulli(key_mc, 0.5, (m_tuples, 2))
    q_tuples = jnp.where(bits[..., None], q_a, q_b)
    mc = float(population_vus_mc(q_tuples))
    assert abs(mc - population) < 1e-2

    n = 160
    row_bits = jax.random.bernoulli(key_u, 0.5, (n,))
    q_rows = jnp.where(row_bits[:, None], q_a, q_b)
    ustat = float(population_vus_ustat(q_rows))
    assert abs(ustat - population) < 1e-2
    assert abs(ustat - auc) < 1e-2
    assert abs(mc - auc) < 1e-2


def test_formula1_gaussian_matches_mann_whitney():
    key = jax.random.PRNGKey(11)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    mu = jnp.array([-1.0, 1.0])
    rho = jnp.array([0.5, 0.5])
    n_per = 4000
    m_tuples = 12000

    x_neg = mu[0] + jax.random.normal(k1, (n_per,))
    x_pos = mu[1] + jax.random.normal(k2, (n_per,))

    def log_q_ratio(x):
        # Equal variance, so the likelihood ratio is the class-1 to class-0 density ratio.
        log_comp = -0.5 * (x[..., None] - mu) ** 2
        return log_comp[..., 1] - log_comp[..., 0]

    auc = float(_mann_whitney_auc(log_q_ratio(x_pos), log_q_ratio(x_neg)))

    classes = jax.random.bernoulli(k3, 0.5, (m_tuples, 2)).astype(jnp.int32)
    xs = mu[classes] + jax.random.normal(k4, (m_tuples, 2))
    log_comp = -0.5 * (xs[..., None] - mu) ** 2
    log_mix = jax.nn.logsumexp(log_comp + jnp.log(rho), axis=-1)
    q_tuples = jnp.exp(log_comp - log_mix[..., None])
    vus = float(population_vus_mc(q_tuples))
    assert abs(vus - auc) < 0.02


def test_equation_53_perfect_and_chance():
    labels = _one_hot_balanced(16, 2)
    perfect = float(label_vus(labels, labels))
    assert abs(perfect - 1.0) < 1e-8

    chance_scores = jnp.full_like(labels, 0.5)
    chance = float(label_vus(labels, chance_scores))
    assert abs(chance - 0.5) < 1e-8

    labels_k3 = _one_hot_balanced(5, 3)
    chance_k3 = float(label_vus(labels_k3, jnp.full_like(labels_k3, 1.0 / 3.0)))
    assert abs(chance_k3 - 1.0 / 6.0) < 1e-8

    perfect_k3 = float(label_vus(labels_k3, labels_k3))
    assert abs(perfect_k3 - 1.0) < 1e-8


def test_u_g_equals_label_product_average_and_near_prod_rho():
    n1, n2 = 5, 7
    labels = jnp.concatenate(
        [
            jnp.tile(jnp.array([1.0, 0.0]), (n1, 1)),
            jnp.tile(jnp.array([0.0, 1.0]), (n2, 1)),
        ]
    )
    ug = float(u_statistic_g(labels))
    # For K=2, g is 1/2 on mixed pairs and 0 otherwise, so
    # U_n(g) = n1*n2 / (n*(n-1)), the average label-product estimator of rho_1*rho_2.
    expected = (n1 * n2) / ((n1 + n2) * (n1 + n2 - 1))
    assert abs(ug - expected) < 1e-12

    balanced = _one_hot_balanced(20, 2)
    ug_balanced = float(u_statistic_g(balanced))
    assert abs(ug_balanced - 0.25) < 0.02

    labels_k3 = _one_hot_balanced(4, 3)
    n0 = n1 = n2 = 4
    n = 12
    expected_k3 = (n0 * n1 * n2) / (n * (n - 1) * (n - 2))
    assert abs(float(u_statistic_g(labels_k3)) - expected_k3) < 1e-12


def test_label_products_not_score_products_under_shrinkage():
    labels = _one_hot_balanced(12, 2)
    # p_hat shrunk toward 1/2. Argmax still recovers the true class.
    scores = 0.5 * labels + 0.5 * 0.5
    label_hat = float(label_vus(labels, scores))
    plug = float(plugin_vus(scores, jnp.array([0.5, 0.5])))
    assert abs(label_hat - 1.0) < 1e-8
    assert plug < 0.9
    assert abs(label_hat - 1.0) < abs(plug - 1.0)


def test_cross_fit_runs_and_beats_shrunk_plugin():
    key = jax.random.PRNGKey(3)
    n_per = 20
    labels_idx = jnp.concatenate(
        [
            jnp.zeros(n_per, dtype=jnp.int32),
            jnp.ones(n_per, dtype=jnp.int32),
        ]
    )
    labels = jax.nn.one_hot(labels_idx, 2, dtype=jnp.float64)
    signal = jnp.array([-2.0, 2.0])[labels_idx]
    features = signal + 0.05 * jax.random.normal(key, (n_per * 2,))
    features = features[:, None]

    smoke = float(cross_fit_vus(features, labels, seed=1, n_steps=200, learning_rate=0.4))
    assert math.isfinite(smoke)
    assert 0.0 < smoke <= 1.0

    detail = cross_fit_vus(
        features,
        labels,
        shrink=0.55,
        seed=1,
        n_steps=400,
        learning_rate=0.5,
        l2=1e-3,
        return_details=True,
    )
    label_hat = float(detail["label"])
    plug = float(detail["plugin"])
    assert math.isfinite(label_hat) and math.isfinite(plug)
    assert 0.0 < label_hat <= 1.0
    assert abs(label_hat - 1.0) < abs(plug - 1.0)
    assert label_hat > plug + 0.05
