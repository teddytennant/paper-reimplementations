"""VUS kernels from arXiv:2609.38875.

Theorem 2.22 gives the mixed volume of K coordinate simplices. Formula (1)
is the population volume under the ROC surface. Equation (53) is the
label-based double/debiased estimator. Permutations are enumerated, so K
is limited to 4. Ties break by the smallest permutation in lexicographic
order. The U-statistic is always complete.
"""

import itertools
import math

import jax
import jax.numpy as jnp

_MAX_K = 4
_PERM_TABLE = {
    k: jnp.asarray(list(itertools.permutations(range(k))), dtype=jnp.int32)
    for k in range(1, _MAX_K + 1)
}


def _perms(k):
    if k not in _PERM_TABLE:
        raise ValueError(f"permutation enumeration supports K <= {_MAX_K}, got {k}")
    return _PERM_TABLE[k]


def _combination_indices(n, k):
    if n < k:
        raise ValueError(f"need n >= K, got n={n}, K={k}")
    return jnp.asarray(list(itertools.combinations(range(n), k)), dtype=jnp.int32)


def _as_one_hot(labels, n_classes=None):
    labels = jnp.asarray(labels)
    if labels.ndim == 1:
        if n_classes is None:
            n_classes = int(jnp.max(labels)) + 1
        return jax.nn.one_hot(labels.astype(jnp.int32), n_classes, dtype=jnp.float64)
    if labels.ndim != 2:
        raise ValueError("labels must be class indices (n,) or one-hot (n, K)")
    return labels.astype(jnp.float64)


def mixed_volume(edge_lengths):
    """Mixed volume of K coordinate simplices, Theorem 2.22.

    edge_lengths[r, k] is the length of simplex r along axis k.
    V = (1/K!) * max_pi prod_r edge_lengths[r, pi(r)].
    """
    edges = jnp.asarray(edge_lengths, dtype=jnp.float64)
    if edges.ndim != 2 or edges.shape[0] != edges.shape[1]:
        raise ValueError("edge_lengths must have shape (K, K)")
    k = int(edges.shape[0])
    perms = _perms(k)
    selected = edges[jnp.arange(k), perms]
    products = jnp.prod(selected, axis=-1)
    return jnp.max(products) / math.factorial(k)


def population_vus_mc(q_tuples):
    """Monte Carlo estimator of formula (1).

    q_tuples has shape (m, K, K). Row r of tuple m is q(X_r), and the m
    tuples are treated as independent. Returns the mean of the Theorem 2.22
    kernel, which is (1/K!) * E[max_pi prod_r q_{pi(r)}(X_r)].
    """
    q_tuples = jnp.asarray(q_tuples, dtype=jnp.float64)
    if q_tuples.ndim != 3 or q_tuples.shape[1] != q_tuples.shape[2]:
        raise ValueError("q_tuples must have shape (m, K, K)")
    return jnp.mean(jax.vmap(mixed_volume)(q_tuples))


def population_vus_ustat(q_sample):
    """Complete U-statistic estimator of formula (1).

    q_sample has shape (n, K). Row i is q(X_i) for an i.i.d. sample from
    the mixture. The kernel is averaged over all increasing K-tuples.
    """
    q_sample = jnp.asarray(q_sample, dtype=jnp.float64)
    if q_sample.ndim != 2:
        raise ValueError("q_sample must have shape (n, K)")
    n, k = q_sample.shape
    k = int(k)
    combos = _combination_indices(int(n), k)

    def kernel(index):
        return mixed_volume(q_sample[index])

    return jnp.mean(jax.vmap(kernel)(combos))


def _ordered_psi(scores, labels):
    """Ordered kernel psi. scores and labels are (K, K) in tuple order."""
    k = int(scores.shape[0])
    perms = _perms(k)
    products = jnp.prod(scores[jnp.arange(k), perms], axis=-1)
    chosen = perms[jnp.argmax(products)]
    return jnp.prod(labels[jnp.arange(k), chosen]) / math.factorial(k)


def _h_kernel(scores, labels):
    """Symmetrized kernel h = average of psi over observation permutations."""
    k = int(scores.shape[0])
    perms = _perms(k)

    def one(sigma):
        return _ordered_psi(scores[sigma], labels[sigma])

    return jnp.mean(jax.vmap(one)(perms))


def _g_kernel(labels):
    """g = (1/K!) * sum_sigma prod_r Z_{sigma(r), r}."""
    k = int(labels.shape[0])
    perms = _perms(k)

    def one(sigma):
        return jnp.prod(labels[sigma, jnp.arange(k)])

    return jnp.mean(jax.vmap(one)(perms))


def u_statistic_g(labels):
    """Complete U-statistic of g. Estimates prod_k rho_k."""
    labels = _as_one_hot(labels)
    n, k = labels.shape
    combos = _combination_indices(int(n), int(k))
    values = jax.vmap(lambda index: _g_kernel(labels[index]))(combos)
    return jnp.mean(values)


def label_vus(labels, scores):
    """Equation (53): theta_hat = U_n(h_pbar) / U_n(g).

    labels are one-hot class indicators, or integer class indices.
    scores[i, k] is p_bar_k(X_i). Scores choose the permutation. Label
    products evaluate it. The U-statistic is complete.
    """
    scores = jnp.asarray(scores, dtype=jnp.float64)
    if scores.ndim != 2:
        raise ValueError("scores must have shape (n, K)")
    labels = _as_one_hot(labels, n_classes=int(scores.shape[1]))
    if labels.shape != scores.shape:
        raise ValueError("labels and scores must have the same shape")
    n, k = labels.shape
    combos = _combination_indices(int(n), int(k))

    def kernels(index):
        return _h_kernel(scores[index], labels[index]), _g_kernel(labels[index])

    hs, gs = jax.vmap(kernels)(combos)
    return jnp.mean(hs) / jnp.mean(gs)


def plugin_vus(scores, rho):
    """Naive plug-in: formula (1) with q replaced by p_hat / rho_hat.

    scores has shape (n, K), rho has shape (K,). The expectation is the
    complete U-statistic on the resulting q rows.
    """
    scores = jnp.asarray(scores, dtype=jnp.float64)
    rho = jnp.asarray(rho, dtype=jnp.float64)
    if rho.shape != (scores.shape[1],):
        raise ValueError("rho must have shape (K,)")
    return population_vus_ustat(scores / rho)


def _fit_multinomial_logistic(features, labels, n_steps, learning_rate, l2):
    """Full-batch multinomial logistic regression by gradient descent."""
    features = jnp.asarray(features, dtype=jnp.float64)
    labels = jnp.asarray(labels, dtype=jnp.float64)
    n_features = int(features.shape[1])
    n_classes = int(labels.shape[1])
    weights = jnp.zeros((n_features, n_classes), dtype=jnp.float64)
    bias = jnp.zeros((n_classes,), dtype=jnp.float64)

    def loss(params, features, labels):
        weight, offset = params
        logits = features @ weight + offset
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        nll = -jnp.mean(jnp.sum(labels * log_probs, axis=-1))
        penalty = 0.5 * l2 * jnp.sum(weight * weight)
        return nll + penalty

    def step(params, _):
        grad_w, grad_b = jax.grad(loss)(params, features, labels)
        weight, offset = params
        updated = (weight - learning_rate * grad_w, offset - learning_rate * grad_b)
        return updated, None

    params, _ = jax.lax.scan(step, (weights, bias), None, length=int(n_steps))
    return params


def _predict_proba(params, features):
    weight, offset = params
    return jax.nn.softmax(features @ weight + offset, axis=-1)


def _stratified_split(labels, seed):
    classes = jnp.argmax(labels, axis=1)
    n_classes = int(labels.shape[1])
    key = jax.random.PRNGKey(int(seed))
    left = []
    right = []
    for class_id in range(n_classes):
        index = jnp.where(classes == class_id)[0]
        if int(index.shape[0]) < 2:
            raise ValueError("each class needs at least 2 observations for 2-fold cross-fit")
        key, subkey = jax.random.split(key)
        index = jax.random.permutation(subkey, index)
        mid = int(index.shape[0]) // 2
        left.append(index[:mid])
        right.append(index[mid:])
    return jnp.concatenate(left), jnp.concatenate(right)


def cross_fit_vus(
    features,
    labels,
    *,
    shrink=0.0,
    seed=0,
    n_steps=300,
    learning_rate=0.4,
    l2=1e-2,
    return_details=False,
):
    """2-fold cross-fit label estimator.

    Fit multinomial logistic regression on fold A, evaluate equation (53)
    on fold B, and reverse. Average the two theta hats. shrink blends the
    fitted probabilities toward 1/K after fitting, which biases a plug-in
    but leaves the label products unchanged when the argmax is preserved.

    If return_details is true, also return the averaged plug-in that uses
    the same out-of-fold scores and the evaluation-fold class frequencies.
    """
    features = jnp.asarray(features, dtype=jnp.float64)
    if features.ndim == 1:
        features = features[:, None]
    labels = _as_one_hot(labels)
    if labels.shape[0] != features.shape[0]:
        raise ValueError("features and labels must have the same number of rows")
    n_classes = int(labels.shape[1])
    fold_a, fold_b = _stratified_split(labels, seed)

    def evaluate(train_index, test_index):
        params = _fit_multinomial_logistic(
            features[train_index],
            labels[train_index],
            n_steps=n_steps,
            learning_rate=learning_rate,
            l2=l2,
        )
        scores = _predict_proba(params, features[test_index])
        if shrink:
            scores = (1.0 - shrink) * scores + shrink / n_classes
        test_labels = labels[test_index]
        label_hat = label_vus(test_labels, scores)
        rho_hat = jnp.mean(test_labels, axis=0)
        plug_hat = plugin_vus(scores, rho_hat)
        return label_hat, plug_hat

    label_b, plug_b = evaluate(fold_a, fold_b)
    label_a, plug_a = evaluate(fold_b, fold_a)
    label_hat = 0.5 * (label_a + label_b)
    plug_hat = 0.5 * (plug_a + plug_b)
    if return_details:
        return {"label": label_hat, "plugin": plug_hat}
    return label_hat
