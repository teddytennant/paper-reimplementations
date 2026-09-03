"""Downstream guarantees: generation, community recovery, linear probes.

Section 4 turns prediction accuracy and recovered geometry into three
guarantees, and each one is evaluated here exactly rather than by sampling.

Generation (Section 4.1, Theorem 4). The pre-trained generator is the fitted
conditional itself; the post-trained generator tilts it by an estimated reward
under KL regularization. Both induce a Markov chain on the d^3 observed-token
contexts, whose stationary distribution gives the plausibility score rho_p, the
reward regret rho_s, and the density ratio C_{Q_q} of Assumption 3.6 against the
pre-training context distribution. The paper's criteria average over positions
of a generated sequence; under the stationary start that average is exactly the
stationary expectation, which is what is computed.

Community recovery (Section 4.2, Theorem 5). Gaussian affinity on the learned
embeddings, unnormalized Laplacian L = D - W, the K eigenvectors of its smallest
eigenvalues, then k-means, then the misclustering rate Phi minimized over
relabelings. The bandwidth sigma is not fixed in the paper; the default here
picks it by the largest Laplacian eigengap, which uses no labels.

Linear probe (Section 4.3, Theorem 6). A ridge-penalized centered multinomial
logistic probe on the contextual representation, predicting the semantic
community of the target token, scored by excess risk against the exact Bayes
rule eta_k(X_{j,c}) = P(Y = k | X_{j,c}).
"""

from __future__ import annotations

import functools
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import optax

jax.config.update("jax_enable_x64", True)


# -- generation ---------------------------------------------------------------


def plausible_mask(enum, sim) -> np.ndarray:
    """S_j(X) as a boolean array aligned with an enumeration's rows."""
    contexts = np.asarray(enum.contexts)
    if enum.semantic:
        A = sim.config.n_groups
        index = contexts[:, 0] * A**2 + contexts[:, 1] * A + contexts[:, 2]
    else:
        index = sim.context_index(contexts)
    return np.asarray(sim.plausible_token_mask)[index]


def generator_context_distribution(q, vocab_size: int, tol: float = 1e-13,
                                   max_iter: int = 200_000) -> np.ndarray:
    """Stationary distribution over contexts of the chain driven by q.

    q has shape (d^3, d) and the chain moves (x1, x2, x3) -> (x2, x3, v).
    """
    d = vocab_size
    q3 = np.asarray(q).reshape(d, d, d, d)
    p = np.full(d**3, 1.0 / d**3)
    for _ in range(max_iter):
        nxt = np.einsum("abc,abcv->bcv", p.reshape(d, d, d), q3).reshape(-1)
        residual = np.abs(nxt - p).sum()
        p = nxt
        if residual < tol:
            return p
    raise RuntimeError("generator chain did not reach stationarity")


def density_ratio(numerator, denominator) -> float:
    """||dP_Q / dP_pre||_infty of Assumption 3.6."""
    num = np.asarray(numerator)
    den = np.asarray(denominator)
    live = num > 0
    if np.any(den[live] <= 0):
        return np.inf
    return float((num[live] / den[live]).max())


def plausibility(q, plausible_mask, context_dist) -> float:
    """rho_p(q, p), the mass the generator puts on the plausible set S_j(X)."""
    per_context = (np.asarray(q) * np.asarray(plausible_mask)).sum(axis=1)
    return float(np.asarray(context_dist) @ per_context)


def reward_regret(q, reward, context_dist) -> float:
    """rho_s(q, r*), the population reward regret of Section 4.1."""
    r = np.asarray(reward)
    gap = r.max(axis=1) - (np.asarray(q) * r).sum(axis=1)
    return float(np.asarray(context_dist) @ gap)


def make_reward(plausible_mask, seed: int, scale: float = 1.0) -> np.ndarray:
    """A true reward consistent with the plausible set, Assumption 3.1 and 3.2.

    Zero off the support and in [scale/2, scale] on it, so the margin
    zeta_j(X) = min_S r* - max_{not S} r* is positive and r* is bounded by R.
    """
    rng = np.random.default_rng(seed)
    mask = np.asarray(plausible_mask)
    return scale * mask * (0.5 + 0.5 * rng.random(mask.shape))


def estimate_reward(reward, epsilon_r: float, seed: int) -> np.ndarray:
    """rhat with sup norm error exactly epsilon_r, Assumption 3.3."""
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=np.shape(reward))
    return np.asarray(reward) + epsilon_r * signs


def tilted_generator(reference, reward_hat, lam: float) -> np.ndarray:
    """The KL-regularized reward maximizer of Section 4.1."""
    logits = np.log(np.maximum(np.asarray(reference), 1e-300)) + np.asarray(reward_hat) / lam
    logits = logits - logits.max(axis=1, keepdims=True)
    w = np.exp(logits)
    return w / w.sum(axis=1, keepdims=True)


def theorem4_plausibility_pre(coverage: float, epsilon_p: float) -> float:
    return 1.0 - coverage * epsilon_p


def theorem4_plausibility_post(coverage: float, epsilon_p: float, epsilon_r: float,
                               lam: float) -> float:
    return 1.0 - coverage * np.exp(2.0 * epsilon_r / lam) * epsilon_p


def theorem4_regret_post(reward_scale: float, epsilon_r: float, lam: float, k: int,
                         coverage: float, epsilon_p: float) -> float:
    return min(
        reward_scale,
        4 * epsilon_r
        + 2 * lam * (1 + np.log(2 * k))
        + 4 * coverage * k**2 * epsilon_p,
    )


def theorem4_regret_uniform(reward_scale: float, epsilon_r: float, lam: float,
                            vocab_size: int) -> float:
    return min(reward_scale, 4 * epsilon_r + 2 * lam * (1 + np.log(vocab_size)))


# -- community recovery -------------------------------------------------------


def squared_distances(points) -> np.ndarray:
    e = np.asarray(points)
    return ((e[:, None, :] - e[None, :, :]) ** 2).sum(axis=-1)


def gamma_h_distances(gamma_h) -> np.ndarray:
    """The squared embedding distance implied by an exact Gamma_h.

    Section 2.3 gives Gamma_h = 1 - exp(-J_A) and J_A is comparable to
    ||E_v1 - E_v2||^2 up to the curvature constants, so -log(1 - Gamma_h) is the
    oracle geometry up to a scale that sigma absorbs. This is what Figure 5 runs
    the clustering pipeline on for its oracle reference.
    """
    return -np.log(np.maximum(1.0 - np.asarray(gamma_h), 1e-12))


def laplacian_eigen(d2, sigma: float):
    w = np.exp(-np.asarray(d2) / (2.0 * sigma**2))
    laplacian = np.diag(w.sum(axis=0)) - w
    values, vectors = np.linalg.eigh(0.5 * (laplacian + laplacian.T))
    return w, values, vectors


def select_sigma(d2, n_clusters: int, n_grid: int = 25) -> float:
    """Pick the bandwidth by the largest Laplacian eigengap.

    The paper does not fix sigma. The eigengap rule uses no labels, so the
    clustering stays unsupervised.
    """
    d2 = np.asarray(d2)
    off = d2[~np.eye(len(d2), dtype=bool)]
    scale = float(np.sqrt(np.median(off)))
    grid = np.geomspace(0.1 * scale, 2.0 * scale, n_grid)
    gaps = [laplacian_eigen(d2, s)[1] for s in grid]
    return float(grid[int(np.argmax([g[n_clusters] - g[n_clusters - 1] for g in gaps]))])


def gaussian_affinity(embeddings, sigma: float | None = None, n_clusters: int = 2):
    """W_{v1,v2} = exp(-||E_v1 - E_v2||^2 / 2 sigma^2)."""
    d2 = squared_distances(embeddings)
    if sigma is None:
        sigma = select_sigma(d2, n_clusters)
    return np.exp(-d2 / (2.0 * sigma**2)), sigma


def kmeans(points, n_clusters: int, seed: int = 0, n_restarts: int = 20,
           n_iter: int = 200):
    """Lloyd's algorithm with k-means++ starts. Returns (labels, objective)."""
    x = np.asarray(points)
    rng = np.random.default_rng(seed)
    best_labels, best_obj = None, np.inf
    for _ in range(n_restarts):
        centers = [x[rng.integers(len(x))]]
        for _ in range(n_clusters - 1):
            d2 = np.min(
                ((x[:, None, :] - np.stack(centers)[None]) ** 2).sum(-1), axis=1
            )
            total = d2.sum()
            probs = d2 / total if total > 0 else np.full(len(x), 1.0 / len(x))
            centers.append(x[rng.choice(len(x), p=probs)])
        centers = np.stack(centers)
        labels = None
        for _ in range(n_iter):
            d2 = ((x[:, None, :] - centers[None]) ** 2).sum(-1)
            new_labels = d2.argmin(axis=1)
            if labels is not None and np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for k in range(n_clusters):
                if np.any(labels == k):
                    centers[k] = x[labels == k].mean(axis=0)
        obj = float(((x - centers[labels]) ** 2).sum())
        if obj < best_obj:
            best_labels, best_obj = labels, obj
    return best_labels, best_obj


def spectral_clustering_from_distances(d2, n_clusters: int, sigma: float | None = None,
                                       seed: int = 0, n_restarts: int = 20):
    """The pipeline of Section 4.2, on the unnormalized Laplacian L = D - W."""
    if sigma is None:
        sigma = select_sigma(d2, n_clusters)
    w, _, vectors = laplacian_eigen(d2, sigma)
    u = vectors[:, :n_clusters]
    labels, obj = kmeans(u, n_clusters, seed=seed, n_restarts=n_restarts)
    return labels, obj, u, sigma, w


def spectral_clustering(embeddings, n_clusters: int, sigma: float | None = None,
                        seed: int = 0, n_restarts: int = 20):
    """Section 4.2 applied to learned token embeddings."""
    return spectral_clustering_from_distances(
        squared_distances(embeddings), n_clusters, sigma, seed, n_restarts
    )


def misclustering_rate(predicted, truth, n_clusters: int) -> float:
    """Phi(chat, c), minimized over relabelings."""
    predicted = np.asarray(predicted)
    truth = np.asarray(truth)
    best = min(
        int((predicted != np.asarray(perm)[truth]).sum())
        for perm in itertools.permutations(range(n_clusters))
    )
    return best / len(truth)


def affinity_bounds(gamma_within: float, gamma_between: float, epsilon_p: float,
                    p_min: float, sigma: float, curvature_lo: float,
                    curvature_hi: float) -> tuple[float, float]:
    """alpha_sigma and beta_sigma of Assumption 4.4.

    alpha_sigma lower bounds the within-community affinities and beta_sigma
    upper bounds the between-community ones, both through Theorem 1. g_w and
    g_b are passed as the squared quantities the paper bounds Gamma_h by.
    """
    slack = np.sqrt(epsilon_p / p_min)
    inner_w = np.sqrt(gamma_within) + slack
    inner_b = np.sqrt(gamma_between) - slack
    if inner_w >= 1.0 or inner_b <= 0.0:
        raise ValueError("Assumption 4.4 is not satisfiable at this epsilon_p")
    alpha = (1.0 - inner_w**2) ** (4.0 / (sigma**2 * curvature_lo))
    beta = (1.0 - inner_b**2) ** (4.0 / (sigma**2 * curvature_hi))
    return float(alpha), float(beta)


def theorem5_bound(alpha: float, beta: float, n_clusters: int, sizes,
                   kappa: float = 0.0) -> float:
    """The misclustering bound of Theorem 5, in its general form."""
    sizes = np.asarray(sizes)
    d = int(sizes.sum())
    d_min, d_max = int(sizes.min()), int(sizes.max())
    return float(
        64
        * (2 + kappa) ** 2
        * n_clusters
        * d_max
        / d
        * ((d - d_min) * beta / (alpha * d_min)) ** 2
    )


def assumption4_ratio(alpha: float, beta: float, n_clusters: int, sizes,
                      kappa: float = 0.0) -> tuple[float, float]:
    """(alpha/beta, the threshold it must exceed) in Assumption 4.4."""
    sizes = np.asarray(sizes)
    d = int(sizes.sum())
    d_min, d_max = int(sizes.min()), int(sizes.max())
    threshold = (
        8 * (2 + kappa) * np.sqrt(n_clusters * d_max) * (d - d_min) / d_min**1.5
    )
    return alpha / beta, float(threshold)


# -- linear probe -------------------------------------------------------------


def class_posteriors(sim, enum) -> np.ndarray:
    """eta_k(X_{j,c}) = P(Y = k | X_{j,c}) for the community of the next token."""
    cond = np.asarray(enum.conditional)
    k = sim.config.n_communities
    out = np.zeros((cond.shape[0], k))
    for c in range(k):
        out[:, c] = cond[:, sim.token_community == c].sum(axis=1)
    return out


def bayes_risk(posteriors, context_dist) -> float:
    return float(1.0 - np.asarray(context_dist) @ np.asarray(posteriors).max(axis=1))


def sample_labeled(posteriors, context_dist, n: int, seed: int):
    """A labeled downstream sample, independent of pre-training (Assumption 5.7)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(context_dist), size=n, p=np.asarray(context_dist))
    eta = np.asarray(posteriors)[idx]
    labels = np.array([rng.choice(eta.shape[1], p=row / row.sum()) for row in eta])
    return idx, labels


@functools.partial(jax.jit, static_argnums=(2, 3, 4))
def _fit_probe(x, y, n_classes, n_steps, learning_rate):
    n, s = x.shape

    def center(p):
        return {
            "w": p["w"] - p["w"].mean(axis=0, keepdims=True),
            "b": p["b"] - p["b"].mean(),
        }

    def loss(p):
        p = center(p)
        logits = x @ p["w"].T + p["b"]
        nll = -jnp.mean(jnp.take_along_axis(jax.nn.log_softmax(logits, -1), y[:, None], 1))
        return nll + (jnp.sum(p["w"] ** 2) + jnp.sum(p["b"] ** 2)) / n

    params = {"w": jnp.zeros((n_classes, s)), "b": jnp.zeros(n_classes)}
    optimizer = optax.adam(learning_rate)
    state = optimizer.init(params)

    def step(carry, _):
        p, state = carry
        value, grad = jax.value_and_grad(loss)(p)
        updates, state = optimizer.update(grad, state, p)
        return (optax.apply_updates(p, updates), state), value

    (params, _), _ = jax.lax.scan(step, (params, state), None, length=n_steps)
    return center(params)


def fit_probe(features, labels, n_classes: int, n_steps: int = 1500,
              learning_rate: float = 0.1):
    """The ridge-penalized centered multinomial logistic probe of Section 4.3.

    Minimizes -1/n sum log P + (1/n)(||W||_F^2 + ||b||^2) subject to
    sum_k W_k = 0 and sum_k b_k = 0. The objective is convex on that subspace.
    """
    return _fit_probe(
        jnp.asarray(features), jnp.asarray(labels), n_classes, n_steps, learning_rate
    )


def probe_excess_risk(probe, features, posteriors, context_dist) -> float:
    """R(hhat) = P(Y != hhat) - P(Y != h*), exactly, over every context."""
    logits = np.asarray(jnp.asarray(features) @ probe["w"].T + probe["b"])
    chosen = logits.argmax(axis=1)
    eta = np.asarray(posteriors)
    risk = 1.0 - float(np.asarray(context_dist) @ eta[np.arange(len(chosen)), chosen])
    return risk - bayes_risk(eta, context_dist)


def one_hot_previous_token(enum, vocab_size: int) -> np.ndarray:
    """The context-free baseline of Figure 6: the token just before the target."""
    previous = np.asarray(enum.contexts)[:, -1]
    out = np.zeros((len(previous), vocab_size))
    out[np.arange(len(previous)), previous] = 1.0
    return out
