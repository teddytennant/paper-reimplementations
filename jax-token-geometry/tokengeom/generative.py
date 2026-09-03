"""Ground-truth generative model, Appendix B.1 of arXiv:2608.30072.

The vocabulary is split into `n_variants` exchangeable token types per semantic
group, the groups are split into communities, and the next-group distribution
depends on the three preceding groups through an order-sensitive nonlinear rule.
Because the variant weights rho_m do not depend on the context, the three token
types of a group share a conditional context distribution and differ only in
marginal frequency, so Gamma_h is exactly zero inside a group. That is the fact
Section 5 uses to check equivalence collapse.

Everything here is a population quantity computed by enumeration over the
A^3 semantic contexts (A = number of semantic groups). Section 5 enumerates the
d^3 observed-token contexts instead; `context_distribution_tokens` and
`gamma_h_tokens_direct` do that too, and `test_generative.py` checks the two
routes agree. The semantic route is the one used everywhere else because the
factorization p(token context) = p(semantic context) * prod_i rho_{m_i} makes
it exact and much cheaper.

Paper defaults are K = 4 communities, 4 groups each, 3 variants (d = 48), a
16-dimensional latent geometry and 5 geometric slots. `SimulationConfig.small()`
shrinks that so the test suite runs in seconds.
"""

from __future__ import annotations

import dataclasses
import functools

import numpy as np

TOL_STATIONARY = 1e-12


@dataclasses.dataclass(frozen=True)
class SimulationConfig:
    """Knobs of the Appendix B.1 construction. Defaults are the paper's."""

    n_communities: int = 4
    groups_per_community: int = 4
    n_variants: int = 3
    latent_dim: int = 16
    variant_exponent: float = 0.8
    position_weights: tuple = (0.2, 0.3, 0.5)
    interaction_scale: float = 0.25
    alpha_lo: float = 0.15
    alpha_hi: float = 0.45
    exploration_mass: float = 0.2
    n_geometric: int = 5
    seed: int = 0
    variant_weights: tuple | None = None  # overrides rho_m = m^-0.8 / sum
    beta: float | None = None  # skips the quantile calibration when set

    @property
    def n_groups(self) -> int:
        return self.n_communities * self.groups_per_community

    @property
    def vocab_size(self) -> int:
        return self.n_groups * self.n_variants

    @property
    def plausible_groups(self) -> int:
        """|S_a(X_{j,c})|, one exploration edge plus the geometric slots."""
        return 1 + self.n_geometric

    @property
    def plausible_tokens(self) -> int:
        """|S_j(X)|, eighteen token types in the paper."""
        return self.plausible_groups * self.n_variants

    @classmethod
    def small(cls, seed: int = 0) -> "SimulationConfig":
        """4 communities of 2 groups, 2 variants each, so d = 16.

        Keeps every structure the theorems need (communities, exchangeable
        variants with unequal frequency, a sparse plausible set) while dropping
        the enumeration from 110592 token contexts to 4096.
        """
        return cls(
            n_communities=4,
            groups_per_community=2,
            n_variants=2,
            latent_dim=8,
            n_geometric=3,
            seed=seed,
        )


def de_bruijn_sequence(k: int, n: int) -> np.ndarray:
    """A de Bruijn sequence B(k, n) by the Frank-Kamenetskii-Moore algorithm."""
    a = [0] * (k * n)
    seq: list[int] = []

    def db(t: int, p: int) -> None:
        if t > n:
            if n % p == 0:
                seq.extend(a[1 : p + 1])
        else:
            a[t] = a[t - p]
            db(t + 1, p)
            for j in range(a[t - p] + 1, k):
                a[t] = j
                db(t + 1, t)

    db(1, 1)
    return np.asarray(seq, dtype=np.int64)


def de_bruijn_successor(n_symbols: int, order: int = 3) -> np.ndarray:
    """r_B: for each order-3 window of groups, its successor on the cycle.

    Returns an array of length n_symbols**order indexed by the window
    (a_{j-3}, a_{j-2}, a_{j-1}) read in base n_symbols with the oldest position
    most significant. Every window occurs exactly once on the cycle, so the map
    is total and the induced chain is a single cycle over the states.
    """
    seq = de_bruijn_sequence(n_symbols, order)
    length = len(seq)
    assert length == n_symbols**order
    succ = np.empty(length, dtype=np.int64)
    powers = n_symbols ** np.arange(order - 1, -1, -1)
    for i in range(length):
        window = seq[(np.arange(order) + i) % length]
        succ[int(window @ powers)] = seq[(i + order) % length]
    return succ


def _random_orthogonal(rng: np.random.Generator, dim: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((dim, dim)))
    return q * np.sign(np.diag(r))


class SemanticModel:
    """The population process of Appendix B.1, with exact enumeration."""

    def __init__(self, config: SimulationConfig | None = None):
        self.config = config or SimulationConfig()
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        A, d, D = cfg.n_groups, cfg.vocab_size, cfg.latent_dim
        if D < cfg.n_communities + 1:
            raise ValueError("latent_dim must exceed the number of communities")

        # rho_m = m^{-0.8} / sum_r r^{-0.8}, the exchangeable variant weights.
        if cfg.variant_weights is None:
            m = np.arange(1, cfg.n_variants + 1, dtype=np.float64)
            self.variant_weights = m**(-cfg.variant_exponent)
        else:
            self.variant_weights = np.asarray(cfg.variant_weights, dtype=np.float64)
        self.variant_weights /= self.variant_weights.sum()

        self.token_group = np.repeat(np.arange(A), cfg.n_variants)
        self.token_variant = np.tile(np.arange(cfg.n_variants), A)
        self.group_community = np.repeat(
            np.arange(cfg.n_communities), cfg.groups_per_community
        )
        self.token_community = self.group_community[self.token_group]

        # Community centers: orthonormal. Group offsets: unit vectors in the
        # orthogonal complement of their span.
        basis = _random_orthogonal(rng, D)
        mu = basis[: cfg.n_communities]
        complement = basis[cfg.n_communities :]
        raw = rng.standard_normal((A, D - cfg.n_communities))
        delta = raw @ complement
        delta /= np.linalg.norm(delta, axis=1, keepdims=True)

        alpha = np.linspace(cfg.alpha_lo, cfg.alpha_hi, cfg.groups_per_community)
        self.alpha = np.tile(alpha, cfg.n_communities)
        u = mu[self.group_community] + self.alpha[:, None] * delta
        self.u = u / np.linalg.norm(u, axis=1, keepdims=True)

        self.rotations = np.stack([_random_orthogonal(rng, D) for _ in range(3)])
        raw_b = rng.standard_normal((D, D))
        self.interaction = raw_b / np.linalg.norm(raw_b, 2)
        self.readout = _random_orthogonal(rng, D)

        self.successor = de_bruijn_successor(A, 3)
        self._build_contexts()
        self.beta = cfg.beta if cfg.beta is not None else self._calibrate_beta()
        self.group_conditional = self._group_conditional(self.beta)
        self.state_distribution = self._stationary_distribution()

    # -- context enumeration -------------------------------------------------

    def _build_contexts(self) -> None:
        cfg = self.config
        A = cfg.n_groups
        idx = np.arange(A**3)
        self.context_groups = np.stack(
            [idx // A**2, (idx // A) % A, idx % A], axis=1
        )  # (A^3, 3), oldest position first

        w = np.asarray(cfg.position_weights, dtype=np.float64)
        u = self.u[self.context_groups]  # (A^3, 3, D)
        linear = np.einsum("i,idk,tid->tk", w, self.rotations, u)
        inter = u[:, 0] * u[:, 1] + u[:, 1] * u[:, 2]
        ctilde = linear + cfg.interaction_scale * (inter @ self.interaction.T)
        self.context_feature = ctilde / np.linalg.norm(ctilde, axis=1, keepdims=True)

        # Compatibility scores c(X)^T O u_a, then the geometric slots.
        self.scores = self.context_feature @ (self.readout @ self.u.T)  # (A^3, A)
        explore = self.successor
        masked = self.scores.copy()
        masked[np.arange(A**3), explore] = -np.inf
        forced = np.zeros(A**3, dtype=bool)
        forced[0] = True  # context (1,1,1) in the paper's 1-based labelling
        masked[0, 0] = np.inf  # force group 1 into G, compete the other slots
        order = np.argsort(-masked, axis=1, kind="stable")
        self.geometric_slots = np.sort(order[:, : cfg.n_geometric], axis=1)
        self.forced_context = forced

        plausible = np.zeros((A**3, A), dtype=bool)
        rows = np.arange(A**3)[:, None]
        plausible[rows, self.geometric_slots] = True
        plausible[rows[:, 0], explore] = True
        self.plausible_group_mask = plausible
        assert plausible.sum(axis=1).min() == cfg.plausible_groups
        assert plausible.sum(axis=1).max() == cfg.plausible_groups

    # -- conditional distribution -------------------------------------------

    def _geometric_weights(self, beta: float) -> np.ndarray:
        """softmax over G(X) of beta * c(X)^T O u_a. Shape (A^3, n_geometric)."""
        rows = np.arange(self.config.n_groups**3)[:, None]
        s = beta * self.scores[rows, self.geometric_slots]
        s = s - s.max(axis=1, keepdims=True)
        e = np.exp(s)
        return e / e.sum(axis=1, keepdims=True)

    def _calibrate_beta(self) -> float:
        """Bisect beta so the lower 1% quantile of the geometric branch at the
        rarest variant hits 1/(n_variants * |S_j(X)|); 1/54 in the paper."""
        cfg = self.config
        target = 1.0 / (cfg.n_variants * cfg.plausible_tokens)
        rho_min = self.variant_weights[-1]

        def quantile(beta: float) -> float:
            w = self._geometric_weights(beta)
            vals = (1.0 - cfg.exploration_mass) * w * rho_min
            return float(np.quantile(vals.reshape(-1), 0.01))

        lo, hi = 0.0, 1.0
        if quantile(lo) <= target:
            raise ValueError("target quantile unreachable, beta would be negative")
        while quantile(hi) > target:
            hi *= 2.0
            if hi > 1e4:
                raise ValueError("beta calibration failed to bracket the target")
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if quantile(mid) > target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _group_conditional(self, beta: float) -> np.ndarray:
        """P(next group = a | semantic context), shape (A^3, A)."""
        cfg = self.config
        A = cfg.n_groups
        out = np.zeros((A**3, A), dtype=np.float64)
        rows = np.arange(A**3)[:, None]
        out[rows, self.geometric_slots] = (
            1.0 - cfg.exploration_mass
        ) * self._geometric_weights(beta)
        np.add.at(out, (np.arange(A**3), self.successor), cfg.exploration_mass)
        return out

    def _stationary_distribution(self) -> np.ndarray:
        """p(X_{j,c}) over semantic contexts, by power iteration to 1e-12."""
        A = self.config.n_groups
        n = A**3
        p = np.full(n, 1.0 / n)
        # state (a, b, c) -> (b, c, a') with probability P(a' | a, b, c)
        head = (np.arange(n) % A**2) * A
        for _ in range(200_000):
            nxt = np.zeros(n)
            np.add.at(
                nxt,
                (head[:, None] + np.arange(A)[None, :]).reshape(-1),
                (p[:, None] * self.group_conditional).reshape(-1),
            )
            residual = np.abs(nxt - p).sum()
            p = nxt
            if residual < TOL_STATIONARY:
                return p
        raise RuntimeError("power iteration did not converge")

    # -- token-level population quantities -----------------------------------

    @functools.cached_property
    def token_conditional(self) -> np.ndarray:
        """p_j(v | semantic context), shape (A^3, d)."""
        return (
            self.group_conditional[:, self.token_group]
            * self.variant_weights[self.token_variant][None, :]
        )

    @functools.cached_property
    def group_marginal(self) -> np.ndarray:
        """p(a) = E_{X,j} P(a | X_{j,c})."""
        return self.state_distribution @ self.group_conditional

    @functools.cached_property
    def token_marginal(self) -> np.ndarray:
        """p(v), the marginal token frequency of Section 2.3."""
        return (
            self.group_marginal[self.token_group]
            * self.variant_weights[self.token_variant]
        )

    @functools.cached_property
    def group_context_conditional(self) -> np.ndarray:
        """p_a(X_{j,c} | a) over semantic contexts, by Bayes' rule."""
        joint = self.state_distribution[:, None] * self.group_conditional
        return (joint / self.group_marginal[None, :]).T  # (A, A^3)

    @functools.cached_property
    def gamma_h_groups(self) -> np.ndarray:
        """Gamma_h between semantic groups, the (A, A) dissimilarity matrix."""
        root = np.sqrt(self.group_context_conditional)
        out = np.maximum(1.0 - root @ root.T, 0.0)
        np.fill_diagonal(out, 0.0)
        return out

    @functools.cached_property
    def gamma_h_tokens(self) -> np.ndarray:
        """Gamma_h between token types.

        Equal to Gamma_h between their groups. The variant weight rho_m cancels
        out of Bayes' rule and the extra factor prod_i rho_{m_i} carried by the
        token context is shared by both arguments, so it sums to one inside the
        Bhattacharyya coefficient. Hence Gamma_h(v1, v2) = 0 exactly for two
        variants of one group, which is the equivalence-collapse target of
        Section 2.4.
        """
        return self.gamma_h_groups[np.ix_(self.token_group, self.token_group)]

    @functools.cached_property
    def plausible_token_mask(self) -> np.ndarray:
        """S_j(X) as a boolean (A^3, d) mask."""
        return self.plausible_group_mask[:, self.token_group]

    @functools.cached_property
    def sparsity_k(self) -> int:
        """Smallest k with p_j(v|X) >= 1/k on the support, Assumption 3.5."""
        p = self.token_conditional
        return int(np.ceil(1.0 / p[p > 0].min()))

    # -- observed-token contexts (the expensive route, used for cross-checks) --

    def _token_context_state(self) -> np.ndarray:
        """Semantic-context index of each of the d^3 observed-token contexts."""
        d, A = self.config.vocab_size, self.config.n_groups
        idx = np.arange(d**3)
        g = self.token_group[np.stack([idx // d**2, (idx // d) % d, idx % d], 1)]
        return g[:, 0] * A**2 + g[:, 1] * A + g[:, 2]

    @functools.cached_property
    def context_distribution_tokens(self) -> np.ndarray:
        """p(X_{j,c}) over the d^3 observed-token contexts."""
        d = self.config.vocab_size
        idx = np.arange(d**3)
        parts = np.stack([idx // d**2, (idx // d) % d, idx % d], axis=1)
        w = self.variant_weights[self.token_variant[parts]].prod(axis=1)
        return self.state_distribution[self._token_context_state()] * w

    def gamma_h_tokens_direct(self) -> np.ndarray:
        """Gamma_h computed by enumerating the d^3 observed-token contexts."""
        pc = self.context_distribution_tokens
        cond = self.token_conditional[self._token_context_state()]  # (d^3, d)
        joint = pc[:, None] * cond
        pv = joint.sum(axis=0)
        contextual = (joint / pv[None, :]).T
        root = np.sqrt(contextual)
        out = np.maximum(1.0 - root @ root.T, 0.0)
        np.fill_diagonal(out, 0.0)
        return out

    # -- sampling -------------------------------------------------------------

    def sample_pairs(self, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """n i.i.d. (context, next token) pairs, the training data of Section 5.

        The context is drawn from the stationary p(X_{j,c}) and the next token
        from p_j(. | X). Contexts are returned as (n, 3) token ids, oldest
        position first.
        """
        rng = np.random.default_rng(seed)
        M = self.config.n_variants
        states = rng.choice(len(self.state_distribution), size=n, p=self.state_distribution)
        groups = self.context_groups[states]
        variants = rng.choice(M, size=(n, 3), p=self.variant_weights)
        contexts = groups * M + variants
        cond = self.group_conditional[states]
        u = rng.random((n, 1))
        next_group = (cond.cumsum(axis=1) < u).sum(axis=1).clip(max=self.config.n_groups - 1)
        next_variant = rng.choice(M, size=n, p=self.variant_weights)
        return contexts, next_group * M + next_variant

    def context_index(self, contexts: np.ndarray) -> np.ndarray:
        """Semantic-context index for a batch of (n, 3) token contexts."""
        A = self.config.n_groups
        g = self.token_group[contexts]
        return g[:, 0] * A**2 + g[:, 1] * A + g[:, 2]
