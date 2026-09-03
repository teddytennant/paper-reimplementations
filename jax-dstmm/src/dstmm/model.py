"""The deep pathway construction of Section 2 and its observed-data likelihood.

Layers are numbered l = 1..h from the observation down. The latent dimensions are
p = r_0 > r_1 > ... > r_h >= 1 and layer l holds K_l local components with
parameters eta_a^(l), Lambda_a^(l), Psi_a^(l), delta_a^(l), as in Section 2.2.
The baseline specification keeps Psi diagonal (Section 2.7), so it is stored as
its diagonal.

A complete path s = (s_1, ..., s_h) has prior probability pi_s = prod_l pi_{s_l}^(l),
and Proposition 1 collapses it into GHSt_p(mu_s, Sigma_s, alpha_s, nu_s) through
the backward recursions (5)-(7) started from mu^(h) = alpha^(h) = 0, Sigma^(h) = I.
The observed density is then the finite mixture over paths in Section 2.4, with
log-likelihood (11) and pathway responsibilities (12).
"""

import dataclasses
import itertools

import jax
import jax.numpy as jnp

from dstmm.ghst import ghst_logpdf


@dataclasses.dataclass(frozen=True)
class DStMM:
    """Parameters of a deep skew-t mixture model.

    dims: (p, r_1, ..., r_h), strictly decreasing per Section 2.
    ks: (K_1, ..., K_h).
    log_pi[l-1]: (K_l,) log mixing proportions of layer l.
    eta[l-1]: (K_l, r_{l-1}) local intercepts.
    lam[l-1]: (K_l, r_{l-1}, r_l) local loading matrices.
    psi_diag[l-1]: (K_l, r_{l-1}) diagonal of the local specific covariance.
    delta[l-1]: (K_l, r_{l-1}) local skewness, held at zero when l is not in the
        active set A of Section 2.2.
    nu: (|S|,) degrees of freedom, one per complete path, in paths order.

    dims and ks are static pytree metadata; everything else is a traced leaf, so
    a DStMM can be passed straight through jit and grad.
    """

    dims: tuple
    ks: tuple
    log_pi: tuple
    eta: tuple
    lam: tuple
    psi_diag: tuple
    delta: tuple
    nu: jnp.ndarray

    @property
    def depth(self):
        return len(self.ks)

    @property
    def paths(self):
        """Complete paths of Section 2.2, in lexicographic order."""
        return list(itertools.product(*[range(k) for k in self.ks]))

    def replace(self, **kw):
        return dataclasses.replace(self, **kw)


jax.tree_util.register_pytree_node(
    DStMM,
    lambda m: (
        (m.log_pi, m.eta, m.lam, m.psi_diag, m.delta, m.nu),
        (m.dims, m.ks),
    ),
    lambda aux, ch: DStMM(aux[0], aux[1], *ch),
)


def collapse_paths(model):
    """Run the recursions (5)-(7) up every complete path.

    Returns (mu_seq, alpha_seq, sigma_seq, log_pi_paths) where entry l of each
    sequence is stacked over paths and holds the layer-l induced quantity of
    Proposition 1, equation (8). Entry 0 is the observation level, so
    mu_seq[0][i], alpha_seq[0][i], sigma_seq[0][i] are the mu_s, alpha_s, Sigma_s
    of equation (10).
    """
    h = model.depth
    paths = model.paths
    per_path = []
    for s in paths:
        mus = [None] * (h + 1)
        als = [None] * (h + 1)
        sgs = [None] * (h + 1)
        r_h = model.dims[h]
        mus[h] = jnp.zeros(r_h)
        als[h] = jnp.zeros(r_h)
        sgs[h] = jnp.eye(r_h)
        for l in range(h, 0, -1):
            a = s[l - 1]
            eta = model.eta[l - 1][a]
            lam = model.lam[l - 1][a]
            psi = model.psi_diag[l - 1][a]
            dlt = model.delta[l - 1][a]
            mus[l - 1] = eta + lam @ mus[l]
            als[l - 1] = dlt + lam @ als[l]
            sgs[l - 1] = jnp.diag(psi) + lam @ sgs[l] @ lam.T
        per_path.append((mus, als, sgs))

    mu_seq = tuple(jnp.stack([pp[0][l] for pp in per_path]) for l in range(h + 1))
    alpha_seq = tuple(jnp.stack([pp[1][l] for pp in per_path]) for l in range(h + 1))
    sigma_seq = tuple(jnp.stack([pp[2][l] for pp in per_path]) for l in range(h + 1))
    log_pi_paths = jnp.stack(
        [sum(model.log_pi[l][s[l]] for l in range(h)) for s in paths]
    )
    return mu_seq, alpha_seq, sigma_seq, log_pi_paths


def path_log_densities(model, y):
    """log pi_s + log g_GHST,p(y_j; mu_s, Sigma_s, alpha_s, nu_s), shape (n, |S|).

    These are the a_js of Section 3.2.
    """
    mu_seq, alpha_seq, sigma_seq, log_pi_paths = collapse_paths(model)
    mu, alpha, sigma = mu_seq[0], alpha_seq[0], sigma_seq[0]

    def one_path(mu_s, sigma_s, alpha_s, nu_s, lp):
        return lp + ghst_logpdf(y, mu_s, sigma_s, alpha_s, nu_s)

    return jax.vmap(one_path, in_axes=(0, 0, 0, 0, 0), out_axes=1)(
        mu, sigma, alpha, model.nu, log_pi_paths
    )


def log_likelihood(model, y):
    """The observed-data log-likelihood, equation (11)."""
    return jnp.sum(jax.scipy.special.logsumexp(path_log_densities(model, y), axis=1))


def per_point_log_likelihood(model, y):
    """log f(y_j; Theta) for each observation, the summand of equation (11)."""
    return jax.scipy.special.logsumexp(path_log_densities(model, y), axis=1)


def path_responsibilities(model, y):
    """tau_js of equation (12), shape (n, |S|)."""
    a = path_log_densities(model, y)
    return jnp.exp(a - jax.scipy.special.logsumexp(a, axis=1, keepdims=True))


def layer_responsibilities(model, tau, layer):
    """tau_ja^(l) = sum over paths with s_l = a of tau_js, for 1-based layer l."""
    return tau @ path_component_mask(model, layer)


def path_component_mask(model, layer):
    """(|S|, K_l) indicator of s_l = a, for 1-based layer l."""
    idx = jnp.array([s[layer - 1] for s in model.paths])
    return (idx[:, None] == jnp.arange(model.ks[layer - 1])[None, :]).astype(jnp.float64)


def n_params(dims, ks, active_set=(1,), df_mode="common"):
    """The implementation-based parameter count d_code of equation (30).

    Per layer l the count is (K_l - 1) free mixing proportions, K_l * r_{l-1}
    intercepts, K_l * r_{l-1} diagonal specific variances, and the loading block
    of equation (19), r_{l-1} r_l - r_l (r_l - 1) / 2, which subtracts the
    rotational non-identifiability correction. A layer in the skewness active set
    A adds K_l * r_{l-1} more. The degrees-of-freedom term d_nu is 1 for a common
    nu, K_1 for a first-layer-specific nu, and prod_l K_l for pathway-specific nu.

    Section 3.7 is explicit that this is the software convention: it makes no
    further subtraction of 2 * sum_{l<h} r_l for the intermediate-coordinate gauge
    of Section 2.7.
    """
    h = len(ks)
    total = 0
    for l in range(1, h + 1):
        r_prev, r_cur, k = dims[l - 1], dims[l], ks[l - 1]
        total += (k - 1) + k * (
            r_prev * r_cur + 2 * r_prev - r_cur * (r_cur - 1) // 2
        )
        if l in active_set:
            total += k * r_prev
    if df_mode == "common":
        d_nu = 1
    elif df_mode == "layer1":
        d_nu = ks[0]
    elif df_mode == "path":
        d_nu = int(jnp.prod(jnp.array(ks)))
    else:
        raise ValueError(f"unknown df_mode {df_mode!r}")
    return total + d_nu


def bic(model, y, active_set=(1,), df_mode="common"):
    """BIC(M) = -2 l(Theta_hat) + d_code(M) log n, Section 3.7. Smaller is better."""
    n = jnp.shape(y)[0]
    d = n_params(model.dims, model.ks, active_set, df_mode)
    return -2.0 * log_likelihood(model, y) + d * jnp.log(n)


def sample_dstmm(key, model, n):
    """Draw from the hierarchy itself, layer by layer, not from the collapsed GHST.

    One path is drawn per observation from the layer-wise proportions, then a
    single W ~ IG(nu_s/2, nu_s/2) per equation (1), then z^(h) from (3) and each
    z^(l-1) from (4), reusing that same W at every transition. Returns
    (y, path_index, w, latent) with latent[l] the layer-l state.
    """
    from dstmm.gig import sample_inverse_gamma

    h = model.depth
    keys = jax.random.split(key, h + 3)

    idx_per_layer = [
        jax.random.categorical(keys[l], model.log_pi[l], shape=(n,)) for l in range(h)
    ]
    stride = jnp.array(
        [int(jnp.prod(jnp.array(model.ks[l + 1 :]))) if l + 1 < h else 1 for l in range(h)]
    )
    path_index = sum(idx_per_layer[l] * stride[l] for l in range(h))

    nu_j = model.nu[path_index]
    w = sample_inverse_gamma(keys[h], 0.5 * nu_j, 0.5 * nu_j, shape=(n,))

    z = jnp.sqrt(w)[:, None] * jax.random.normal(
        keys[h + 1], (n, model.dims[h]), dtype=jnp.float64
    )
    latent = {h: z}
    noise_keys = jax.random.split(keys[h + 2], h)
    for l in range(h, 0, -1):
        a = idx_per_layer[l - 1]
        eta = model.eta[l - 1][a]
        lam = model.lam[l - 1][a]
        psi = model.psi_diag[l - 1][a]
        dlt = model.delta[l - 1][a]
        mean = eta + jnp.einsum("nij,nj->ni", lam, z) + w[:, None] * dlt
        eps = jax.random.normal(noise_keys[l - 1], mean.shape, dtype=jnp.float64)
        z = mean + jnp.sqrt(w)[:, None] * jnp.sqrt(psi) * eps
        latent[l - 1] = z
    return z, path_index, w, latent


def random_init(key, dims, ks, active_set=(1,), nu0=8.0, scale=1.0, df_mode="common"):
    """A rough starting point: random loadings, spread intercepts, unit specific variances."""
    h = len(ks)
    n_paths = int(jnp.prod(jnp.array(ks)))
    keys = jax.random.split(key, 3 * h)
    log_pi, eta, lam, psi_diag, delta = [], [], [], [], []
    for l in range(1, h + 1):
        r_prev, r_cur, k = dims[l - 1], dims[l], ks[l - 1]
        log_pi.append(jnp.log(jnp.full((k,), 1.0 / k)))
        eta.append(scale * jax.random.normal(keys[3 * (l - 1)], (k, r_prev), jnp.float64))
        lam.append(
            jax.random.normal(keys[3 * (l - 1) + 1], (k, r_prev, r_cur), jnp.float64)
            / jnp.sqrt(r_cur)
        )
        psi_diag.append(jnp.ones((k, r_prev)))
        if l in active_set:
            delta.append(
                0.1 * jax.random.normal(keys[3 * (l - 1) + 2], (k, r_prev), jnp.float64)
            )
        else:
            delta.append(jnp.zeros((k, r_prev)))
    return DStMM(
        dims=tuple(dims),
        ks=tuple(ks),
        log_pi=tuple(log_pi),
        eta=tuple(eta),
        lam=tuple(lam),
        psi_diag=tuple(psi_diag),
        delta=tuple(delta),
        nu=jnp.full((n_paths,), float(nu0)),
    )


def init_from_data(key, y, dims, ks, active_set=(1,), nu0=8.0, df_mode="common"):
    """A data-scaled starting point: first-layer intercepts at random observations.

    The deep-mixture likelihood is multimodal and Section 3.6 recommends multiple
    starts or a warm start from the symmetric fit. A random_init with no view of
    the data usually lands in a first-layer split that mixes the groups; putting
    the layer-1 intercepts on actual observations and scaling the loadings and
    specific variances by the observed spread fixes most of that.
    """
    h = len(ks)
    n_paths = int(jnp.prod(jnp.array(ks)))
    keys = jax.random.split(key, 3 * h + 1)
    sd = jnp.std(y, axis=0)

    log_pi, eta, lam, psi_diag, delta = [], [], [], [], []
    for l in range(1, h + 1):
        r_prev, r_cur, k = dims[l - 1], dims[l], ks[l - 1]
        log_pi.append(jnp.log(jnp.full((k,), 1.0 / k)))
        if l == 1:
            pick = jax.random.choice(keys[0], y.shape[0], (k,), replace=False)
            eta.append(y[pick])
            scale = sd
        else:
            eta.append(
                0.5 * jax.random.normal(keys[3 * (l - 1)], (k, r_prev), jnp.float64)
            )
            scale = jnp.ones(r_prev)
        lam.append(
            0.7
            * scale[None, :, None]
            * jax.random.normal(keys[3 * (l - 1) + 1], (k, r_prev, r_cur), jnp.float64)
            / jnp.sqrt(r_cur)
        )
        psi_diag.append(jnp.broadcast_to(0.5 * scale**2, (k, r_prev)))
        if l in active_set:
            delta.append(
                0.1
                * scale[None, :]
                * jax.random.normal(keys[3 * (l - 1) + 2], (k, r_prev), jnp.float64)
            )
        else:
            delta.append(jnp.zeros((k, r_prev)))
    return DStMM(
        dims=tuple(dims),
        ks=tuple(ks),
        log_pi=tuple(log_pi),
        eta=tuple(eta),
        lam=tuple(lam),
        psi_diag=tuple(psi_diag),
        delta=tuple(delta),
        nu=jnp.full((n_paths,), float(nu0)),
    )
