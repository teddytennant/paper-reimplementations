"""Pullback Fisher of a univariate Gaussian predictive family."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp


def gaussian_fim(mu, v):
    """Fisher information of N(mu, v) in the (mu, v) parameterization.

    I = diag(1/v, 1/(2 v^2)). mu is part of the parameter vector but does
    not appear in this closed form.
    """
    mu = jnp.asarray(mu)
    v = jnp.asarray(v)
    # Tie the result dtype to both parameters without changing the value.
    v = v + 0.0 * mu
    return jnp.diag(jnp.stack([1.0 / v, 1.0 / (2.0 * v * v)]))


def pullback_fisher(mu_and_v_fn, x):
    """g(x) = J(x)^T I(phi(x)) J(x), J = d(mu, v)/dx with shape (2, d)."""
    x = jnp.asarray(x)

    def phi(z):
        mu, v = mu_and_v_fn(z)
        return jnp.stack([jnp.asarray(mu), jnp.asarray(v)])

    jac = jax.jacrev(phi)(x)
    mu, v = mu_and_v_fn(x)
    info = gaussian_fim(mu, v)
    return jac.T @ info @ jac


def mc_pullback_diag(mu_and_v_fn, x, key, n_samples):
    """Monte Carlo diagonal of the pullback Fisher.

    y_s ~ N(mu(x), v(x)), and
    g_jj_hat = (1/S) sum_s (d/dx_j log p(y_s; phi(x)))^2.
    The score is taken with respect to x through phi, with y held fixed.
    """
    x = jnp.asarray(x)
    mu, v = mu_and_v_fn(x)
    y_s = mu + jnp.sqrt(v) * jax.random.normal(key, (n_samples,))

    def squared_score(y):
        def log_p(z):
            mu_z, v_z = mu_and_v_fn(z)
            mu_z = jnp.asarray(mu_z)
            v_z = jnp.asarray(v_z)
            return -0.5 * jnp.log(2.0 * jnp.pi * v_z) - 0.5 * (y - mu_z) ** 2 / v_z

        return jax.grad(log_p)(x) ** 2

    return jnp.mean(jax.vmap(squared_score)(y_s), axis=0)


def acquisition_sensitivity(alpha_bar, theta, key, n_samples):
    """C_alpha(theta) = E_zeta[ grad_theta alpha_bar^T I(theta)^{-1} grad_theta alpha_bar ]."""
    theta = jnp.asarray(theta)
    zetas = jax.random.normal(key, (n_samples,))
    info = gaussian_fim(theta[0], theta[1])
    inv_diag = 1.0 / jnp.diag(info)

    def one(zeta):
        grad_theta = jax.grad(lambda th: alpha_bar(th, zeta))(theta)
        return jnp.dot(grad_theta * inv_diag, grad_theta)

    return jnp.mean(jax.vmap(one)(zetas))


def gradient_bound(alpha_bar, mu_and_v_fn, x, key, n_samples):
    """||grad_x alpha||_2 <= sqrt(C_alpha(theta) * tr(g(x)))."""
    x = jnp.asarray(x)
    mu, v = mu_and_v_fn(x)
    theta = jnp.stack([jnp.asarray(mu), jnp.asarray(v)])
    c_alpha = acquisition_sensitivity(alpha_bar, theta, key, n_samples)
    metric = pullback_fisher(mu_and_v_fn, x)
    return jnp.sqrt(c_alpha * jnp.trace(metric))


def grad_norm(alpha_mean_fn, x):
    """Euclidean norm of the gradient of a deterministic acquisition alpha(x)."""
    return jnp.linalg.norm(jax.grad(alpha_mean_fn)(jnp.asarray(x)))
