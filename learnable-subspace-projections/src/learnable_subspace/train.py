"""Train unconstrained subspace bases on a frozen two-layer MLP."""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from learnable_subspace.projector import forward_input, orthogonal_factor, orthogonality_penalty


class TwoLayerMLP(eqx.Module):
    """Width-matched MLP. W and b are frozen. Only V is trained."""

    W1: jax.Array
    b1: jax.Array
    W2: jax.Array
    b2: jax.Array
    V1: jax.Array
    V2: jax.Array

    def logits(self, x, alpha=1.0, mask1=None, mask2=None):
        U1 = orthogonal_factor(self.V1)
        U2 = orthogonal_factor(self.V2)
        hidden = forward_input(self.W1, x, self.b1, U1, alpha=alpha, mask=mask1)
        hidden = jax.nn.gelu(hidden)
        return forward_input(self.W2, hidden, self.b2, U2, alpha=alpha, mask=mask2)


def softmax_kl(logits_p, logits_q):
    """Mean KL(softmax(logits_p) || softmax(logits_q)) over the batch."""
    log_p = jax.nn.log_softmax(logits_p, axis=-1)
    log_q = jax.nn.log_softmax(logits_q, axis=-1)
    p = jnp.exp(log_p)
    return jnp.sum(p * (log_p - log_q), axis=-1).mean()


def subspace_loss(V1, V2, W1, b1, W2, b2, x, lambda_ort):
    """KL(dense || compressed) + lambda_ort * L_ort.

    The teacher is the same net at alpha = 0 and is stopped so it has no
    gradient. W and b are used but are not arguments of the differentiated
    pytree in fit_mlp.
    """
    U1 = orthogonal_factor(V1)
    U2 = orthogonal_factor(V2)

    def logits(xi, alpha):
        hidden = forward_input(W1, xi, b1, U1, alpha=alpha)
        hidden = jax.nn.gelu(hidden)
        return forward_input(W2, hidden, b2, U2, alpha=alpha)

    student = jax.vmap(lambda xi: logits(xi, 1.0))(x)
    teacher = jax.vmap(lambda xi: logits(xi, 0.0))(x)
    teacher = jax.lax.stop_gradient(teacher)
    kl = softmax_kl(teacher, student)
    penalty = orthogonality_penalty(V1) + orthogonality_penalty(V2)
    return kl + lambda_ort * penalty


def init_mlp(key, width=16, k=4, batch=32):
    """Frozen Gaussian weights, zero bias, Gaussian V, fixed-shape batch."""
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    scale = 1.0 / jnp.sqrt(width)
    W1 = jax.random.normal(k1, (width, width)) * scale
    W2 = jax.random.normal(k2, (width, width)) * scale
    b1 = jnp.zeros((width,))
    b2 = jnp.zeros((width,))
    V1 = jax.random.normal(k3, (width, k))
    V2 = jax.random.normal(k4, (width, k))
    x = jax.random.normal(k5, (batch, width))
    model = TwoLayerMLP(W1=W1, b1=b1, W2=W2, b2=b2, V1=V1, V2=V2)
    return model, x


def fit_mlp(model, x, steps=40, lr=1e-2, lambda_ort=1e-3):
    """Adam on V only. Returns the updated module and the loss trace.

    losses[0] is the loss before any update. losses[-1] is the loss after
    `steps` updates. Weight and bias arrays are not passed to the optimizer.
    """
    optimizer = optax.adam(lr)
    opt_state = optimizer.init((model.V1, model.V2))
    W1, b1, W2, b2 = model.W1, model.b1, model.W2, model.b2

    @jax.jit
    def step(V1, V2, opt_state):
        def loss_fn(Vs):
            return subspace_loss(Vs[0], Vs[1], W1, b1, W2, b2, x, lambda_ort)

        _loss, grads = jax.value_and_grad(loss_fn)((V1, V2))
        updates, opt_state = optimizer.update(grads, opt_state, (V1, V2))
        V1, V2 = optax.apply_updates((V1, V2), updates)
        return V1, V2, opt_state

    V1, V2 = model.V1, model.V2
    losses = [subspace_loss(V1, V2, W1, b1, W2, b2, x, lambda_ort)]
    for _ in range(steps):
        V1, V2, opt_state = step(V1, V2, opt_state)
        losses.append(subspace_loss(V1, V2, W1, b1, W2, b2, x, lambda_ort))
    model = eqx.tree_at(lambda m: (m.V1, m.V2), model, (V1, V2))
    return model, jnp.stack(losses)
