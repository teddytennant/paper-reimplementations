"""Training and exact population evaluation.

Section 5 fits on sampled context and next-token pairs but evaluates every
population quantity, in particular the prediction error

    epsilon_p = E_{X,j} L(p_j(X); p_hat_E(Theta_j(X)))

of Assumption 1.1, by enumeration over the d^3 observed-token contexts. Both
happen here. Optimization follows Appendix B.2: AdamW at a constant 3e-4 with
weight decay 1e-4, gradient clipping at one, and a projection of the augmented
head rows onto the radius-4 ball after every update.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import optax

from tokengeom import model as M
from tokengeom.generative import SemanticModel

jax.config.update("jax_enable_x64", True)

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
BATCH_SIZE = 512

PAPER_CHECKPOINTS = (
    0, 20, 50, 100, 200, 400, 800, 2000, 5000,
    10_000, 20_000, 50_000, 100_000, 200_000,
)


@dataclasses.dataclass
class Enumeration:
    """Enumerated contexts with their exact population weights.

    Two enumerations are used. The observed-token one has a row per d^3 context
    and is what the encoder consumes. The semantic one has a row per A^3
    semantic context; it carries the same population weight on each class and
    the same conditional, so every population quantity that depends on the
    context only through p_j(. | X) is identical, at a factor A^3/d^3 of the
    cost. Section 5 states both counts, 4096 and 110592, for the paper's sizes.
    """

    contexts: jnp.ndarray  # (n, 3) token ids, or group ids when semantic
    context_prob: jnp.ndarray  # p(X_{j,c})
    conditional: jnp.ndarray  # p_j(v | X)
    entropy: float  # sum_X p(X) sum_v p log p, the constant in the KL
    semantic: bool = False


def enumerate_contexts(sim: SemanticModel) -> Enumeration:
    d = sim.config.vocab_size
    idx = np.arange(d**3)
    contexts = np.stack([idx // d**2, (idx // d) % d, idx % d], axis=1)
    pc = sim.context_distribution_tokens
    cond = sim.token_conditional[sim.context_index(contexts)]
    with np.errstate(divide="ignore", invalid="ignore"):
        plogp = np.where(cond > 0, cond * np.log(cond), 0.0)
    entropy = float(pc @ plogp.sum(axis=1))
    return Enumeration(
        contexts=jnp.asarray(contexts),
        context_prob=jnp.asarray(pc),
        conditional=jnp.asarray(cond),
        entropy=entropy,
    )


def enumerate_semantic_contexts(sim: SemanticModel) -> Enumeration:
    """The A^3 semantic contexts.

    p_j(. | X) depends on the context only through the semantic groups of its
    three tokens, so the unrestricted representation of Section 2.2 is constant
    on each class and this enumeration solves the same population problem. The
    exponential family of Section 2.3 is unchanged too: grouping the token
    contexts of one class adds their p(X_{j,c}) and leaves Theta fixed, so the
    base measure and A are the same function.
    """
    cond = sim.token_conditional
    with np.errstate(divide="ignore", invalid="ignore"):
        plogp = np.where(cond > 0, cond * np.log(cond), 0.0)
    return Enumeration(
        contexts=jnp.asarray(sim.context_groups),
        context_prob=jnp.asarray(sim.state_distribution),
        conditional=jnp.asarray(cond),
        entropy=float(sim.state_distribution @ plogp.sum(axis=1)),
        semantic=True,
    )


def expand_to_token_contexts(values, sim: SemanticModel) -> np.ndarray:
    """Lift a per-semantic-context array to the d^3 observed-token contexts."""
    d = sim.config.vocab_size
    idx = np.arange(d**3)
    contexts = np.stack([idx // d**2, (idx // d) % d, idx % d], axis=1)
    return np.asarray(values)[sim.context_index(contexts)]


def population_log_predictions(params, enum: Enumeration, config: M.EncoderConfig,
                               chunk: int = 8192) -> jnp.ndarray:
    """log p_hat_E(v | Theta_j(X)) on every enumerated context."""
    out = []
    n = enum.contexts.shape[0]
    for start in range(0, n, chunk):
        out.append(M.log_predictions(params, enum.contexts[start : start + chunk], config))
    return jnp.concatenate(out, axis=0)


def population_loss(params, enum: Enumeration, config: M.EncoderConfig) -> float:
    """epsilon_p, the population KL of Assumption 1.1, exactly."""
    logq = population_log_predictions(params, enum, config)
    cross = -jnp.sum(enum.context_prob[:, None] * enum.conditional * logq)
    return float(cross + enum.entropy)


def make_scan(config: M.EncoderConfig, optimizer, contexts, targets):
    """One jitted scan over a block of updates.

    The updates are folded into a `lax.scan` rather than a Python loop because
    the model is small enough that per-step dispatch would dominate the run
    time of the test suite.
    """

    def one_step(carry, batch_idx):
        params, opt_state = carry
        loss, grads = jax.value_and_grad(M.cross_entropy)(
            params, contexts[batch_idx], targets[batch_idx], config
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = M.project_head(optax.apply_updates(params, updates))
        return (params, opt_state), loss

    @jax.jit
    def run(params, opt_state, batch_idx):
        (params, opt_state), losses = jax.lax.scan(one_step, (params, opt_state), batch_idx)
        return params, opt_state, losses

    return run


def train(
    sim: SemanticModel,
    config: M.EncoderConfig,
    seed: int = 0,
    n_steps: int = 4000,
    batch_size: int = BATCH_SIZE,
    n_train: int = 500_000,
    checkpoints: tuple = (),
    learning_rate: float = LEARNING_RATE,
):
    """Fit the encoder and head. Returns (params, checkpoint list).

    Each checkpoint is a (step, params) pair, giving the training path that
    Figures 2, 4 and 5 plot against epsilon_p in decreasing order.
    Appendix B.2 evaluates at PAPER_CHECKPOINTS.
    """
    contexts, targets = sim.sample_pairs(n_train, seed=seed + 991)
    contexts = jnp.asarray(contexts)
    targets = jnp.asarray(targets)

    params = M.project_head(M.init_params(config, seed))
    optimizer = optax.chain(
        optax.clip_by_global_norm(GRAD_CLIP),
        optax.adamw(learning_rate, weight_decay=WEIGHT_DECAY),
    )
    opt_state = optimizer.init(params)
    run = make_scan(config, optimizer, contexts, targets)

    rng = np.random.default_rng(seed + 17)
    marks = sorted({0, n_steps} | {int(c) for c in checkpoints if 0 <= c <= n_steps})
    saved = []
    step = 0
    for target in marks:
        if target > step:
            idx = jnp.asarray(rng.integers(0, n_train, size=(target - step, batch_size)))
            params, opt_state, _ = run(params, opt_state, idx)
            step = target
        if target in checkpoints or target == n_steps:
            saved.append((step, jax.tree.map(jnp.array, params)))
    return params, saved


def fit_unrestricted(
    sim: SemanticModel,
    enum: Enumeration,
    width: int,
    seed: int = 0,
    n_steps: int = 4000,
    learning_rate: float = 0.05,
    z_norm: float = M.REPRESENTATION_NORM,
    head_radius: float = M.HEAD_RADIUS,
):
    """Solve the population problem (1) of Section 2.2 with Theta unrestricted.

    Section 2.2 first lets the representation map be flexible enough to pick any
    Z in R^s for each context, which reduces token prediction to

        min_E E_{X,j} L(p_j(X); M_E),   M_E = {p_hat_E(Z) : Z in R^s}.

    That is what this fits: one free vector per enumerated context plus the
    shared head, on the exact population objective, with the same norm
    constraints Appendix B.2 puts on the encoder output and the head rows. It is
    how the tests reach an epsilon_p small enough for the error terms in
    Theorem 1 and Assumption 4 to be informative, which a small transformer
    trained for a few thousand steps cannot do.

    Returns a params dict with the same keys the encoder uses plus "z", so
    :func:`tokengeom.geometry` can treat it like any other fitted pair.
    """
    rng = np.random.default_rng(seed)
    n = enum.contexts.shape[0]
    d = sim.config.vocab_size
    scale = 1.0 / np.sqrt(width)
    params = {
        "embed": jnp.asarray(rng.standard_normal((d, width)) * scale),
        "head_bias": jnp.zeros(d),
        "z": jnp.asarray(rng.standard_normal((n, width)) * scale),
    }
    weight = enum.context_prob
    target = enum.conditional

    def project(p):
        z = z_norm * p["z"] / jnp.linalg.norm(p["z"], axis=1, keepdims=True)
        norms = jnp.sqrt(jnp.sum(p["embed"] ** 2, axis=1) + p["head_bias"] ** 2)
        factor = jnp.minimum(1.0, head_radius / jnp.maximum(norms, 1e-12))
        return {
            "embed": p["embed"] * factor[:, None],
            "head_bias": p["head_bias"] * factor,
            "z": z,
        }

    def loss(p):
        q = jax.nn.log_softmax(p["z"] @ p["embed"].T + p["head_bias"], axis=-1)
        return -jnp.sum(weight[:, None] * target * q)

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    def one_step(carry, _):
        p, state = carry
        value, grads = jax.value_and_grad(loss)(p)
        updates, state = optimizer.update(grads, state, p)
        return (project(optax.apply_updates(p, updates)), state), value

    @jax.jit
    def run(p, state):
        (p, state), values = jax.lax.scan(one_step, (p, state), None, length=n_steps)
        return p, state, values

    params, _, values = run(project(params), opt_state)
    return params, float(values[-1] + enum.entropy)


def unrestricted_population_loss(params, enum: Enumeration) -> float:
    """epsilon_p for a pair fitted by :func:`fit_unrestricted`."""
    q = jax.nn.log_softmax(params["z"] @ params["embed"].T + params["head_bias"], axis=-1)
    return float(-jnp.sum(enum.context_prob[:, None] * enum.conditional * q) + enum.entropy)
