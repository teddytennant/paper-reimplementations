"""NormPaST-Risk model: multi-scale encoder, paper-air SSM, segment risk.

The selective recurrence is a real diagonal SSM scanned with jax.lax.scan.
A is stored as -softplus(a_raw) so every finite channel is strictly negative.
Paper-air modulation is tilde_h = h + e * (0.75 + 0.5 * g).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from normpast_risk.segments import aggregate_segments, project_tokens


MODULATION_BIAS = 0.75
MODULATION_SCALE = 0.5
PAPER_DOMINANT_THRESHOLD = 0.5

DEFAULTS = {
    "d": 96,
    "n_layers": 2,
    "n_experts": 4,
    "n_tasks": 25,
    "gamma": 128,
    "kernel_sizes": (3, 7),
    "dilations": (1, 2),
    "dropout_rate": 0.0,
    "in_dim": 4,
    "lambda_seg": 0.1,
    "lambda_norm": 0.2,
}


def transition_A(a_raw):
    """Negative softplus parameterization. A < 0 for finite a_raw."""
    return -jax.nn.softplus(a_raw)


def paper_air_gate(h, embedding, weight, bias):
    """g = sigmoid(W_g [h; e] + b_g)."""
    joined = jnp.concatenate([h, embedding], axis=-1)
    return jax.nn.sigmoid(joined @ weight.T + bias)


def paper_air_modulate(h, embedding, gate):
    """tilde_h = h + e * (0.75 + 0.5 * g)."""
    return h + embedding * (MODULATION_BIAS + MODULATION_SCALE * gate)


def ssm_step(v, abar, bbar, c, tilde_h):
    """One selective step: v_new = Abar * v + Bbar * tilde_h, y = C * v_new."""
    v_new = abar * v + bbar * tilde_h
    y = c * v_new
    return v_new, y


def ssm_scan(abar, bbar, c, tilde_h, v0=None):
    """Scan the diagonal SSM over time with jax.lax.scan.

    Inputs are (S, d) or (batch, S, d). Returns y and the post-step states
    with the same layout. v0 defaults to zeros.
    """
    batched = abar.ndim == 3
    if not batched:
        abar = abar[None, ...]
        bbar = bbar[None, ...]
        c = c[None, ...]
        tilde_h = tilde_h[None, ...]
        if v0 is not None and v0.ndim == 1:
            v0 = v0[None, ...]
    batch, _length, dim = tilde_h.shape
    if v0 is None:
        v0 = jnp.zeros((batch, dim), dtype=tilde_h.dtype)
    else:
        v0 = jnp.asarray(v0, dtype=tilde_h.dtype)

    def step(v, xs):
        abar_j, bbar_j, c_j, h_j = xs
        v_new, y = ssm_step(v, abar_j, bbar_j, c_j, h_j)
        return v_new, (y, v_new)

    xs = tuple(jnp.moveaxis(t, 1, 0) for t in (abar, bbar, c, tilde_h))
    _final, (y_seq, v_seq) = jax.lax.scan(step, v0, xs)
    y = jnp.moveaxis(y_seq, 0, 1)
    v = jnp.moveaxis(v_seq, 0, 1)
    if not batched:
        return y[0], v[0]
    return y, v


def layer_norm(x, scale, bias, eps: float = 1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    centered = x - mean
    var = jnp.mean(jnp.square(centered), axis=-1, keepdims=True)
    return scale * centered * jax.lax.rsqrt(var + eps) + bias


def _same_padding(kernel_size: int, dilation: int):
    effective = (kernel_size - 1) * dilation + 1
    total = effective - 1
    left = total // 2
    right = total - left
    return (left, right)


def conv1d_same(x, kernel, bias, dilation: int):
    """Map (batch, T, C_in) through a same-length dilated conv, then add bias."""
    x_nct = jnp.transpose(x, (0, 2, 1))
    pad = _same_padding(kernel.shape[-1], int(dilation))
    y = jax.lax.conv_general_dilated(
        x_nct,
        kernel,
        window_strides=(1,),
        padding=(pad,),
        rhs_dilation=(int(dilation),),
        dimension_numbers=("NCH", "OIH", "NCH"),
    )
    return jnp.transpose(y, (0, 2, 1)) + bias


def _dropout(x, rate, rng):
    if rate is None or rate == 0.0:
        return x, rng
    if rng is None:
        raise ValueError("rng is required when dropout_rate > 0")
    rng, sub = jax.random.split(rng)
    keep = 1.0 - float(rate)
    mask = jax.random.bernoulli(sub, p=keep, shape=x.shape)
    return x * mask.astype(x.dtype) / jnp.asarray(keep, dtype=x.dtype), rng


def _glorot(key, shape):
    shape = tuple(shape)
    if len(shape) == 1:
        fan_in, fan_out = shape[0], 1
    elif len(shape) == 2:
        fan_out, fan_in = shape
    elif len(shape) == 3:
        fan_in = shape[1] * shape[2]
        fan_out = shape[0] * shape[2]
    else:
        raise ValueError(f"unsupported shape {shape}")
    limit = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit, dtype=jnp.float32)


def _normal(key, shape, std=0.02):
    return jax.random.normal(key, shape, dtype=jnp.float32) * std


def _zeros(shape):
    return jnp.zeros(shape, dtype=jnp.float32)


def _split(key):
    key, sub = jax.random.split(key)
    return key, sub


def init_params(
    key,
    d: int = DEFAULTS["d"],
    n_layers: int = DEFAULTS["n_layers"],
    n_experts: int = DEFAULTS["n_experts"],
    n_tasks: int = DEFAULTS["n_tasks"],
    in_dim: int = DEFAULTS["in_dim"],
    kernel_sizes=DEFAULTS["kernel_sizes"],
    dilations=DEFAULTS["dilations"],
    expert_hidden: int | None = None,
):
    """Initialize a parameter pytree. a_raw starts at 0, so A = -softplus(0)."""
    if isinstance(key, int):
        key = jax.random.PRNGKey(key)
    kernel_sizes = tuple(kernel_sizes)
    dilations = tuple(dilations)
    if len(kernel_sizes) != len(dilations):
        raise ValueError("kernel_sizes and dilations must have the same length")
    if expert_hidden is None:
        expert_hidden = d

    kernels = []
    biases = []
    for width in kernel_sizes:
        key, sub = _split(key)
        kernels.append(_glorot(sub, (d, in_dim, int(width))))
        biases.append(_zeros((d,)))
    key, sub = _split(key)
    w_f = _glorot(sub, (d, len(kernel_sizes) * d))
    b_f = _zeros((d,))

    key, sub = _split(key)
    w_s = _glorot(sub, (d, 2 * d + 2))
    b_s = _zeros((d,))

    key, sub = _split(key)
    e_state = _normal(sub, (2, d))

    layers = []
    for _ in range(n_layers):
        key, k_g = _split(key)
        key, k_delta = _split(key)
        key, k_b = _split(key)
        key, k_c = _split(key)
        layers.append(
            {
                "W_g": _normal(k_g, (d, 2 * d)),
                "b_g": _zeros((d,)),
                "a_raw": _zeros((d,)),
                "W_delta": _normal(k_delta, (d, d)),
                "b_delta": _zeros((d,)),
                "W_B": _normal(k_b, (d, d)),
                "b_B": _zeros((d,)),
                "W_C": _normal(k_c, (d, d)),
                "b_C": _zeros((d,)),
                "ln_scale": jnp.ones((d,), dtype=jnp.float32),
                "ln_bias": _zeros((d,)),
            }
        )

    key, sub = _split(key)
    w_n = _glorot(sub, (d, d))
    b_n = _zeros((d,))

    key, sub = _split(key)
    task_embed = _normal(sub, (n_tasks, d))
    key, sub = _split(key)
    w_r = _glorot(sub, (n_experts, 2 * d))
    b_r = _zeros((n_experts,))

    expert_w = []
    expert_b = []
    expert_out = []
    expert_out_b = []
    for _ in range(n_experts):
        key, sub = _split(key)
        expert_w.append(_glorot(sub, (expert_hidden, d)))
        expert_b.append(_zeros((expert_hidden,)))
        key, sub = _split(key)
        expert_out.append(_glorot(sub, (expert_hidden,)))
        expert_out_b.append(jnp.asarray(0.0, dtype=jnp.float32))

    key, sub = _split(key)
    w_a = _normal(sub, (d,))
    return {
        "encoder": {
            "kernels": tuple(kernels),
            "biases": tuple(biases),
            "W_f": w_f,
            "b_f": b_f,
        },
        "segment": {"W_s": w_s, "b_s": b_s},
        "e_state": e_state,
        "layers": tuple(layers),
        "normative": {"W_n": w_n, "b_n": b_n},
        "risk": {
            "task_embed": task_embed,
            "W_r": w_r,
            "b_r": b_r,
            "W_expert": jnp.stack(expert_w, axis=0),
            "b_expert": jnp.stack(expert_b, axis=0),
            "w_expert_out": jnp.stack(expert_out, axis=0),
            "b_expert_out": jnp.stack(expert_out_b, axis=0),
            "w_chi": jnp.asarray(1.0, dtype=jnp.float32),
            "w_delta": jnp.asarray(1.0, dtype=jnp.float32),
            "w_a": w_a,
            "w_p": jnp.asarray(1.0, dtype=jnp.float32),
            "b_p": jnp.asarray(0.0, dtype=jnp.float32),
        },
    }


def layer_transition(params):
    """A for every SSM layer, each strictly negative under the softplus map."""
    return tuple(transition_A(layer["a_raw"]) for layer in params["layers"])


def encode(params, x, dilations, dropout_rate=0.0, rng=None):
    streams = []
    for kernel, bias, dilation in zip(
        params["encoder"]["kernels"], params["encoder"]["biases"], dilations
    ):
        streams.append(jax.nn.gelu(conv1d_same(x, kernel, bias, dilation)))
    fused = jnp.concatenate(streams, axis=-1)
    h = fused @ params["encoder"]["W_f"].T + params["encoder"]["b_f"]
    h, rng = _dropout(h, dropout_rate, rng)
    return h, rng


def apply_ssm_layer(layer, h, e_table, rho, dropout_rate=0.0, rng=None):
    """One paper-air selective SSM layer. h is (batch, S, d)."""
    index = (rho >= PAPER_DOMINANT_THRESHOLD).astype(jnp.int32)
    embedding = e_table[index]
    gate = paper_air_gate(h, embedding, layer["W_g"], layer["b_g"])
    tilde = paper_air_modulate(h, embedding, gate)
    delta = jax.nn.softplus(tilde @ layer["W_delta"].T + layer["b_delta"])
    b_coeff = tilde @ layer["W_B"].T + layer["b_B"]
    c_coeff = tilde @ layer["W_C"].T + layer["b_C"]
    a_coeff = transition_A(layer["a_raw"])
    abar = jnp.exp(delta * a_coeff)
    bbar = delta * b_coeff
    y, v = ssm_scan(abar, bbar, c_coeff, tilde)
    y, rng = _dropout(y, dropout_rate, rng)
    h_out = layer_norm(h + y, layer["ln_scale"], layer["ln_bias"])
    aux = {
        "g": gate,
        "tilde_h": tilde,
        "A": a_coeff,
        "embedding": embedding,
        "v": v,
    }
    return h_out, aux, rng


def _risk_head(risk, tokens, states, task_id, hat_next):
    """Task-aware experts on segment tokens, calibrated by chi and delta.

    The sample probability is sigmoid(w_p * sum_j beta_j * z_j + b_p), where
    z_j is the pre-sigmoid segment logit and beta is a softmax over states.
    """
    batch, n_seg, _dim = states.shape
    task_embed = risk["task_embed"][task_id]
    pooled = jnp.mean(states, axis=1)
    router_logits = jnp.concatenate([pooled, task_embed], axis=-1) @ risk["W_r"].T + risk["b_r"]
    alpha = jax.nn.softmax(router_logits, axis=-1)

    hidden = jnp.einsum("bsd,khd->bskh", tokens, risk["W_expert"]) + risk["b_expert"]
    hidden = jax.nn.gelu(hidden)
    expert_score = jnp.einsum("bskh,kh->bsk", hidden, risk["w_expert_out"]) + risk["b_expert_out"]
    mixed = jnp.einsum("bk,bsk->bs", alpha, expert_score)

    change = jnp.linalg.norm(states[:, 1:] - states[:, :-1], axis=-1)
    chi = jnp.concatenate(
        [jnp.zeros((batch, 1), dtype=states.dtype), change],
        axis=1,
    )
    # Detach the deviation used for calibration so L_cls and L_seg do not
    # train the normative head. L_norm still sees the live prediction.
    deviation = jnp.linalg.norm(tokens[:, 1:] - jax.lax.stop_gradient(hat_next), axis=-1)
    delta = jnp.concatenate(
        [jnp.zeros((batch, 1), dtype=states.dtype), deviation],
        axis=1,
    )
    z = mixed + risk["w_chi"] * chi + risk["w_delta"] * jax.lax.stop_gradient(delta)
    segment_risk = jax.nn.sigmoid(z)

    attn_logits = jnp.einsum("bsd,d->bs", states, risk["w_a"])
    beta = jax.nn.softmax(attn_logits, axis=-1)
    sample_logit = risk["w_p"] * jnp.sum(beta * z, axis=-1) + risk["b_p"]
    sample_prob = jax.nn.sigmoid(sample_logit)
    return {
        "segment_logit": z,
        "segment_risk": segment_risk,
        "sample_logit": sample_logit,
        "sample_prob": sample_prob,
        "chi": chi,
        "delta": delta,
        "alpha": alpha,
        "beta": beta,
        "mixed": mixed,
    }


def apply(
    params,
    x,
    q,
    task_id,
    *,
    gamma: int = DEFAULTS["gamma"],
    dilations=DEFAULTS["dilations"],
    dropout_rate: float = DEFAULTS["dropout_rate"],
    rng=None,
):
    """Forward pass.

    x is (batch, T, 4), q is (batch, T), task_id is (batch,).
    segment_risk has shape (batch, S) and sample_prob has shape (batch,).
    Both probabilities lie in (0, 1) for finite logits.
    """
    x = jnp.asarray(x, dtype=jnp.float32)
    q = jnp.asarray(q, dtype=jnp.float32)
    task_id = jnp.asarray(task_id, dtype=jnp.int32)
    if x.ndim != 3:
        raise ValueError("x must have shape (batch, T, channels)")
    if q.shape != x.shape[:2]:
        raise ValueError("q must have shape (batch, T)")
    if task_id.shape != (x.shape[0],):
        raise ValueError("task_id must have shape (batch,)")

    h, rng = encode(params, x, dilations, dropout_rate=dropout_rate, rng=rng)
    aggregated = aggregate_segments(h, q, gamma)
    tokens = project_tokens(
        aggregated["features"], params["segment"]["W_s"], params["segment"]["b_s"]
    )
    states = tokens
    layer_aux = []
    for layer in params["layers"]:
        states, aux, rng = apply_ssm_layer(
            layer,
            states,
            params["e_state"],
            aggregated["rho"],
            dropout_rate=dropout_rate,
            rng=rng,
        )
        layer_aux.append(aux)

    hat_next = states[:, :-1, :] @ params["normative"]["W_n"].T + params["normative"]["b_n"]
    risk = _risk_head(params["risk"], tokens, states, task_id, hat_next)
    outputs = {
        "segment_risk": risk["segment_risk"],
        "sample_prob": risk["sample_prob"],
        "segment_logit": risk["segment_logit"],
        "sample_logit": risk["sample_logit"],
        "tokens": tokens,
        "states": states,
        "hat_next": hat_next,
        "paper_mean": aggregated["paper_mean"],
        "air_mean": aggregated["air_mean"],
        "n_paper": aggregated["n_paper"],
        "n_air": aggregated["n_air"],
        "rho": aggregated["rho"],
        "paper_dominant": aggregated["paper_dominant"],
        "features": aggregated["features"],
        "chi": risk["chi"],
        "delta": risk["delta"],
        "alpha": risk["alpha"],
        "beta": risk["beta"],
        "A": tuple(aux["A"] for aux in layer_aux),
        "point_hidden": h,
    }
    return outputs


class NormPaSTRisk:
    """Paper-air state-space network for segment-level Alzheimer's risk.

    Constructor defaults follow the transcribed settings: hidden size 96,
    2 SSM layers, 4 experts, kernels (3, 7) with dilations (1, 2), segment
    stride 128, and 25 tasks. Dropout defaults to 0 so CPU tests stay
    deterministic. Pass dropout_rate=0.3 to match the paper table.
    """

    def __init__(
        self,
        d: int = DEFAULTS["d"],
        n_layers: int = DEFAULTS["n_layers"],
        n_experts: int = DEFAULTS["n_experts"],
        n_tasks: int = DEFAULTS["n_tasks"],
        gamma: int = DEFAULTS["gamma"],
        kernel_sizes=DEFAULTS["kernel_sizes"],
        dilations=DEFAULTS["dilations"],
        dropout_rate: float = DEFAULTS["dropout_rate"],
        in_dim: int = DEFAULTS["in_dim"],
        lambda_seg: float = DEFAULTS["lambda_seg"],
        lambda_norm: float = DEFAULTS["lambda_norm"],
        expert_hidden: int | None = None,
        key=0,
    ):
        self.d = d
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.n_tasks = n_tasks
        self.gamma = gamma
        self.kernel_sizes = tuple(kernel_sizes)
        self.dilations = tuple(dilations)
        self.dropout_rate = dropout_rate
        self.in_dim = in_dim
        self.lambda_seg = lambda_seg
        self.lambda_norm = lambda_norm
        self.expert_hidden = d if expert_hidden is None else expert_hidden
        self.params = init_params(
            key,
            d=d,
            n_layers=n_layers,
            n_experts=n_experts,
            n_tasks=n_tasks,
            in_dim=in_dim,
            kernel_sizes=self.kernel_sizes,
            dilations=self.dilations,
            expert_hidden=self.expert_hidden,
        )

    def apply(self, x, q, task_id, rng=None, params=None):
        return apply(
            self.params if params is None else params,
            x,
            q,
            task_id,
            gamma=self.gamma,
            dilations=self.dilations,
            dropout_rate=self.dropout_rate,
            rng=rng,
        )

    def __call__(self, x, q, task_id, rng=None, params=None):
        return self.apply(x, q, task_id, rng=rng, params=params)

    def loss(self, x, q, task_id, y, rng=None, params=None):
        from normpast_risk.loss import total_loss

        outputs = self.apply(x, q, task_id, rng=rng, params=params)
        return total_loss(
            outputs,
            y,
            lambda_seg=self.lambda_seg,
            lambda_norm=self.lambda_norm,
        )
