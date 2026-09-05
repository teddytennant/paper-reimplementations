"""MaskCode: the inner feedback encoder and decoder of Section IV.

Layout of the token axis for every phase t >= 2 and for the decoder: the first n
tokens are the variable nodes of the outer code's Tanner graph, carrying the
code history (17)/(27), and the last n-k tokens are the check nodes, carrying
the soft syndrome (19)/(28). That is exactly the ordering the adjacency matrix
(23) assumes, which is why the attention mask and the input embedding line up.
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp

from .channel import ChannelConfig, feedback, forward, normalize_power
from .codes import LinearBlockCode
from .ops import llr_from_erf
from .transformer import TransformerStack


@dataclass(frozen=True)
class MaskCodeConfig:
    """Architecture knobs. `width` is the embedding dimension l of (14) and (20)."""

    width: int = 32
    n_layers: int = 2
    n_heads: int = 4
    mlp_mult: int = 2
    phases: int = 3
    use_mask: bool = True
    use_soft_syndrome: bool = True


class InitialEncoder(eqx.Module):
    """Section IV-A, the t = 1 encoder.

    E_1 = (w_1)^T (2c - 1) + P_1                                       (14)
    o_1 = Trans(E_1; O_n, phi_1)                                       (15)
    x_1 = abs(o_1) * (2c - 1)                                          (16)

    (16) forces the sign of x_1 to be the sign of the BPSK codeword. That is
    what lets the receiver and, through the feedback link, the transmitter form
    a soft syndrome from y_1 at all. The paper is explicit that the constraint
    applies only at t = 1; carrying it into later phases would reduce them to
    amplitude modulation.
    """

    w_in: jax.Array  # w_1^(e), shape (width,)
    pos: jax.Array  # P_1^(e), shape (n, width)
    trans: TransformerStack

    def __init__(self, code: LinearBlockCode, cfg: MaskCodeConfig, *, key: jax.Array):
        kw, kp, kt = jax.random.split(key, 3)
        self.w_in = jax.random.normal(kw, (cfg.width,)) * 0.5
        self.pos = jax.random.normal(kp, (code.n, cfg.width)) * 0.02
        self.trans = TransformerStack(
            cfg.width, cfg.n_layers, cfg.n_heads, cfg.mlp_mult, key=kt
        )

    def __call__(self, bipolar_c: jax.Array, mask: jax.Array) -> jax.Array:
        emb = bipolar_c[:, None] * self.w_in[None, :] + self.pos
        o = self.trans(emb, mask)
        return jnp.abs(o) * bipolar_c


class SyndromeEncoder(eqx.Module):
    """Section IV-B, the t >= 2 encoder, and (with T rows instead of t) the decoder.

    Code part      E^(e,c)_t = [2c-1; y~_1; ...; y~_(t-1)]  in R^(t x n)   (17)
    Syndrome part  s^(e) from (18) and (19)                 in R^(1 x (n-k))
    Embedding      E_t = [ W^(e,c)_t E^(e,c)_t | (w^(e,s)_t)^T s^(e) ] + P_t  (20)
    Output         o_t = Trans(E_t; M, phi_t)                              (25)
                   x_t = (o_t)_{1:n}                                       (26)

    The decoder (IV-C) is the same module with the code rows replaced by
    [y_1; ...; y_T] (27), the syndrome taken from y_1 with sigma_ff instead of
    sigma_eff (28), and no truncation of the head output until (31).
    """

    w_code: jax.Array  # W^(e,c)_t, shape (rows, width)
    w_syn: jax.Array  # w^(e,s)_t, shape (width,)
    pos: jax.Array  # P_t, shape (2n-k, width)
    trans: TransformerStack
    n: int = eqx.field(static=True)

    def __init__(
        self,
        code: LinearBlockCode,
        cfg: MaskCodeConfig,
        rows: int,
        *,
        key: jax.Array,
    ):
        kc, ks, kp, kt = jax.random.split(key, 4)
        self.w_code = jax.random.normal(kc, (rows, cfg.width)) / jnp.sqrt(rows)
        self.w_syn = jax.random.normal(ks, (cfg.width,)) * 0.5
        self.pos = (
            jax.random.normal(kp, (code.n + code.n_checks, cfg.width)) * 0.02
        )
        self.trans = TransformerStack(
            cfg.width, cfg.n_layers, cfg.n_heads, cfg.mlp_mult, key=kt
        )
        self.n = code.n

    def __call__(
        self, code_rows: jax.Array, soft_syn: jax.Array, mask: jax.Array
    ) -> jax.Array:
        """code_rows is (rows, n), soft_syn is (n_checks,). Returns (2n-k,)."""
        var_tokens = code_rows.T @ self.w_code  # (n, width)
        chk_tokens = soft_syn[:, None] * self.w_syn[None, :]  # (n_checks, width)
        emb = jnp.concatenate([var_tokens, chk_tokens], axis=0) + self.pos
        return self.trans(emb, mask)


class MaskCode(eqx.Module):
    """The whole inner feedback code: T encoders and one decoder.

    Per Algorithm 1 the encoders do not share weights across phases; theta^(e)_t
    is a separate parameter set for each t.
    """

    initial: InitialEncoder
    later: tuple[SyndromeEncoder, ...]
    decoder: SyndromeEncoder
    cfg: MaskCodeConfig = eqx.field(static=True)

    def __init__(self, code: LinearBlockCode, cfg: MaskCodeConfig, *, key: jax.Array):
        keys = jax.random.split(key, cfg.phases + 1)
        self.initial = InitialEncoder(code, cfg, key=keys[0])
        # Phase t sees 2c-1 plus t-1 feedback vectors, so t rows in (17).
        self.later = tuple(
            SyndromeEncoder(code, cfg, rows=t, key=keys[t - 1])
            for t in range(2, cfg.phases + 1)
        )
        self.decoder = SyndromeEncoder(code, cfg, rows=cfg.phases, key=keys[-1])
        self.cfg = cfg


def _soft_syndrome_input(
    code: LinearBlockCode, y: jax.Array, sigma: float, enabled: bool
) -> jax.Array:
    """(18) then (19). With `enabled` False the syndrome tokens are fed zeros.

    The disabled path is the ablation of Sec. V-D: the check-node tokens still
    exist, so the token count and the mask are unchanged and only the syndrome
    information is removed.
    """
    if not enabled:
        return jnp.zeros(y.shape[:-1] + (code.n_checks,), dtype=y.dtype)
    return code.soft_syndrome(llr_from_erf(y, sigma))


def run_batch(
    model: MaskCode,
    code: LinearBlockCode,
    c: jax.Array,
    channel: ChannelConfig,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """One full forward pass of Algorithm 1 over a batch of codewords.

    `c` is (batch, n) of 0/1 floats. Returns the decoder LLRs c_hat, shape
    (batch, n), and the stack of transmitted parity messages, shape
    (T, batch, n), which is what the power constraint is checked against.
    """
    cfg = model.cfg
    full_mask = code.attention_mask() if cfg.use_mask else jnp.zeros_like(
        code.attention_mask()
    )
    zero_mask = code.zero_mask()
    bipolar = 2.0 * c - 1.0

    keys = jax.random.split(key, 2 * cfg.phases)
    ff_keys, fb_keys = keys[: cfg.phases], keys[cfg.phases :]

    # -- phase 1 ------------------------------------------------------------
    x1 = jax.vmap(model.initial, in_axes=(0, None))(bipolar, zero_mask)
    x1 = normalize_power(x1, channel.power)
    y1 = forward(x1, channel.sigma_ff, ff_keys[0])

    xs = [x1]
    ys = [y1]
    # (17) accumulates 2c-1 first, then the feedback vectors in phase order.
    history = [bipolar]

    # -- phases 2..T --------------------------------------------------------
    for idx, enc in enumerate(model.later):
        t = idx + 2
        y_tilde = feedback(ys[-1], channel.sigma_fb, fb_keys[t - 2])
        history.append(y_tilde)
        if idx == 0:
            # y~_1 is the fixed structural reference for the soft syndrome at
            # every t >= 2, so it is computed once and reused (Sec. IV-A).
            enc_syn = _soft_syndrome_input(
                code, history[1], channel.sigma_eff, cfg.use_soft_syndrome
            )
        rows = jnp.stack(history, axis=1)  # (batch, t, n)
        o = jax.vmap(enc, in_axes=(0, 0, None))(rows, enc_syn, full_mask)
        x_t = normalize_power(o[:, : code.n], channel.power)  # (26)
        xs.append(x_t)
        ys.append(forward(x_t, channel.sigma_ff, ff_keys[t - 1]))

    # -- decoder ------------------------------------------------------------
    dec_syn = _soft_syndrome_input(
        code, ys[0], channel.sigma_ff, cfg.use_soft_syndrome
    )
    dec_rows = jnp.stack(ys, axis=1)  # (batch, T, n)   eq (27)
    o_d = jax.vmap(model.decoder, in_axes=(0, 0, None))(dec_rows, dec_syn, full_mask)
    c_hat = o_d[:, : code.n]  # (31)
    return c_hat, jnp.stack(xs, axis=0)


def run_repetition(
    code: LinearBlockCode, c: jax.Array, channel: ChannelConfig, phases: int, key: jax.Array
) -> jax.Array:
    """The Repeat baseline of Sec. V-A: send 2c-1 every phase, ignore the feedback.

    Averaging T i.i.d. observations of the same symbol cuts the noise variance
    by T, so the combined LLR is 2 * T * mean(y) / sigma_ff^2.
    """
    bipolar = normalize_power(2.0 * c - 1.0, channel.power)
    keys = jax.random.split(key, phases)
    ys = jnp.stack([forward(bipolar, channel.sigma_ff, k) for k in keys], axis=0)
    return 2.0 * phases * jnp.mean(ys, axis=0) / channel.sigma_ff**2
