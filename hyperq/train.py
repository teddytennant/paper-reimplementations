"""Train a tiny bidirectional masked-diffusion model with HyperQ QKV branches.

Backbone weights are frozen. Only HyperQ branch parameters are updated.
Run from this directory:

  LD_LIBRARY_PATH=/run/current-system/sw/share/nix-ld/lib \\
    /home/nixos/arxiv-impl-work/.venv/bin/python train.py --steps 200 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for path in (str(SRC), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import jax
import jax.numpy as jnp
from jax.tree_util import tree_leaves, tree_map

from hyperq.readout import edges, emit, init_params, readout, residual

VOCAB = 32
MASK_ID = VOCAB
SEQ = 16
D_MODEL = 32
N_BLOCKS = 2
N_HEADS = 4
N_QUBITS = 8
STRIDE = 3
D_A = 8
BATCH = 8
MLP = 64
LR = 1e-3
ADAM_B1 = 0.9
ADAM_B2 = 0.999
ADAM_EPS = 1e-8
GRAD_CLIP = 1.0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train a tiny HyperQ masked-diffusion model")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def count_params(tree) -> int:
    total = 0
    for leaf in tree_leaves(tree):
        if hasattr(leaf, "dtype") and hasattr(leaf, "size") and jnp.issubdtype(leaf.dtype, jnp.floating):
            total += int(leaf.size)
    return total


def layer_norm(x, scale, bias, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return scale * (x - mean) * jax.lax.rsqrt(var + eps) + bias


def attention(qkv):
    b, s, _ = qkv.shape
    hd = D_MODEL // N_HEADS
    q, k, v = jnp.split(qkv, 3, axis=-1)

    def heads(t):
        return t.reshape(b, s, N_HEADS, hd).transpose(0, 2, 1, 3)

    q, k, v = heads(q), heads(k), heads(v)
    logits = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(jnp.asarray(hd, dtype=q.dtype))
    weights = jax.nn.softmax(logits, axis=-1)
    out = jnp.einsum("bhqk,bhkd->bhqd", weights, v)
    return out.transpose(0, 2, 1, 3).reshape(b, s, D_MODEL)


def randn(key, shape, scale=0.02):
    return scale * jax.random.normal(key, shape)


def init_backbone(key):
    k_embed, k_pos, k_un, *block_keys = jax.random.split(key, 3 + N_BLOCKS)
    blocks = []
    for bk in block_keys:
        keys = jax.random.split(bk, 4)
        blocks.append({
            "ln1_s": jnp.ones((D_MODEL,)),
            "ln1_b": jnp.zeros((D_MODEL,)),
            "wqkv": randn(keys[0], (D_MODEL, 3 * D_MODEL)),
            "bqkv": jnp.zeros((3 * D_MODEL,)),
            "wo": randn(keys[1], (D_MODEL, D_MODEL)),
            "bo": jnp.zeros((D_MODEL,)),
            "ln2_s": jnp.ones((D_MODEL,)),
            "ln2_b": jnp.zeros((D_MODEL,)),
            "w1": randn(keys[2], (D_MODEL, MLP)),
            "b1": jnp.zeros((MLP,)),
            "w2": randn(keys[3], (MLP, D_MODEL)),
            "b2": jnp.zeros((D_MODEL,)),
        })
    return {
        "embed": randn(k_embed, (VOCAB + 1, D_MODEL)),
        "pos": randn(k_pos, (SEQ, D_MODEL)),
        "blocks": blocks,
        "ln_f_s": jnp.ones((D_MODEL,)),
        "ln_f_b": jnp.zeros((D_MODEL,)),
        "unembed": randn(k_un, (D_MODEL, VOCAB)),
        "unembed_b": jnp.zeros((VOCAB,)),
    }


def init_branches(key, n_qubits):
    keys = jax.random.split(key, N_BLOCKS * 2)
    branches = []
    for i in range(N_BLOCKS):
        k_hq, k_b = keys[2 * i], keys[2 * i + 1]
        branches.append({
            "hq": init_params(k_hq, n_qubits, D_MODEL, D_A),
            "W_B": randn(k_b, (3 * D_MODEL, n_qubits)),
            "s": jnp.array(1.0, dtype=jnp.float32),
            "r": jnp.array(1.0, dtype=jnp.float32),
        })
    return branches


def hq_delta(branch, h, incident):
    """Map each token hidden state through emit -> readout -> residual onto QKV."""
    b, s, d = h.shape
    flat = h.reshape((b * s, d))

    def one(vec):
        alpha, omega, psi = emit(vec, branch["hq"])
        z = readout(alpha, omega, psi, incident)
        r = jnp.maximum(branch["r"], jnp.asarray(1e-3, dtype=branch["r"].dtype))
        return residual(z, branch["W_B"], branch["s"], r)

    return jax.vmap(one)(flat).reshape(b, s, 3 * D_MODEL)


def block_forward(x, block, branch, incident):
    h = layer_norm(x, block["ln1_s"], block["ln1_b"])
    qkv = h @ block["wqkv"] + block["bqkv"]
    qkv = qkv + hq_delta(branch, h, incident)
    attn = attention(qkv)
    x = x + attn @ block["wo"] + block["bo"]
    h2 = layer_norm(x, block["ln2_s"], block["ln2_b"])
    hidden = jax.nn.gelu(h2 @ block["w1"] + block["b1"])
    x = x + hidden @ block["w2"] + block["b2"]
    return x


def forward(branches, backbone, tokens, incident):
    backbone = tree_map(jax.lax.stop_gradient, backbone)
    x = backbone["embed"][tokens] + backbone["pos"][None, :, :]
    for i in range(N_BLOCKS):
        x = block_forward(x, backbone["blocks"][i], branches[i], incident)
    x = layer_norm(x, backbone["ln_f_s"], backbone["ln_f_b"])
    return x @ backbone["unembed"] + backbone["unembed_b"]


def diffusion_loss(branches, backbone, tokens, key, incident):
    """t uniform on (0, 1]; mean over masked tokens of -log p(true token) / t."""
    b, s = tokens.shape
    t_key, m_key = jax.random.split(key)
    t = 1.0 - jax.random.uniform(t_key, (b, 1))
    mask = jax.random.uniform(m_key, (b, s)) < t
    masked = jnp.where(mask, MASK_ID, tokens)
    logits = forward(branches, backbone, masked, incident)
    logp = jax.nn.log_softmax(logits)
    token_logp = jnp.take_along_axis(logp, tokens[..., None], axis=-1)[..., 0]
    weighted = (-token_logp) / t
    mask_f = mask.astype(jnp.float32)
    return jnp.sum(weighted * mask_f) / jnp.maximum(jnp.sum(mask_f), 1.0)


def synthetic_corpus(key, n_seq=64):
    k1, k2 = jax.random.split(key)
    start = jax.random.randint(k1, (n_seq, 1), 0, VOCAB)
    step = jax.random.randint(k2, (n_seq, 1), 1, 5)
    pos = jnp.arange(SEQ)[None, :]
    return (start + step * pos) % VOCAB


def sample_batch(corpus, key):
    idx = jax.random.randint(key, (BATCH,), 0, corpus.shape[0])
    return corpus[idx]


def adam_init(params):
    zeros = tree_map(jnp.zeros_like, params)
    return {"m": zeros, "v": zeros, "t": jnp.array(0, dtype=jnp.int32)}


def clip_grads(grads, max_norm=GRAD_CLIP):
    sq = sum(jnp.sum(jnp.square(g)) for g in tree_leaves(grads))
    scale = jnp.minimum(1.0, max_norm / (jnp.sqrt(sq) + 1e-6))
    return tree_map(lambda g: g * scale, grads)


def adam_apply(params, grads, state, lr=LR):
    t = state["t"] + 1
    m = tree_map(lambda m, g: ADAM_B1 * m + (1.0 - ADAM_B1) * g, state["m"], grads)
    v = tree_map(lambda v, g: ADAM_B2 * v + (1.0 - ADAM_B2) * jnp.square(g), state["v"], grads)
    mhat = tree_map(lambda m: m / (1.0 - ADAM_B1 ** t), m)
    vhat = tree_map(lambda v: v / (1.0 - ADAM_B2 ** t), v)
    new_params = tree_map(
        lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + ADAM_EPS),
        params,
        mhat,
        vhat,
    )
    return new_params, {"m": m, "v": v, "t": t}


@jax.jit
def train_step(branches, opt_state, backbone, tokens, key, incident):
    loss, grads = jax.value_and_grad(diffusion_loss)(branches, backbone, tokens, key, incident)
    grads = clip_grads(grads)
    branches, opt_state = adam_apply(branches, grads, opt_state)
    return branches, opt_state, loss


def main(argv=None):
    args = parse_args(argv)
    if args.steps < 1:
        raise SystemExit("--steps must be >= 1")

    key = jax.random.PRNGKey(args.seed)
    k_bb, k_br, k_data, k_loop = jax.random.split(key, 4)

    graph = edges(N_QUBITS, STRIDE)
    degree = int(graph.incident.shape[1])
    incident = graph.incident
    branches = init_branches(k_br, N_QUBITS)
    backbone = init_backbone(k_bb)
    opt_state = adam_init(branches)
    corpus = synthetic_corpus(k_data)
    n_branch = count_params(branches)

    loss_start = None
    loss_end = None
    for step in range(args.steps):
        k_loop, k_batch, k_loss = jax.random.split(k_loop, 3)
        tokens = sample_batch(corpus, k_batch)
        branches, opt_state, loss = train_step(
            branches, opt_state, backbone, tokens, k_loss, incident
        )
        if step == 0 or step == args.steps - 1:
            value = float(loss)
            print(f"step {step} loss {value:.6f}", flush=True)
            if step == 0:
                loss_start = value
            if step == args.steps - 1:
                loss_end = value

    payload = {
        "loss_start": loss_start,
        "loss_end": loss_end,
        "n_qubits": N_QUBITS,
        "degree": degree,
        "branch_params": n_branch,
        "backbone_frozen": True,
    }
    (ROOT / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


if __name__ == "__main__":
    main()
