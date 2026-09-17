# jax-prm

JAX kernel of Persistent Recurrent Memory between decoder-only transformer layers (arXiv:2609.17251).

## Install

```bash
pip install -e ".[tests]"
```

Requires `jax`, `jaxlib`, and `numpy`. Tests need `pytest`.

## Tests

```bash
pytest -q
```

CPU is enough.

## What this is

A decoder-only transformer with a Persistent Recurrent Memory (PRM) module inserted after the lower half of the stack (`n_layers // 2`). Memory is a bank of slots `M ∈ R^{B × n_slots × d_mem}` initialized at zero. At each token t the slots cross-attend to `H_{<=t}`, a GRU updates each slot, slots are mean-pooled, and the result is written back with gated addition `H_t ← H_t + σ(W_g m) ⊙ (W_p m)`.

## What does not match the paper

No TinyStories, no 50k-step training, no 5-seed eval table, and no narrative-position linear probe.

The PDF uses a single persistent vector `s ∈ R^d` with `d_model = 192`; this kernel uses a slot bank (`n_slots=16`, `d_mem=128`) and paper-scale defaults `d_model=256`, `n_heads=8`, `d_ff=1024`, `vocab_size=8000`, `context=256` as specified for the implementation.

Gated addition follows `H + σ(W_g m) ⊙ (W_p m)` rather than the PDF's `g = σ(W_g [H; s'])`. The GRU is unrolled across tokens (one step per position, causal observe) rather than applied `L=2` times to a single full-sequence observation. There is no layer-norm after the GRU step.

Auxiliary loss internals are underspecified in the PDF. If `aux_lambda > 0`, this package adds `MSE(stop_gradient(mean H_lower), W_aux mean M)` to the next-token LM loss. Default `λ = 0`, which the paper's ablation found matched `λ = 0.1`.

Blocks are pre-LN with GELU FFN and learned (not rotary) position embeddings of length `context`. Input and output embeddings are untied.
