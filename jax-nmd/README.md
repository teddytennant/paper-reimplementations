# jax-nmd

JAX kernel of Neural Modal Decomposition from observables (arXiv:2609.14402).

## Install

```bash
pip install -e ".[tests]"
```

Requires `jax`, `jaxlib`, `numpy`. Tests need `pytest`.

## Tests

```bash
pytest -q
python scripts/smoke.py
```

CPU is enough.

## What this is

The kernel synthesizes a multi-port impedance as a pole-residue sum
`Zij(ω) = sum_k l_{k,i} r_{k,j} / (jω - p_k)`, with poles constrained to the
open left half-plane by `p_k = -softplus(u_k) + j v_k`. A tiny cavity encoder
maps an 8x8 binary geometry to tokens. PoleNet maps those tokens to K poles.
Two independent AmpNet heads are queried once per port and return complex
couplings as `softplus(mag) * exp(j phase)`. N-port Z is assembled from N
per-port queries, then converted to S by `S = (Z - Z0 I)(Z + Z0 I)^{-1}`
with Z0 = 50. Training uses only observables: per-entry MAE on real and
imaginary parts of S, a log-magnitude Z term, and an anchor-gauge penalty.
Geometry-to-response pairs are synthetic LTI systems with known poles and
couplings, not measured cavities.

L_S is the mean absolute error over real and imaginary parts of every
S-matrix entry (and frequency). L_{Z,log} is
`mean |log(|Zpred|+ε) - log(|Zgt|+ε)|` with ε = 1e-2. The gauge term uses a
boolean `is_anchor` mask per port rather than a hardcoded cavity row index.

## What does not match the paper

- No 18x18 cavity CNN, no 14.5M-parameter model, and no EMX dataset.
- Default K=4 and token dim 32, not K=32 and Dt=384.
- AmpNet/PoleNet are one-block single-head attention, not multi-layer
  multi-head transformers with hybrid row embeddings and sinusoidal column PE.
- Anchor ports are marked by an `is_anchor` boolean per port. The paper
  uses r_anchor = 17 on 18-row cavities; that row index is not hardcoded here.
- No AAA comparison, no zero-shot MAE numbers, no 291k RF samples.
- Geometry is a random binary grid (or any 8x8 array) and is not inverted
  from a real electromagnetic field.
