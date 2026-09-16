# jax-gralod

Plug-and-play continuous level-of-detail (LOD) for image restoration, in JAX (arXiv:2609.16578).

## Install

```bash
pip install -e ".[tests]"
```

Requires `jax`, `jaxlib`, `numpy`. Tests need `pytest`.

## Tests and smoke

```bash
pytest -q
python scripts/smoke.py
```

CPU is enough for `pytest`. The smoke script runs a 16x16 dummy restorer for 20 Adam steps and prints `train_loss_0` and `train_loss_20`.

## What this is

GraLoD reuses a backbone encoder pyramid `{F^l}`, aligns it into a shared LOD space `{Z^l}` with lightweight `A_l`, predicts a stage-conditioned field `lambda_s` in `[0, L-1]`, interpolates two neighboring levels, and residual-injects `F_LOD` into the decoder. Training adds MSFC calibration and structure-aware regularization. Inference keeps only `A_l`, `R_{s,l}`, `G_s`, the query, and `H_s`.

All tensors are channel-last NHWC. Parameters are explicit pytrees (`init(key, cfg) -> params`); there is no Flax module tree.

## What does not match the paper

- No FSNet, ALGNet, or StarIR backbones. A tiny conv encoder/decoder exists only for tests and smoke.
- No Rain100 / GoPro / SIDD / etc. datasets and no ImageNet-scale training.
- Internals of `G_s`, `H_s`, and `P_s` are a reasonable lightweight conv (1x1 project, concat, Conv3x3, SimpleGate, Conv1x1). The paper underspecifies these blocks.
- The aligner `A_l` follows the prose order Conv1x1 -> DWConv3x3 -> SimpleGate -> SCA -> Conv1x1 -> LayerNorm. The composition formula is written `A_l = LN o Conv1x1 o DWConv3x3 o SG o SCA o Conv1x1`, which disagrees with that prose; this package follows the prose.
- Reconstruction is L1 (the paper says "follows the original backbone").
- Probes and MSFC errors both take stop-gradient inputs `sg(Concat(D_s, Z_s^l))`, so the auxiliary heads cannot hijack the backbone. Probes are discarded at inference.
- Anti-aliased downsampling uses a separable binomial `[1,2,1]/4` blur, then stride subsample (exact 2x) or bilinear resize. Not a learned low-pass.
