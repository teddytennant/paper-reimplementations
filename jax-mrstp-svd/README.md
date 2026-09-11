JAX implementation of multi-term randomized semi-tensor-product T-SVD (https://arxiv.org/abs/2609.11168).

This package implements the *algebra* of Xiao, Zhang, Qin, and Wang (arXiv:2609.11168v1): the tensor semi-tensor product under an invertible linear transform, multi-term STP-SVD (MSTP-SVD), and the randomized + power-iteration variant (MRSTP-SVD). It does **not** reproduce the paper's image/video compression or completion experiments.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[tests]"
JAX_PLATFORMS=cpu python -m pytest -q
```

On this machine the same flow is `uv venv && uv pip install -e ".[tests]"`.

Optional reconstruction demo on a synthetic third-order tensor:

```bash
JAX_PLATFORMS=cpu python scripts/smoke.py
```

## What is implemented

| Paper | Code |
| --- | --- |
| Def. 2.2–2.3, Lemma 2.2, Remark 2.1 | `matrix_stp`, `t_product` |
| Eq. (1), Def. 2.4–2.7 | `Transform` (`fft` / `dft` / `identity`), `tensor_transpose`, `identity_tensor` |
| Def. 3.1, Lemma 3.1, Remark 3.1, Thm. 3.1 | `tensor_stp` |
| Eq. (2), Lemmas 4.1–4.2 | `rearrange`, `kronecker_terms` |
| Algs. 1–2, Thms. 4.1–4.2 | `stp_svd_matrix` |
| Alg. 3, Thm. 4.3 eq. (7)–(8) | `mstp_svd`, `mstp_reconstruct` |
| Alg. 5, Thm. 5.1 | `mrstp_svd` (uniform truncation rank via `rank=`) |
| Algs. 6–9 (truncated variants) | `rank=` on `mstp_svd` / `mrstp_svd` |

Default transform is unnormalized FFT along tubes (the paper's experimental default). Identity and an explicit DFT matrix are included so tests can check that STP reduces to the t-product and that FFT matches DFT.

Tensors are stored as JAX arrays with shape `(n1, n2, n3)` and **frontal slices** `A[:, :, k]`. Block partitions `(m1, m2, n1, n2)` mean the spatial unfolding is `m1 m2 × n1 n2`, matching the paper.

## Gaps versus the paper

- No RGB-image / video compression, no tensor completion (eqs. (11)–(14)), no PSNR/SSIM tables.
- Truncation rank is a single integer `rank` applied to every term and slice, not a full matrix `R ∈ N^{k×l}_+`.
- DCT and random orthogonal transforms from Section 6 are not implemented.
- Algorithm 5 prints `G ∈ R^{n1 n2 × (k+s) × l}`; Theorem 5.1 (and the multiply `R(Ā^{(j)}) Ḡ^{(j)}`) require `G ∈ R^{m2 n2 × (k+s) × l}`. The code follows the theorem.

## Suggested GitHub slug

`jax-mrstp-svd`
