JAX kernel of Ability-Residual Decoupled Modeling for Affective Cognitive Diagnosis (ARACD), NCDM backbone plus ability residual and guess/slip affect: https://arxiv.org/abs/2609.21214

```
pip install -e . && pytest
```

## What does not match the paper

- no ASSIST/Junyi (or any public CD dataset); synthetic student/item IDs only
- no six backbones: NCDM only (no DINA/IRT/MIRT/RCD/SCD)
- no author GitHub (psychosiwa/ARACD) wrap or port
- no leakage probes, PCA, or long-tail analyses
- NCDM MLP is Linear→sigmoid→Linear→scalar then outer σ, not the original three-layer monotonic NCDM
- LayerNorm on the concept-attention query is last-axis population LN without affine scale/bias
- low-rank dimension defaults to 16, not the paper’s 1024
- no unlabeled contrastive affect recipe; optional MSE only if affect labels are passed
- w_SI and w_C are learnable scalars initialized at 1; γ_SI is a fixed hyperparameter defaulting to 1.5
