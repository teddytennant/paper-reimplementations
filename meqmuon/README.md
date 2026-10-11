# MeqMuon

Matrix-equilibrating Muon: orthogonalize hidden-layer updates, then normalize the more imbalanced side.

## Install

From this directory:

```
pip install -e .
```

## Tests

```
pytest
```

## What does not match the paper

No LLM pretraining and no C4 numbers. Table 4 is not reproduced.

Algorithm 1 orthogonalizes the momentum buffer Bt. Section 4 says the reported runs used Nesterov momentum. Default follows Algorithm 1. nesterov=True orthogonalizes G + mu*B after the buffer update, and only on hidden weights.

Rectangular orthogonalization transposes so the Gram matrix is the smaller side, matching Muon, which the paper says Orth follows. The displayed recurrence writes X @ X.T without that transpose.

Rows and columns with RMS <= 1e-7 are left unscaled, and an all-zero matrix is returned as zeros. The paper excludes those entries from the coefficient of variation but does not write the zero-row inverse.

A 1e-7 epsilon is added to the Frobenius norm. The paper writes a bare division.

No learning-rate schedule. The caller passes eta. The paper's runs use 5% linear warmup then cosine decay.

This is float32/float64 JAX, not bfloat16 CUDA Newton-Schulz.
