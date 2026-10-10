# fitr

Trust-region Bayesian optimization whose side lengths come from the diagonal pullback Fisher of a Gaussian predictive posterior, in JAX on CPU.

```
pip install -e .
```

```
JAX_PLATFORMS=cpu python -m pytest
```

This package does not run the paper's BoTorch, LogEI, IBNN, or FullFITR experiments. Analytic EI has the same argmax as LogEI. Restarts recenter on the best observation; the paper's REI center selection is not implemented. The Monte Carlo diagonal estimator defaults to a small S, not 256.
