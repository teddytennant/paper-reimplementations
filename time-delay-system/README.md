JAX implementation of a time-delay sequence model with a semantic lag estimator and a diagonal state (arXiv:2610.12864).

```bash
pip install -e .
python -m pytest
```

We follow the algorithms where they disagree with the main text: Algorithm 1 multiplies the decay into the cosine after W_proj (eq 14 is softmax(kappa s - lambda j) with no projection), Algorithm 6 uses gamma tanh(s_hat) times (C x_bar) (eq 19 writes gamma psi(s_hat) C x_bar), and Algorithms 3 and 6 set z = d/2 rather than eq 22's (s_{k-1} + d)/2. This is not the 12-layer 768-d IMDB/WikiText run; the O(1) claim is the parallel cumsum for the diagonal state while the lag estimator remains windowed attention O(T L), and a 40-step CPU Adam smoke on synthetic delayed-copy sequences had loss 1.386295 at step 0 and 0.030365 at step 40.
