# jax-mfspnet

MFSPNet in JAX: PSO over a variable-length dense-block encoding scored by a
model-free validation-loss EMA surrogate, from arXiv:2609.02460.

## install

```
uv venv && uv pip install -e ".[test]"
```

## run

```
python -m mfspnet --compare-random
python -m pytest -q
```

## what does not match the paper

This is the search algorithm, the estimator and the stacking rule at toy scale
on synthetic 8x8 images. It does not reproduce the reported 3.91% CIFAR-10,
17.68% CIFAR-100, 1.91% SVHN or 28.29% ImageNet top-1 error, which need about
three GPU days of search plus full training; nothing here downloads a dataset.
Appendix A gives its bias and variance formulas but not the trajectories behind
its Monte Carlo numbers, so `signal_noise.paper_scenario` is a reconstruction
picked to land on the four reported values. Appendix E's correlations are
checked on synthetic learning curves, which assumes the paper's premise about
validation against training loss rather than confirming it on trained networks.
Stacking follows Eq. 11, the normalized form, not the unnormalized version in
Alg. 4 line 10.
