Schedule optimization for tau-leaping in masked discrete diffusion ([arXiv:2609.21960](https://arxiv.org/abs/2609.21960)): dependence density ρ, factorization-error integral, finite-K stationarity shooting, and the limiting smooth schedule.

```
pip install -e . && pytest
```

## What does not match the paper

- no trained masked diffusion model
- no §6 ρ estimators, no large-N asymptotics, no ImageNet, no exchangeable-mixture experiments
- ι is planted; combinatorial CMI on a real process is not implemented
- last geometric knot is forced to 1 so the sampler always finishes
