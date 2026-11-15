RoSA, rotational sparse adaptation for fine-tuning, from https://arxiv.org/abs/2610.06243.

```bash
JAX_PLATFORMS=cpu python -m pytest -q
```

Tests check Algorithm 1's rotating schedule (including a short last group and the epoch wrap), the bottom-rho magnitude mask, cold-start moments, and truncated backward on a toy stack. They do not fine-tune Llama, do not match the paper's commonsense numbers, and do not cache prefix activations. Training always runs the full forward pass: the paper's cached-prefix optimization is not part of this implementation. Optimizer-state accounting follows the scaling identity (l/L) Psi rho K_opt and is not a measured VRAM number.
