Stiefel-AdamW ([arXiv:2609.21039](https://arxiv.org/abs/2609.21039)): AdamW on linear factorization blocks W=BA with A constrained to the Stiefel manifold via tangent projection and a Cayley retraction.

```
pip install -e . && pytest
```

## API

`StiefelAdamWState` holds factors `(B, A)` and Adam moments. `step(state, g_b, g_a, ...)` takes Euclidean factor gradients `∇_B L(BA)`, `∇_A L(BA)`. `step_from_dW(state, dW, ...)` takes `dL/dW` and converts with `G^B = (dL/dW) A^T`, `G^A = B^T (dL/dW)`.

Geometry lives in `stiefel_adamw.geometry`: `project` / `project_a1` (eq. A.1), `qr_retract`, `cayley_retract` (eqs. A.2–A.3).

A CPU/GPU smoke factorization is `python scripts/smoke.py` (prints a JSON line with final loss and `max |A A^T - I|`).

## What does not match the paper

- no GPT2/ViT/Mistral, no OpenWebText, no LoRA fine-tuning
- default retraction is Cayley via the fixed-point iteration of Appendix A.2 (α=1 because ξ already includes −η); QR is optional
- bias correction is on by default (Algorithm 1 omits it); both on/off are tested
- decoupled weight decay is implemented like AdamW (omitted in the Algorithm 1 exposition). On A it is added to the ambient direction and then killed by `P_X`, since `X` is normal to St(n, r)
- AMSGrad (`max` on second moments) is not used; the paper notes it is not needed in practice
- no decreasing `η_t = η/√t` from the theorem; constant step size
- Cayley uses a fixed number of iterations (default 8), not an adaptive residual
- tests and smoke are synthetic `||BA − W*||_F^2` factorizations, not language-model blocks
