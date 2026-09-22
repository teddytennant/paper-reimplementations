Unconstrained p=2 solver for the optimal high-order monotone VI method (arXiv:2609.23557). X = R^d, residual ||F(x)||.

```
pip install -e '.[tests]'
PYTHONPATH=src python -m pytest
PYTHONPATH=src python -m optimal_mvi.demo
```

## What does not match the paper

- p=2 only. The outer schedule (equation 8) is coded for general integer p; the tensor step and the shipped solver are p=2.
- X = R^d only. The normal cone is {0} and the residual is ||F(x)||. Constrained sets and a general cubic solver are not implemented.
- a_p, m_p, and theta_p are existential in the paper. Concrete choices: m_p = 1.0, a_p = 0.5, theta_p = 0.5. They set the initial S and rho; the acceptance test (1/mu) ||F(u)|| <= delta is the real gate, not the existential S. With these constants, S_j = S0 exactly as in Algorithm 3 already meets the test, so the S_j = S0*(j+1) growth deviation was not needed.
- No minimax benchmark table.
- Phase I decreases the extra anchor, mu_k = mu*(1+m_p)^{R-1-k}, and steps on z |-> F(z) + (mu_k - mu)(z - x) (Algorithm 2). Phase II steps on the target operator F. An increasing schedule mu*(1+m_p)^k with Phase II at mu_R does not drive the acceptance residual down. R and S are iteration counts (the paper's R is a distance). Each trial is capped at R_j + S_j <= 40.
- On the cubic example, Algorithm 1 at T=6 and T=8 leaves ||v|| about 0.091 and 0.053 even with exact resolvents (Theorem 4.1's constant is not small). The demo uses T=20, where ||v|| < 1e-2. The threshold is not weakened.
