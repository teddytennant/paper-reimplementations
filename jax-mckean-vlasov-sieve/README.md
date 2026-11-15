# jax-mckean-vlasov-sieve

JAX implementation of the ReQU sieve maximum likelihood estimator from
"Nonparametric inference for density-dependent McKean-Vlasov diffusions"
(arXiv:2609.01166).

## install

uv pip install -e .

## run

python -m mkv_sieve.fit --n 2000

Only the OU potential (1.2) is implemented. The sieve keeps the structural clamp
kappa <= xi <= K of eq (4.1) but not the sparsity budget or the Holder ball of
eq (4.2), which exist for the entropy bounds of Section 5 and not for the fit.
Nothing asymptotic is checked: the rates of Sections 5 and 6 and the Assouad
lower bound of Section 7 need sample sizes this does not run at, so the tests
assert the exact structural facts instead. Data is drawn from the stationary law
directly, so Section 6's transfer to finite-time observations of (1.1) is never
exercised. The profile ODE and the normalisation integral sit on fixed grids,
and Adam is not guaranteed to reach the global minimiser of (3.9).
