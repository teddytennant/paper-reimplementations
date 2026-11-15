# jax-matnorm-partial-em

JAX implementation of "An efficient EM algorithm for both element-wise and structural
missingness in matrix-variate normal mixture models" (arXiv:2609.00616): exact EM, the
proposed partial EM, the submatrix specialization, and the mixture extension.

## install

    uv venv && uv pip install -e . && uv pip install pytest

## run

    python -m matnorm_pem.demo

## what is not here

No hyperspectral image application, and no mixture of spatial factor analyzers, so
Section 4.2 and Appendix A are missing. The timing tables are not reproduced: the paper
runs 15x20 and 21x24 matrices with N up to 3000 in R and C, while this is synthetic data
at single-digit dimensions, sized to keep the tests fast. The efficiency claim is
checked by counting factorizations and their orders instead of by wall clock.
