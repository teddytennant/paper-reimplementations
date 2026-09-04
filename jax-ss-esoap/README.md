# jax-ss-esoap

JAX implementation of "SS-ESOAP: Self-Scaled Adaptive Preconditioning for
Physics-Informed Learning" (arXiv:2608.29448).

## install

uv pip install -e .

## run

python -m ss_esoap.train

Algorithm 2 is here in full with the Table 6 defaults, next to a SOAP baseline
that follows Section 2.4 rather than the reference SOAP implementation. Two
constants in the Algorithm 2 signature, tau_min and Twarm, never get a value in
the paper; the defaults here are 0.1 and 0. tau_min does real work, because tau
scales as the fifth power of the gradient magnitude and collapses to the floor
once training gets accurate. SS-ESOAP beats Adam by roughly 45x on the shrunk
Section 5.2 quadratic in the tests, but it loses to Adam on the Poisson demo,
and most of the quadratic win comes from the adaptive basis rather than the
self-scaling. Not reproduced: the eight PDE benchmarks, PirateNet, the 4.1-hour
Boussinesq run, and every baseline other than SOAP.
