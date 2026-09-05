# jax-hybrid-pbe

JAX implementation of "Differentiable Hybrid Modelling for Learning and
Optimising Chemical Transport Processes from Experimental Data"
(arXiv:2609.04011).

## install

uv pip install -e .

## run

python -m hybrid_pbe.train --steps 500

The 13 experimental trajectories of Bötschi et al. are not public and are not
here, so the ground truth is synthetic: the solver generates trajectories with
the published M1 law and the hybrid model has to recover it. Fluxes are plain
first order upwind, not the paper's high resolution scheme, so the same grid is
more diffusive, and the default grid is 64 x 32 against their 360 x 120. Two
equations needed a reading. Eq (7) prints S = (c - c*)/c*, but every other
equation and the Smin/Smax bounds need S = c/c*. Eqs (13) and (14) only give
physical growth rates with T in Kelvin, though the paper lists T in Celsius.
Both are noted in kinetics.py.
