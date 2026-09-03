# jax-geosprint

JAX implementation of "GeoSPRINT: Geometric Redundancy-Aware Step Pruning for
Inference in Diffusion Trajectories" (arXiv:2609.02160).

## install

uv pip install -e .

## run

python -m geosprint.demo --nfe 20

Everything here runs on a 2D two-moons diffusion model that trains in a couple of
seconds on CPU. The CIFAR-10, LSUN Church and Stable Diffusion v1.5 FID numbers are
not reproduced, and no pretrained model is downloaded. At toy scale the schedule
beats uniform DDIM at 20 and 30 NFE and loses at 10 and 40, so the paper's
consistent win does not appear; the end-to-end tests assert what does hold and say
which numbers they measured. Three things the paper leaves open, and what I picked:
rho_logSNR is |dlambda/dt| for the variance-preserving log-SNR, since the formula is
never written out; the K quantiles are linspace(0, 1, K), so both ends of the range
are always in the schedule; and both densities are normalized before Eq. 7 blends
them, which is what makes beta a mixing weight. Equation and algorithm line numbers
are in the docstrings.

The suite has been run on CPU and on an H200 (jax 0.11.1, CUDA). One test used to
claim more than the toy model supports: that GeoSPRINT's blend beats pure curvature
spacing at every NFE budget, checked on the one seed the fixtures train with. That's
a single-draw claim and it doesn't hold up -- retraining the model, reference pass
and target at 45 independent seeds (25 CPU, 20 H200) put the blend ahead at every
budget in 4 of 45 draws, and CPU and GPU agreed on the distribution to within noise.
So the fixed-seed test that happened to land on a lucky draw passed reliably on CPU
and failed on roughly half of GPU runs, for the same reason on both: not a backend
bug, a test that measured one coin flip and called it a property of the method. It
now asserts what actually holds across seeds; see the docstring on
test_blend_beats_pure_curvature_spacing in tests/test_toy_end_to_end.py for the
numbers.
