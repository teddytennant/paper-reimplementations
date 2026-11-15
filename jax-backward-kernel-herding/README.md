# jax-backward-kernel-herding

JAX implementation of "Revisiting Thinning Methods for Kernel Learning Problems" (arXiv:2609.07432).

## install

uv pip install -e .

## run

python -m bkh.demo --n 64 --keep 32

Toy kernels and synthetic points; not the paper's GP/KSVM benchmark tables.
