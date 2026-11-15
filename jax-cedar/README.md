# jax-cedar

JAX implementation of "CEDAR: Error-Bounded Residual Routing for Efficient Long-Context Attention" (arXiv:2609.07237).

## install

uv pip install -e .

## run

python -m pytest -q

Reference code for hybrid exact-summary attention, error-aware chunk refinement, and the reconstruction bound. There is no fused online-softmax kernel, no DHSA chunker, and no 128K speedup claim. Tests cover identities, the output-error bound, and a small clustered reconstruction study only.
