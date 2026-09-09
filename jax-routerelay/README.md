# jax-routerelay

JAX implementation of "RouteRelay: Event-Triggered Cross-Layer Route Reuse for Efficient Dynamic Sparse Attention" (arXiv:2609.07306).

## install

uv pip install -e .

## run

python -m pytest -q

No fused GPU kernel and no DHSA backend. The CPU toy drift is not the paper's 384x192x32-layer table.
