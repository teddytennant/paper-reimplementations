PSA-GML (https://arxiv.org/abs/2609.29150) as a JAX kernel: particle-swarm warm start of STAR-RIS amplitudes and phases with a closed-form ZF fitness, then a first-order coordinate-wise LSTM meta-optimizer on unconstrained amplitudes, phases, and the collapsed precoder.

Install: `pip install -e ".[dev]"`

Run: `pytest -q` and `python train.py`

What does not match: i.i.d. Rayleigh channels only (no Saleh-Valenzuela), toy Nt/N/K, fewer PSO particles and iterations, short LSTM unroll and hidden size, no attempt to match Table I (11.06 bits/s/Hz).
