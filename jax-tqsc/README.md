JAX kernel of arXiv:2609.20523 TQSC (Transformer-based Quantum State Characterizer): a 13-token encoder that maps a measurement vector to a physical single-qubit density matrix.

```
python -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install jax jaxlib numpy pytest
.venv/bin/pip install -e ".[tests]"
JAX_PLATFORMS=cpu .venv/bin/pytest -q
JAX_PLATFORMS=cpu .venv/bin/python scripts/smoke.py
```

On NixOS, `import jax` may need `LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH`.

This is not the paper's SPDC/MMF optical experiment, MNIST transmission task, or Qiskit Bloch-ball suite. Training uses a synthetic 13-channel HVDR+aux simulator instead of experimental coincidence counts; width and step count may differ from the 152,292-parameter trained model.
