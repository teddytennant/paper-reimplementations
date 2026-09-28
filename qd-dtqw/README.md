JAX kernel of hybrid quantum diffusion with a classically simulated DTQW forward process (arXiv:2609.31070).

## Install

```
python3 -m venv .venv
.venv/bin/pip install -e ".[tests]"
```

## Run

```
.venv/bin/python -m pytest
.venv/bin/python train.py
```

## What does not match the paper

No IBM/Qiskit hardware (`ibm_torino`), no BloodMNIST/BraTS/FractureMNIST3D, no 2D or 3D U-Net, no FID or SSIM. NISQ noise is a depolarizing (and amplitude-damping) channel instead of a real-device noise fingerprint. The reverse model is a two-layer time-conditioned conv net on synthetic 4x4 discrete images. Tests use Nq=2 (K=4) rather than 5–6 position qubits.
