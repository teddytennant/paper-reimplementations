Small JAX core of arXiv 2609.36950: diffusion coefficient, exact compositional Gaussian score, hierarchical blockwise sampling, and path-regularized fine-tuning.

Not implemented: the BMP cell-line model, the 940 measurements, and the SLCP benchmark numbers.

Install:

    uv venv /tmp/hbds-venv && uv pip install -e . --python /tmp/hbds-venv/bin/python

Run:

    /tmp/hbds-venv/bin/python -m hbds.finetune --steps 40
    /tmp/hbds-venv/bin/pytest -q
An H200 smoke was not submitted. The queue start for this session's jobs was 2026-10-03 23:17Z, outside the session. `python -m hbds.finetune --steps 40` did run on CPU: loss 0.887865 to 0.278935, 369 parameters.
