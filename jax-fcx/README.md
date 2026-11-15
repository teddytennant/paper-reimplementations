JAX implementation of FCx (Feasible Counterfactual Explanations) from arXiv:2609.19383.

## Install

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

## Tests

```bash
pytest -q
```

CPU is enough.

## What does not match the paper

No PC-algorithm, no Adult/Law School/KDD/Folktables tables, no REVISE/C-CHVAE/CEM/FACE comparison. LOF is batch-kNN in latent space not sklearn LOF on decoded CFs. The CVAE is a five-layer ReLU MLP without batch-norm, trained on synthetic [0, 1] tabular data rather than the paper's census benchmarks.
