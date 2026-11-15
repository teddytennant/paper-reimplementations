JAX kernel of Q-DEQ local QUBO DEQ solving plus W8A8 fake quant (arXiv:2609.24042).

## Install

```
python3 -m venv .venv && .venv/bin/pip install -e ".[tests]"
```

## Run

```
.venv/bin/python -m pytest -q
.venv/bin/python train.py
```

## What does not match the paper

no iTransformer, no five public MTS datasets, no Kaiwu CIM, no reported MSE tables, 8-bit encoding range [-1,1] is our choice because the paper omits offset/B, tiny MLP not the paper's backbone.
