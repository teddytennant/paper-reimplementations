# bsca

Biquaternionic temporal knowledge graph embeddings with complex-valued attention (https://arxiv.org/abs/2609.14279).

## Install

```bash
pip install -e ".[tests]"
```

## Tests

```bash
python -m pytest -q
```

## What does not match the paper

No ICEWS/GDELT/YAGO runs; embedding dim in tests is 4 not 2000; complex attention uses Re⟨a,e⟩ before softmax because softmax is not defined on C; N3 is applied to complex component moduli; no 3090 150-epoch grid.
