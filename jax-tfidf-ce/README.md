JAX implementation of TF-IDF weighted cross-entropy for language models (https://arxiv.org/abs/2609.11029).

```
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[tests]"
JAX_PLATFORMS=cpu python -m pytest -q
JAX_PLATFORMS=cpu python scripts/smoke.py
```

What does not match the paper: no 1.1B-13B decoder-only models, no LoRA, no Pile/WikiText memorization eval, toy logits and synthetic tokens only.
