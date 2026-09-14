JAX implementation of token-logit to byte-logit conversion (Marginalize-It and End-Of-Token) from Breaking the Token Ceiling (https://arxiv.org/abs/2609.12303).

```
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[tests]"
JAX_PLATFORMS=cpu python -m pytest -q
JAX_PLATFORMS=cpu python scripts/smoke.py
```

What does not match the paper: no 1B models, no 1T-byte sweep, no Llama/Gemma comparisons, no logit dumping.
