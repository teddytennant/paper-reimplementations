JAX implementation of Suan preference optimization (https://arxiv.org/abs/2609.08634).

```
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[tests]"
JAX_PLATFORMS=cpu python -m pytest -q
JAX_PLATFORMS=cpu python scripts/smoke.py
```

What does not match the paper: no LLM, no PKU-Safe-RLHF, no SFT, no over-refusal eval, toy categorical policy only.
