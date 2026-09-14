JAX library for flow duality and source geometry for categorical generation (https://arxiv.org/abs/2609.10863).

```
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[tests]"
JAX_PLATFORMS=cpu python -m pytest -q
JAX_PLATFORMS=cpu python scripts/smoke.py
```

What does not match the paper: no language-modeling pilot, no OpenWebText, no GPT-2 vocabulary figures, no learned-transport plots.
