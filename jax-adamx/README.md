JAX implementation of AdamX, cosine-controlled AMSGrad (https://arxiv.org/abs/2609.11867).

```
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[tests]"
JAX_PLATFORMS=cpu python -m pytest -q
JAX_PLATFORMS=cpu python scripts/smoke.py
```

What does not match the paper: no MNIST/CIFAR-10 CifarNet runs, no comparison table against Yogi/Lion/Adan, Algorithm 2 OCO projection omitted, toy MLP only.
