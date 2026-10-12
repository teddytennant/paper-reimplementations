implementation in JAX of scan-parallel variational inference for latent jump models (arXiv:2610.12986).

```bash
pip install -e .
python -m pytest
```

The recognition net is a small causal MLP, not the paper's pre-norm transformer; no REFIT or USHCN. An 8-step CPU AdamW run on one synthetic batch dropped the loss from 304.09 to 235.97. The H200 job was cancelled while still queued (start time 15:01 UTC), so there is no GPU result.
