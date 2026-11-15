# qcoc

Per-head QCOC compresses a key-value memory into prototypes and fits prototype values so log-multiplicity attention matches full attention on the observed queries.

```bash
pip install -e .
pytest
```

This package does not train TabPFN and does not use OpenML. It uses vanilla scaled dot-product attention plus a log-multiplicity bias, not TabPFN scalable-softmax. `eps_num=1e-12` is a numerical floor, not a published constant.
