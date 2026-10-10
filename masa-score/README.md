# masa

Matrix Approximation Sparse Attention scores: pair, block, vertical, and slash, in JAX.

## Install

From this directory:

```
pip install -e .
```

`pyproject.toml` sets `pythonpath = ["src"]` for pytest, so an editable install is optional.

## Tests

```
JAX_PLATFORMS=cpu python -m pytest tests -q
```

## Not the paper

This package only implements the score identities used to rank sparse units. It does not train or run an LLM, does not call FlashAttention, and does not evaluate LongBench or RULER.

Soft row balancing divides each row of scores by the mean absolute value of that row plus a small epsilon so rows are comparable. Clipping caps scores at a multiple of the median positive score before ranking. The default multiple is 10. Both are practical stabilizers. The multiple is a practical choice, not a claimed theorem constant. They are not transcribed as paper equations.
