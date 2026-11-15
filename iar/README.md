One-atom Iterative Atom Refinement (IAR) for dictionary learning, from arXiv:2609.23812.

```
pip install -e '.[tests]'
PYTHONPATH=src python -m pytest
PYTHONPATH=src python -m iar.demo
```

What does not match the paper: no K=300 full-combinatorial figure, no MOD benchmark, no non-orthogonal large-scale table. Tested a fixed 4 by 6 update with L=2; monotonicity and 12 restarts at K=7, s=2, L=3 (M=21); amplification at K=11, s=2, L=5 (M=55). K=7, s=2, L=3 and K=9, s=2, L=4 plateau at sqrt(L/(L+1)) which is below 0.9, so the 0.9 check uses the smallest combinatorial instance where it holds.

Ties at a top-L cutoff are broken by a stable ascending argsort (numpy mergesort), keeping the last L indices.
