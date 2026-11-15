Direct Spectral Laplacian Estimation (DSLE) estimates a graph Laplacian from node observations by regularizing effective resistance, as described in arXiv:2609.23668.

Install and run:

```
pip install -e '.[tests]'
PYTHONPATH=src python -m pytest
PYTHONPATH=src python -m dsle.demo
```

What does not match the paper

There is no GLASSO or iterative MLE baseline, no 50-node timing table, and no real sensor dataset. The synthetic check is a unit-weight path on 12 nodes, plus an underdetermined draw of 12 Gaussian samples on a 24-node path.
