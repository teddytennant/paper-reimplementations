JAX implementation of the VUS U-statistic and the label-based estimator in equation 53, arXiv:2609.38875.

```bash
pip install -e .
pytest
python train.py
```

What does not match: no double-ML nuisance zoo, no group feasible-error set, no Gini coefficient, no claimed coverage rates. Cross-fitting is a 2-fold split, not the K-block scheme in section 3.2. Influence-function standard errors are not implemented.
