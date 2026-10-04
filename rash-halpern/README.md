This package implements Algorithm 1 of arXiv 2609.28792 (Wang and Atia): the approximately shifted robust Halpern iteration for finite robust average-reward MDPs. Phase I estimates the vector gain by iterating an exact robust Bellman oracle from the origin, and Phase II applies the printed anchored update shifted by that estimate. The three Appendix B models use analytic row minimizers, and the same iteration can be run with the nominal operator T0.

Install and run from this directory:

    pip install -e .
    pytest
    python train.py

What this does not match: this code does not implement the gain-bias certificate theory or the history-dependent opponent results, and it does not reproduce the N=16384 table. It does implement Algorithm 1 and the three Appendix B models with analytic row minimizers. Ties favor c in Models I and II and favor risky in Model III. On Model I the printed update selects t at N=1 and N=2 and c at N=3, N=4, and N=32.
