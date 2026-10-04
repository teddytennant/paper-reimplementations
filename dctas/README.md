DCTAS implements the fixed-confidence cost-aware track-and-stop algorithm for Condorcet dueling bandits from Sarvesh Gharat, Nikhil Karamchandani, and Jayakrishnan Nair (arXiv 2609.30360).

```
pip install -e .
pytest
python demo.py
```

This package does not include the LLM preference matrices from the paper and does not include the optimality proof. If the empirical preference matrix has no Condorcet winner, the plug-in winner is the arm maximizing min over j not equal to i of p_hat[i, j], with ties broken by the smallest arm index. A suboptimal arm with alpha_m = 0 is assigned alpha_m = 1e-12 before the reciprocal is taken. Time is indexed by t_total, the number of pulls so far including initialization: the exploration guard compares N_k with sqrt(t_total) before the pull, one allocation vector is stored for each pull after initialization and is included in that pull's cumulative target, and the stopping check uses post-pull counts in Z and post-pull t_total in beta.
