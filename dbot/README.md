JAX kernel of D-BOT (arXiv:2609.37239): bisimulation distance as a bicausal optimal-transport LP, with the envelope gradient in the model logits.

PYTHONPATH=src python -m pytest
PYTHONPATH=src python train.py

Does not reproduce the N=16 random-walk curves or the imitation-learning runs. Those use SOMCOT on sample transitions and a policy-gradient inner loop. This solves the LP exactly on enumerated chains with at most four states, and checks that the envelope gradient matches finite differences and that fitting a softmax kernel reduces the distance.

The inner solver in the paper is SOMCOT. Here that step is an exact linear program: scipy.optimize.linprog on the enumerated occupancy constraints, not a stochastic mirror-descent iteration. The parametric kernel is the row-wise softmax of a logit matrix, Eq. 16. Imitation learning from observations (Algorithm 3) is not implemented.
