JAX implementation of ENCORE, exact non-equilibrium control with replica exchange (arXiv:2610.02538).

Install with `pip install -e .` and run `pytest`.

Cut from the paper: no diffusion SDE, no molecules, no images, no guided-proposal stability analysis. The invariance test is on a 3-state chain. It checks detailed balance of one adjacent-level Metropolis exchange against the product of the trajectory targets at those two levels, which is what Theorem 3.1(a) reduces to for one exchange. It also checks that the product of the three level-wise trajectory targets is a left eigenvector of each parity's one-step kernel.
