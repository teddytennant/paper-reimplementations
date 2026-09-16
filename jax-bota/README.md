JAX kernel of Branched Optimal Transport Amortization (BOTA, arXiv:2609.15072): discrete-time soft-atomic branched OT plus branched flow matching on 2D particles.

## Install

```
pip install -e ".[dev]"
```

## Run

```
JAX_PLATFORMS=cpu python -m pytest -q
JAX_PLATFORMS=cpu python scripts/smoke.py
```

The smoke script is also the GPU job command (drop `JAX_PLATFORMS=cpu` to use a GPU).

## What this kernel implements

Stage A (Sec. 5.1). Particles move by Euler `x_{k+1} = x_k + v_k / T`. The objective is the Riemann sum of Lemma 1's soft-atomic cost `C_soft = sum_k |v_k| m_k^alpha` plus a terminal MSE onto the target samples. Soft mass uses a Gaussian softmax onto a bounding-box proxy grid. Gradient descent on the velocity tensor.

Stage B (Sec. 5.2). Linear interpolant `(x_t, u_t)` along those discrete paths, distilled by

`L_BFM(theta) = E[ ||v_theta(x_t, t) - u_t(x_t)||^2 ]`

into a tiny SiLU MLP. Optional CNF integrand `||v|| p^alpha - lambda log p` from Eq. 9 is available as a functional.

Static 2-source / 2-sink edge costs `sum ell m^alpha` check the paper's alpha invariant without training: at alpha=1 sharing a trunk is not rewarded; at alpha=0.5 the branched graph is strictly cheaper.

## What does not match the paper

Does not train ImageNet or biology models. 2D particle BOT + tiny MLP CFM only; discrete-solver details may not match the paper's unpublished hyperparameters.

Further gaps:

- Proxy centers are an axis-aligned bounding-box grid. The paper never says how `{c_k}` are placed.
- Kernel bandwidth `sigma=0.5`, terminal weight `lambda=5`, `T=4..8`, and SGD/Adam step sizes are chosen here, not published.
- Lemma 1 in the main text averages particle speeds `|v_j|`; the appendix averages velocity vectors. Default is the main-text scalar-speed form used by Sec. 5.1 (`speed_mode="vector"` is the appendix variant).
- The MLP concatenates scalar time and uses a small hidden width, not a 64-d time embedding or 3x256 layers.
- Discrete solver is Euler + SGD, not an unpublished specialized solver.
- Source/target particles are paired by index (after sorting on x). No Sinkhorn matching.
- CNF log-likelihood (Eq. 9) is not used in the two-stage path; Stage B is MSE flow matching as in Sec. 5.2.
- No FFHQ, MNIST, or TedSim experiments.
