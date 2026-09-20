CPU exact-special-case COMPASS for Ordered Clustered TSP (arXiv:2609.20352).

```
cargo test
```

What does not match the paper. This crate is Algorithm 1's exact special case (candidate endpoint pairs, Held-Karp sub-solver, layered DAG DP) plus the Connect-first baseline. It does not include EARLI/PPO/cuOpt, GPU triplet-optimization candidates, SISR bridge repair, LKH, the São Paulo or 100K-node pipelines, or 16×H100 execution. Intra-cluster subproblems use Held-Karp instead of an RL+GA sub-solver. Geometry candidates use λ = 0.1 from Appendix C.1; source (b) is omitted.
