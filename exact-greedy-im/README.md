Exact Independent Cascade inference and exact greedy influence maximization on graphs of treewidth at most 2, following arXiv:2609.19960.

## Install

```
cargo test
```

## Run

```
cargo test
```

The library has no CLI. Call `influence`, `influence_and_gradient`, or `exact_greedy` on a `Graph` plus a `TreeDecomposition`. Helpers build decompositions for paths, trees (width 1), and 2-trees / series-parallel bags of size 3.

## What does not match the paper

Only width ≤ 2 is implemented. There are no k-tree n=10k experiments and no IMM / Monte Carlo runtime tables. Tree decompositions are supplied by the caller or built for trees and 2-trees; Bodlaender's linear-time algorithm is not included.
