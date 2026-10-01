# semi-ladder-cds

Rust implementation of the compact representation of minimal set covers and the connected-dominating-set algorithm for d-semi-ladder-free graphs (arXiv:2609.39666).

## Install

cargo build

## Run

cargo test

## What does not match the paper

The group Steiner tree routine is exponential in the number of groups, not the polynomial-space algorithm cited as Proposition 7. Theorem 4's (1+ε) lossy compression is not implemented. Solve grows one seed by the intersection of every set containing the seed plus the new element. There is no attempt at the k^{kd+2} nm enumeration bound as a measured runtime.
