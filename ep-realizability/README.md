Rust implementation of Equitable Partition realizability (https://arxiv.org/abs/2609.16762).

```
cargo test
cargo run --bin realize
```

What does not match the paper: Algorithm 2 returns an n×k node-to-block profile T; `phi_eps` collapses that to a K×K cell-constant AH (regular sequences) so `realize` can run, then greedily adjusts toward `n_edges` (the edge budget is not an Algorithm 2 input). Layer-preserving switches are intra-layer 2-swaps; the paper's 50-sample burn-in/thinning ensemble is not implemented. Vertex matching by degree rank in Section 5 is unused: `edge_overlap` identifies vertices by index. Cora, CiteSeer, and PubMed are not included.
