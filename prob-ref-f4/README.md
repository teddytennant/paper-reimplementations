Dense GF(p) probabilistic row echelon form of Demin, arXiv:2609.14433.

```
cargo test
cargo run
```

What does not match the paper: no F4, no Macaulay matrices, no msolve sparse linear algebra. Dense GF(p) Algorithm 1 (Monagan-Pearce-Steel) plus rowspace certification only. The paper's Algorithm 3 certifies a Groebner basis inside F4; this crate certifies a proposed REF by reducing the original rows and falls back to deterministic elimination.
