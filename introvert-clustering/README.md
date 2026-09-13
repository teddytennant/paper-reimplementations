Rust implementation of introvert clustering and layered introvert network decomposition (https://arxiv.org/abs/2609.10044).

```
cargo test
cargo run -- 8
```

This is the randomized MPX + trim construction (Lemma 1, Lemma 2, Theorem 3). It does not implement the deterministic Ghaffari-Grunau construction (Theorem 4 / Appendix A), LOCAL round counts, list (3/2+eps) Delta-edge coloring, or (1/4-eps)-locally balanced cut.
