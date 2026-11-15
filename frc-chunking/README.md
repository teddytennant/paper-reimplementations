Rust implementation of Fixed-Range Chunking and Dual-Boundary Matching chunk alignment (https://arxiv.org/abs/2609.12674).

```
cargo test
cargo run -- 3 4 5 12 4 --m 5 --M 10
```

Relative positions use `rp(i) = i / (n-1)` for `n > 1` (and `0` when `n == 1`), so the first unit is 0 and the last is 1.

What does not match the paper: no 7B fine-tune, no IWSLT2017/GlobVDoc, no LLM Sim, no four training formats as trainers, default Sim is overlap/injected scores. Dual-boundary scores add the next-chunk start term to each top-k candidate end rather than summing that term over all top-k candidates onto every column (the latter is a no-op for argmax).
