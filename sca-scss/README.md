Rust implementation of the Strong Connectivity Augmentation reduction to SCSS with two edge costs (https://arxiv.org/abs/2609.11160).

```
cargo test
cargo run
```

What does not match the paper: no Frank-Tardos weight reduction, no O(k^16)-bit kernel, no approximation discussion. DP is exact on n<=12 terminals unweighted and n<=10 weighted.
