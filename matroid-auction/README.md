Auction algorithm for weighted matroid intersection (arXiv 2610.00920).

```
cargo test
```

Does not match the paper: no streaming implementation. Theorem 1 is Algorithm 1 with internal epsilon eps/5, and approximate() does that rescaling. Caller epsilon above 1/2 is rejected because the writeup assumes the internal parameter is at most 1/10.
