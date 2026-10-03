Counting and Filtering vocabulary construction and Min-Cost Encoding.

```
cargo test
```

Does not match the paper: sequential counting instead of parallel workers (the counts are the same); the token-preference schedule eps(t) = eta * id / |V| is not in the paper (eta defaults to 0); LM training at 1.8B/8B is not included; the top-m cut is applied after counting, and all 256 single-byte tokens are force-included so MCE's assumption holds.
