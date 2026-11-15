Counting and Filtering vocabulary construction and Min-Cost Encoding (arXiv:2610.01127).

```
cargo test
```

Preference cost is section 4.1's formula, scaled by eta (1 is the paper, 0 drops it).

Does not match the paper: sequential counting instead of parallel workers (the counts are the same); LM training at 1.8B/8B is not included; the top-m cut is applied after counting, and all 256 single-byte tokens are force-included so MCE's assumption holds.
