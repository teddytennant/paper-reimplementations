# cyclic_pivot_gray

Cyclic pivot Gray codes for well-ordered k-degenerate graphs, arXiv [2610.09410](https://arxiv.org/abs/2610.09410).

```
cargo test
```

Algorithms 1-6: pruned BRGC listing, ranking, and unranking for binary strings of weight at most k, then the same three for well-ordered k-degenerate graphs. Ranks are 1-based, as in the paper. The listing is checked against Example 8 (n=4, k=2), string rank 5 for 0110 (Example 11), and graph rank 7 for the n=3, k=2 graph with edges (1,2) and (1,3) (Example 14).
