Commands:

    cd /home/nixos/arxiv-impl-work/2609.34086
    cargo test

Results: 10 passed, 0 failed (2 lib unit tests + 8 tests in tests/exact.rs). Doc-tests: 0.

Mismatches vs arXiv:2609.34086:

- IEEE-754 f64 complex arithmetic, not exact real-RAM.
- Occupied screen is the nMAD surrogate (Def. VII.1) with implementer-chosen tau_cv = 1e-8; the paper does not calibrate tau_cv.
- Per-stage transform is a plain p_ℓ-point DFT (naive for p ≤ 48), not mixed-radix 64 p_ℓ batching.
- No root-index-oracle cost-model claims.
- Label decode is arg + round + root-membership verify, not comparison-model binary search.
- Dense fallback is a standard unnormalized DFT (naive or mixed-radix Cooley–Tukey), not a tuned mixed-radix FFT.
