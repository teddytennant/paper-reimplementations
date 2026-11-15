Certificate-governed CRT sparse FFT from arXiv:2609.34086.

## Install

From this directory:

    cargo build --release

## Run

    cargo test

The public entry point is `sparse_fft`. Stage primes for an engineered length are chosen by `engineered_primes`. Arithmetic is `f64` complex.

## What does not match the paper

Arithmetic is IEEE-754 f64 complex, not the paper's exact real-RAM model. The occupied equal-magnitude pre-screen is the nMAD surrogate of Definition VII.1 with implementer-chosen tau_cv = 1e-8; the paper does not calibrate this threshold. There is no mixed-radix 64 p_ℓ batching of the per-stage transform; each stage is a plain p_ℓ-point DFT. No root-index-oracle cost-model claims are made. Label decoding uses arg, round, and root-membership verification rather than a comparison-model binary-search decode. Dense fallback is a standard unnormalized DFT, not a tuned mixed-radix FFT of a 64 p_ℓ design-stage convention.
