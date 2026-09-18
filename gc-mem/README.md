Rust implementation of GC-Mem (arXiv:2609.16073): temporal dominance Φ_T over a retrieved RAG context.

## Install

```
cargo test
```

## Run

```
cargo test
```

The library has no CLI. Call `temporal_dominance` on a retrieved set `R_k` with an injected conflict oracle `κ`. Optional STM is a FIFO unioned with bag top-k and deduplicated before `Φ_T`. Timestamp re-ranking and multi-factor scoring are baselines, not the protocol.

## What does not match the paper

no MiniLM/Qdrant/Redis, no 137760-chunk benchmark, κ is an injected oracle not an LLM NLI prompt, no generation CR-Acc tables.
