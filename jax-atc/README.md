JAX implementation of Abstract Token Curriculum (ATC) for parity ([arXiv:2609.19717](https://arxiv.org/abs/2609.19717)): a one-layer transformer trained on a bit-curriculum, appending its last hidden embedding as the next continuous chain-of-thought token with answer-only supervision.

```
pip install -e . && pytest
```

## What does not match the paper

- Phase `r` uses `r-1` continuous CoT tokens plus a learned final-answer token and predicts from that final position (§3 / eqs 1–9). Section 5.1 instead says the model is given `i` intermediate tokens in phase `i`.
- Next CoT tokens are the stop-gradient last-layer hidden state at `z_final` (`f_tf` of §2, identity `T_i`). Section 3 says to append the hidden state at the last CoT position (or at `x_n` after phase 1); that extractor is available as `ATCConfig.extract="xn_or_last_cot"` but is not the default.
- The default model is the theoretical one-layer softmax attention of §3.1 (no MLP, no LayerNorm, no residual, no positions). The tiny training path is a 1-layer 1-head transformer with residual, LayerNorm, MLP, and positions, embedding dim 32 rather than 64.
- No n=20 / 15,000-iteration / 20-seed / 20-phase run (Figure 2). Smoke training is n=4, two phases, a few hundred AdamW steps.
- Graph reachability (trees, candidate hiding ρ=0.75, curriculum backtracking, truncated BPTT) and arithmetic addition are not implemented.
- No HuggingFace models or downloaded datasets.
