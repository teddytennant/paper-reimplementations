JAX implementation of DIPrune, training-free dual importance pruning of visual tokens, from https://arxiv.org/abs/2610.08341.

## Install

```bash
pip install -e ".[test]"
```

## Run

```bash
JAX_PLATFORMS=cpu python -m pytest -q
JAX_PLATFORMS=cpu python demo.py
```

## What does not match the paper

There is no LLaVA, no Qwen-VL, no benchmark number, and no full transformer forward. `prune_tokens` is a pure function of a token sequence, a visual mask, and visual scores. It is not a hook inside an MLLM. Text tokens are never scored and never removed.

The appendix calls the decoupled-attention subset a window of the most recent text queries, with length about 0.8 times the text length. It does not give an index slice or a rounding rule. The window is the last `L_win` rows of `Q`. `L_win` is `max(1, min(M, round(window_ratio * M)))`, and `round` is Python `round` (halves to even).

The paper does not specify rank ties or selection ties. Smaller index gets the better rank, via a stable sort (a lexicographic sort on score descending, then index ascending). Rank 1 is most salient. When scores tie for inclusion, the smaller index is kept.

The paper prunes at layer 10 on LLaVA-NeXT-7B. That layer index is not hardcoded. The paper default delta is 1, but the scorer does not take a layer index. Callers pass the two rank vectors they already picked. `m_max` defaults to 200, `alpha` to 0.6, `beta` to 0.4, and `window_ratio` to 0.8. Those are the paper defaults, and they are function arguments.

`bound.py` is the theorem in theory.tex: the token-wise inner bound, then the Young relaxation `w_i`. Under a budget of K retained tokens, keeping the K largest `w_i` minimizes that relaxed bound. DIPrune does not compute `w_i` at inference. Gradients are unavailable in the training-free setting, so the inference score is `alpha * S_align + beta * S_evol`, which replaces `w_i`.
