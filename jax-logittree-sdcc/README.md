# jax-logittree-sdcc

A JAX implementation of the trajectory tree, LogitTree, the packed 4D attention mask
and SDCC from "MemoryWalker: Stop Training Agents on Contexts They Never Saw"
(arXiv:2609.00865).

## install

    uv venv
    uv pip install -e ".[dev]"

## run

    .venv/bin/python -m pytest -q
    .venv/bin/python -m memorywalker.demo

## what this does not do

The thing being checked is gradient equivalence, in float64, on a toy two-layer
transformer over rollouts of a dozen tokens. There is no agent, no retriever, no
benchmark: the seven web-search evaluations with
TC-RAG, AgentFold and MemexRL, the Qwen3-4B drift numbers behind Figure 1b, and the
WideSearch results are all out of reach here. The sign asymmetry the paper reports
between the two pitfalls belongs to a trained model, so the pitfall tests check that
each one leaks rather than which way. Section 4.2's mask over the physical union is
off by one at a junction and stops being exact past one layer; the 4D path uses
Appendix G's branch-replicated packing instead, with the union form kept in
`union_view.py` next to a test of where it breaks.
