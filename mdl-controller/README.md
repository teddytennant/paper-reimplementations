Zero-parameter Memory Decision Layer (MDL) geometric controller ([arXiv:2609.22043](https://arxiv.org/abs/2609.22043)).

```
pip install -e '.[tests]'
python -m pytest
```

## What does not match the paper
- no MiniLM / sentence-transformers
- no TruthfulQA / HaluEval / LLM backends
- no 0.14 ms latency claim
- nmin/nmax not calibrated on 80-question splits (exposed as args)
- lexical conflict detector is a small keyword/antonym stub, not the paper's full detector
- no 56.04% hallucination reduction experiment
