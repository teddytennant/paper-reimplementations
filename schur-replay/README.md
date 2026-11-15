# schur-replay

Schur Replay NVFP4 scale selection (arXiv 2609.36654, section 4.1 and Appendix Algorithm 1).

pip install -e .

pytest -q

Does not match the paper's Qwen or Llama runs, does not report W4A4 recovery numbers, and does not include ModelOpt or LLM Compressor infrastructure. Algorithms 3-5 (memory tiering, checkpoints, tensor-parallel row splits) are out of scope. Column order is the order given by the caller.
