# jax-cadmp

JAX implementation of CADMP, from "Detecting Object Hallucinations in Large
Vision-Language Models via Cross-Modal Attention Drifts and Mask-Based
Verification" (arXiv:2609.02028).

## install

uv pip install -e .

## run

python -m cadmp.demo

The LVLM is an input here, not part of the repo. CADMP takes a cross-modal
attention tensor of shape [L, H, N] plus the target-token probability before and
after masking, so this is the feature extraction and the detector and nothing
else. Nothing is run against LLaVA-1.5-7B, Qwen2.5-VL-7B or InternVL-2.5-4B, and
the POPE, AMBER, Pascal VOC and COCO-Caption numbers are not reproduced. The
demo trains on synthetic attention traces built to have the structure the paper
reports in Figure 3, so its AUROC says the pipeline works and nothing about
whether the paper's result replicates.
