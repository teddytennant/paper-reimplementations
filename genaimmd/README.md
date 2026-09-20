GenAIMMD (arXiv:2609.20461) as a small JAX library on the 2D Wolfe-Quapp toy.

```
pip install -e . && pytest
```

What does not match the paper: no polymer; no 50-cycle / 200-epoch training in the default smoke; no PSIS fine-tuning; affine RealNVP only; CPU tests use tiny N and few steps; authors' PyTorch scripts at CompPhysVienna/paper_genaimmd are not used. The 2D CV is ξ = pB (Fig. 2B) rather than the log-committor q used for the polymer; bias windows are sigmoid(round(q)), i.e. integer centers in log-space mapped back to pB. Committor hidden activations are tanh and RealNVP coupling nets use ReLU with s clipped to [-5, 5] (neither specified). The third Wolfe-Quapp well is not a metastable state. Conditioned BG energy training uses U'_ξ̂ from eq. (10) in the KL term.
