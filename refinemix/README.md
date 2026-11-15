# RefineMix

RefineMix trains a discrete diffusion denoiser on out-of-distribution sequences only at noise levels t <= tmax, and on the target alone above that cutoff (Algorithm 1 of arXiv 2610.12340).

From this directory:

    python -m pytest -q
    python scripts/synthetic_shift.py --steps 50 --device cpu

No install is required. `python -c "import refinemix"` works from the repository root.

This is not the 139.3M DiT, not BERT tokenization, not the text/protein corpora, not the conditional baseline, and not EMA decay 0.9999 training at 3e-4 for thousands of steps. The denoiser here is a small pure JAX module. The log-linear schedule and the masked continuous-time NELBO match Appendix B and equation 13. Theorem 1 is checked on an enumerable toy domain, not by sampling a trained model.
