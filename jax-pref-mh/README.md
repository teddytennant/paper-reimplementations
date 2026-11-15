# jax-pref-mh

JAX implementation of Pref-MH from "When Metropolis and Hastings Meet Bradley and
Terry: Exact MCMC From Preference Voting" (arXiv:2609.00905), an exact MH sampler
that decides accept/reject from a fixed budget of binary pairwise judge votes
without ever evaluating the target density.

## install

    uv venv && uv pip install -e .

## run

    python -m pref_mh.demo

Runs the Appendix C.1 synthetic validation: Oracle MH, Pref-MH and the invalid plug-in
baseline on the 241-state target, with cumulative TV distance to pi.

## what is not here

No LLM or VLM judges, so none of the text generation, image generation, molecular
design or machine translation experiments. Judges are synthetic Bradley-Terry judges
built from a latent score on small discrete targets, which is what makes the
exactness claims checkable against a pi you can write down.
