# jax-echo-oftrl

JAX implementation of "Constant Individual Regret in General Games"
(arXiv:2608.31166).

## install

uv pip install -e .

## run

python -m echo_oftrl.demo --horizon 2000

The cascade, the filter identities and Lemma 3.1's lifted map are the paper's,
checked to 1e-11 in float64. The regret constant is not. Theorem 3.3 needs
eta = 2^-94 / (N^20 Xi_max), about 1e-40, which does nothing in finite time, so
this runs at eta = 1 and only the horizon-independence gets tested. I could not
measure any gap between the cascade and ordinary one-step optimism on games
this small, at any learning rate I tried; both are flat, so the control the
tests use is optimism switched off, which does separate. All players get the
same action count, Q_m(theta) comes from 60 bisection steps, and float64 is
forced on at import because the bisection does not survive float32.

The suite passes on CPU and on one H200 under CUDA. Two float64 results that
agree bit for bit on CPU can differ by an ulp on the GPU when XLA compiles them
into differently fused kernels, so the one test that wants exact equality,
`test_updates_use_only_the_players_own_gradients`, gets it by calling the same
function twice on the same shapes and changing only the other players'
gradients. Comparisons between the fused run and the replay driver carry a
tolerance instead.
