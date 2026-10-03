# nuct

Classical cores of the nonuniform Chebyshev transform, arXiv:2609.40277
(Gupta, *A Nonuniform Chebyshev Transform for Quantum Linear Algebra*).

The package is importable as `nuct`. It is a small JAX library (float64 /
complex128). Quantum circuits are not implemented.

```bash
python -m pytest
```

Run that from this directory. JAX float64 is enabled on import.

## What is implemented

Index convention throughout: **row = frequency / degree, column = node**.

* **Definition 1.1 and Eq. (3).** Nodes `x_k = -1 + 2k/N`, `N = 2^q`, `q >= 2`.
  Unnormalized coefficients `c_j = sum_k f_k T_j(x_k)`, normalized matrix
  `(C_N)_{jk} = T_j(x_k) / sqrt(N)`, and `c = sqrt(N) C_N f`.
* **Theorem 5.1.** `t_k = arccos(x_k) / (2π) ∈ (0, 1/2]`, `t^-_k = (-t_k) mod 1`,
  `(F_t)_{jk} = N^{-1/2} exp(-2π i j t_k)`, and `C_N = (F_t + F_{t^-}) / 2`.
  The endpoint `t_0 = 1/2` is stored exactly.
* **Eq. (6), integer split.** `τ = a / 2^m` is split from the bits of `a`, not
  by rounding a float of `τ`. `sunw = round(N τ)`, `σ = sunw mod N`,
  `z ∈ [-1, 1]`, and `N τ = sunw + z/2`. The phase identity
  `exp(-2π i k τ) = exp(-2π i k σ / N) exp(-i π (k/N) z)` is tested,
  including the wrap `sunw = N`, `σ = 0`.
* **Kernel factorization.** `H(x, z) = exp(-i π x z / 2)` on `[-1, 1]^2`,
  with coefficients `α_{ℓ r} = 4 i^r J_{(ℓ+r)/2}(-π/4) J_{(r-ℓ)/2}(-π/4)`
  when `ℓ ≡ r (mod 2)`, else 0, and `α'_{ℓ r} = η_ℓ η_r α_{ℓ r}`
  (`η_0 = 1/2`, `η_n = 1` for `n >= 1`). `K` is an argument. The Lambert-W
  schedule is not hard-coded.
* **Eq. (31) on stored nodes.** `w_k = 2k/N - 1`, `v_r(k) = T_r(w_k)`,
  `u_r(j) = exp(-i π z_j / 2) sum_ℓ α'_{ℓ r} T_ℓ(z_j)`, unitary DFT
  `F_{kσ} = N^{-1/2} exp(-2π i k σ / N)`, and
  `F_τ^{(K)} = sum_{r < K} diag(v_r) F M_σ diag(u_r)`. As `K` grows this
  approaches the NUDFT of the **stored** nodes, in entrywise and operator norm.
* **Conditioning-free angle rounding.** The exact stored offset goes into
  `arccos`; only the output angle is rounded. `|T_ℓ(z) - cos(ℓ φ̃)| <= ℓ δ_φ`
  with no factor `κ(y) = 1/sqrt(1-y^2)`. Checked at `z = 1 - 1e-12`.
* **Node-error operator bound.** `||F_t - F_τ||_2 <= (2π / sqrt(3)) N^{3/2} ||t - τ||_∞`,
  checked on a random perturbation with a positive left-hand side.
* **Classical row oracle.** Given row `ℓ`, the columns `j` with `σ_j = ℓ`.
  `d_τ = max d_ℓ`, and `||M_σ||_2 = sqrt(d_τ)`. For stored equispaced-in-`x`
  nodes with `N ||t - τ||_∞ < 1/(2π)`, `d_τ <= min{5, N}` on `N = 16` and
  `N = 32`.

## Omissions

* Quantum circuits: block encodings, QROM, LCU state preparation, SELECT,
  controlled rotations, the QFT gate list, and amplitude amplification.
* Gate counts, qubit counts, and Clifford+T synthesis.
* The eight-candidate quantum row-access circuit of Proposition 5.5. The
  classical oracle and the multiplicity bound are implemented; the gate
  construction is not.
* Lemma 5.6's controlled reuse of a negative-node circuit. Theorem 5.1 uses
  the exact reflected nodes `t^-`. The stored negative-node split is not
  implemented.
* The end-to-end block-encoding error budget (Theorem 5.10) and the
  Lambert-W formula for `K(ε)`.

## Conventions the PDF leaves ambiguous

1. **Tie convention.** Lemma 4.2 says `round` has a fixed tie convention and
   does not name it. This code uses **round half away from zero** (half up
   for `τ >= 0`). At a tie, `sunw = floor(N τ) + 1` and `z = -1`. The pinned
   example is `N = 8`, `m = 4`, `a = 15` (`τ = 1 - 1/(2N)`): `sunw = 8`,
   `σ = 0`, `z = -1`. Round-half-to-even also keeps `|z| <= 1`. It is not
   used, and the formula is not switched to make `d_τ <= 5` true. Under this
   convention the Proposition 5.5 hypothesis holds for the tested stored
   nodes, and `d_τ <= 5` holds with it.
2. **Index names.** Theorem 5.1 writes `(F_t)_{jk}` with `j` the frequency
   and `k` the node. Section 4 writes `(F_t)_{kj}` with those names swapped.
   Same matrix. This library uses row = frequency, column = node.
3. **Equation numbers.** The implementation brief's operator-norm "Eq. (28)"
   is the display the extracted PDF labels Eq. (29):
   `||F_t - F_τ|| <= (2π/sqrt(3)) N^{3/2} ||t-τ||_∞`. The brief's
   conditioning-free "Eq. (27)" is the claim in Lemma 4.3 (extracted PDF
   Eq. (39)): `|T_n(w) - cos(n φ̃)| <= n δ_φ`, with no `κ`. The formulas
   implemented are the ones the brief states.
4. **Angle grid.** The paper only requires `|φ̃ - arccos z| <= δ_φ`. Rounding
   is to the nearest multiple of `2 δ_φ`, so the error is at most `δ_φ`.
5. **Quantizing irrational `t_k`.** The split of a stored integer `a` is exact.
   Choosing `a` from `t_k = arccos(x_k)/(2π)` is only specified up to
   `|τ - t| <= 2^{-m}`, an `m`-bit value in `[0, 1/2]`, with `k = 0` exact.
   That step is float64 round-half-up of `t * 2^m` (`m <= 53`, so `2^m` is
   an exact float64 integer), then the integer split. Tests check the
   realized `|τ - t|`.
6. **`m <= q`.** The paper's bit formula for `z` assumes `m >= q + 1`. If
   `m <= q` then `N τ` is an integer, `z = 0`, and `N τ = sunw + z/2` still
   holds. `τ = 1` (`a = 2^m`) is accepted as a boundary; the paper's nodes
   lie in `[0, 1)`.
