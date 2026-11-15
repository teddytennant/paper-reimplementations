"""Exact compilation of a routed linear operator over real vectors."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


def estimate_saving(k: int, m: int) -> int:
    """Estimated saving of a sum of m inputs shared by k outputs.

    Separate accumulation costs k*m add/sub operations. Forming the shared
    sum and accumulating it into the k outputs costs m+k. The estimate is
    k*m - m - k. A candidate is accepted only when this value is positive,
    so a 2-input sum shared by 2 outputs (saving 0) is not reused.
    """
    return k * m - m - k


@dataclass(frozen=True)
class SharedSum:
    """Canonical signed sum and the outputs that reuse it.

    terms are (input_index, sign) with sign in {+1, -1}.
    users are (output_index, orientation). Orientation -1 means the output
    adds -U, the all-signs-reversed set.
    """

    terms: tuple[tuple[int, int], ...]
    users: tuple[tuple[int, int], ...]
    saving: int

    @property
    def inputs(self) -> frozenset[int]:
        return frozenset(i for i, _s in self.terms)


@dataclass(frozen=True)
class CompiledLinear:
    signed_h: jnp.ndarray
    raw_mask: jnp.ndarray
    shares: tuple[SharedSum, ...]
    naive_pmult: int
    compiled_pmult: int
    naive_rescale: int
    compiled_rescale: int
    shared_sum_saving: int

    def naive(self, inputs, gammas, raw_weights):
        """y_j = sum_i gamma_j * h_{j,i} * A_i + sum_i g_{j,i} * A_i."""
        return _evaluate(self, inputs, gammas, raw_weights, shared=False)

    def compiled(self, inputs, gammas, raw_weights):
        """Grouped reconstruction, shared signed sums, one rescale count."""
        return _evaluate(self, inputs, gammas, raw_weights, shared=True)


def compile_routed_linear(signed_h, raw_mask) -> CompiledLinear:
    """Compile routes over packed inputs.

    signed_h[j, i] in {-1, 0, +1}: nonzero values are the signed route
    (add, subtract, or skip when 0). raw_mask[j, i] marks a raw slot-wise
    product. A pair is one route, not both.
    """
    h = jnp.asarray(signed_h)
    if h.ndim != 2:
        raise ValueError("signed_h must have shape (outputs, inputs)")
    h = h.astype(jnp.int32)
    if not jnp.all((h >= -1) & (h <= 1)):
        raise ValueError("signed h must be in {-1, 0, +1}")
    raw = jnp.asarray(raw_mask).astype(bool)
    if raw.shape != h.shape:
        raise ValueError("raw_mask must match signed_h")
    if bool(jnp.any((h != 0) & raw)):
        raise ValueError("a group is either signed or raw, not both")

    n_out, _n_in = h.shape
    has_signed = jnp.any(h != 0, axis=1)
    has_raw = jnp.any(raw, axis=1)
    naive_pmult = int(jnp.sum(h != 0) + jnp.sum(raw))
    # Group reconstruction: one gamma multiply per output with a signed term.
    compiled_pmult = int(jnp.sum(has_signed) + jnp.sum(raw))
    naive_rescale = naive_pmult
    compiled_rescale = int(jnp.sum(has_signed | has_raw))
    shares = _select_shares(h)
    saving = sum(share.saving for share in shares)
    return CompiledLinear(
        signed_h=h,
        raw_mask=raw,
        shares=shares,
        naive_pmult=naive_pmult,
        compiled_pmult=compiled_pmult,
        naive_rescale=naive_rescale,
        compiled_rescale=compiled_rescale,
        shared_sum_saving=saving,
    )


def _canonicalize(terms: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    ordered = tuple(sorted((int(i), int(s)) for i, s in terms))
    flipped = tuple(sorted((i, -s) for i, s in ordered))
    return min(ordered, flipped)


def _contains(signed_map: dict[int, int], terms: tuple[tuple[int, int], ...]) -> bool:
    return all(signed_map.get(i) == s for i, s in terms)


def _select_shares(signed_h: jnp.ndarray) -> tuple[SharedSum, ...]:
    n_out, n_in = signed_h.shape
    maps: list[dict[int, int]] = []
    for j in range(n_out):
        row = {}
        for i in range(n_in):
            sign = int(signed_h[j, i])
            if sign != 0:
                row[i] = sign
        maps.append(row)

    patterns: set[tuple[tuple[int, int], ...]] = set()
    for a in range(n_out):
        for b in range(a + 1, n_out):
            direct: list[tuple[int, int]] = []
            reversed_terms: list[tuple[int, int]] = []
            for i in set(maps[a]) & set(maps[b]):
                if maps[a][i] == maps[b][i]:
                    direct.append((i, maps[a][i]))
                elif maps[a][i] == -maps[b][i]:
                    reversed_terms.append((i, maps[a][i]))
            for terms in (direct, reversed_terms):
                if len(terms) >= 2:
                    patterns.add(_canonicalize(terms))

    scored: list[tuple[int, int, tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]] = []
    for terms in patterns:
        flipped = tuple(sorted((i, -s) for i, s in terms))
        users: list[tuple[int, int]] = []
        for j, signed_map in enumerate(maps):
            if _contains(signed_map, terms):
                users.append((j, 1))
            elif _contains(signed_map, flipped):
                users.append((j, -1))
        k = len(users)
        m = len(terms)
        saving = estimate_saving(k, m)
        if saving > 0 and k >= 2:
            scored.append((saving, m, terms, tuple(users)))

    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    used = [set() for _ in range(n_out)]
    accepted: list[SharedSum] = []
    for _saving, _m, terms, users in scored:
        inputs = {i for i, _s in terms}
        viable = [
            (j, orient)
            for j, orient in users
            if inputs.isdisjoint(used[j])
        ]
        k = len(viable)
        saving = estimate_saving(k, len(terms))
        if saving <= 0 or k < 2:
            continue
        accepted.append(SharedSum(terms=terms, users=tuple(viable), saving=saving))
        for j, _orient in viable:
            used[j].update(inputs)
    return tuple(accepted)


def _as_batch(inputs, gammas, raw_weights, n_out: int, n_in: int):
    a = jnp.asarray(inputs)
    if a.ndim == 1:
        a = a[:, None]
    if a.shape[0] != n_in:
        raise ValueError("inputs must have one vector per packed input")
    slots = a.shape[-1]
    gamma = jnp.asarray(gammas)
    if gamma.ndim == 1:
        gamma = gamma[:, None]
    if gamma.shape[0] != n_out:
        raise ValueError("gammas must have one vector per output")
    if gamma.shape[-1] == 1 and slots != 1:
        gamma = jnp.broadcast_to(gamma, (n_out, slots))
    raw = jnp.asarray(raw_weights)
    if raw.ndim == 2:
        raw = raw[:, :, None]
    if raw.shape[0] != n_out or raw.shape[1] != n_in:
        raise ValueError("raw_weights must have shape (outputs, inputs, slots)")
    if raw.shape[-1] == 1 and slots != 1:
        raw = jnp.broadcast_to(raw, (n_out, n_in, slots))
    return a, gamma, raw


def _evaluate(plan: CompiledLinear, inputs, gammas, raw_weights, shared: bool):
    n_out, n_in = plan.signed_h.shape
    a, gamma, raw = _as_batch(inputs, gammas, raw_weights, n_out, n_in)
    slots = a.shape[-1]
    consumed = [set() for _ in range(n_out)]
    shared_acc = [jnp.zeros((slots,), dtype=a.dtype) for _ in range(n_out)]
    if shared:
        for share in plan.shares:
            total = jnp.zeros((slots,), dtype=a.dtype)
            for i, sign in share.terms:
                total = total + sign * a[i]
            for j, orient in share.users:
                shared_acc[j] = shared_acc[j] + orient * total
                consumed[j].update(share.inputs)

    ys = []
    for j in range(n_out):
        if not shared:
            y = jnp.zeros((slots,), dtype=a.dtype)
            for i in range(n_in):
                sign = int(plan.signed_h[j, i])
                if sign != 0:
                    y = y + gamma[j] * (sign * a[i])
                if bool(plan.raw_mask[j, i]):
                    y = y + raw[j, i] * a[i]
            ys.append(y)
            continue
        acc = shared_acc[j]
        has_signed = bool(consumed[j])
        for i in range(n_in):
            sign = int(plan.signed_h[j, i])
            if sign == 0 or i in consumed[j]:
                continue
            acc = acc + sign * a[i]
            has_signed = True
        if has_signed:
            y = gamma[j] * acc
        else:
            y = jnp.zeros((slots,), dtype=a.dtype)
        for i in range(n_in):
            if bool(plan.raw_mask[j, i]):
                y = y + raw[j, i] * a[i]
        ys.append(y)
    return jnp.stack(ys, axis=0)
