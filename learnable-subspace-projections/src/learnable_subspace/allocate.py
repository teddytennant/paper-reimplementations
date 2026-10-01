"""Greedy KL allocation of dropped directions under a savings target."""


def monotone_running_max(delta):
    """Replace delta(k) by max_{j <= k} delta(j) so the table is nondecreasing."""
    running = None
    out = []
    for value in delta:
        value = float(value)
        running = value if running is None else max(running, value)
        out.append(running)
    return out


def savings(k, N, c, d):
    """Parameter savings at dropped-direction count k.

    s(0) = 0. For k > 0, s(k) = N - c * (d - k).
    """
    k = int(k)
    if k == 0:
        return 0.0
    return float(N) - float(c) * (float(d) - k)


def untied_c(din, dout):
    """c_u = din + dout for an untied layer."""
    return int(din) + int(dout)


def untied_unit(din, dout, delta, side="input"):
    """One untied allocation unit.

    N_u is the dense weight count din * dout. Bias is excluded because it is
    never projected. d_u is din on the input side and dout on the output side,
    so d_u - k is the kept factor rank.
    """
    if side not in ("input", "output"):
        raise ValueError("side must be 'input' or 'output'")
    return {
        "delta": list(delta),
        "N": int(din) * int(dout),
        "c": untied_c(din, dout),
        "d": int(din) if side == "input" else int(dout),
    }


def allocate(units, target_savings):
    """Greedy moves until total savings reach the target.

    Each unit supplies delta, N, c, and d. delta is made nondecreasing by a
    running maximum. While savings < target, pick the move k -> k' (k' > k)
    with the smallest (delta(k') - delta(k)) / (s(k') - s(k)) among moves
    that strictly increase savings. Ties break toward the earlier unit, then
    the smaller k'.

    Jumps are required: s(k) can dip below s(0) for small k, so a unit may
    need to skip those operating points.
    """
    deltas = [monotone_running_max(u["delta"]) for u in units]
    ks = [0] * len(units)

    def total(chosen):
        return sum(
            savings(k, u["N"], u["c"], u["d"]) for k, u in zip(chosen, units)
        )

    max_steps = sum(max(len(table) - 1, 0) for table in deltas) + 1
    for _ in range(max_steps):
        if total(ks) >= target_savings:
            break
        best = None
        for i, u in enumerate(units):
            k = ks[i]
            s_k = savings(k, u["N"], u["c"], u["d"])
            table = deltas[i]
            for k_prime in range(k + 1, len(table)):
                s_kp = savings(k_prime, u["N"], u["c"], u["d"])
                ds = s_kp - s_k
                if ds <= 0.0:
                    continue
                ratio = (table[k_prime] - table[k]) / ds
                cand = (ratio, i, k_prime)
                if best is None or cand < best:
                    best = cand
        if best is None:
            break
        _ratio, i, k_prime = best
        ks[i] = k_prime
    return tuple(ks)
