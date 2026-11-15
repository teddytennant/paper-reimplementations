"""Figure 2 tiny vocab: print Marginalize-It and End-Of-Token B1/B2/B3."""

from __future__ import annotations

import jax.numpy as jnp

from jax_byte_logits import EOT, byte_probabilities, token_byte_tables

TOKENS = ["isu", "isk", "is", "impl", "imap", "ino", "mit", "max", "new"]
PROBS = jnp.asarray([0.4, 0.1, 0.1, 0.05, 0.05, 0.1, 0.05, 0.05, 0.1])


def _fmt(dist, keys):
    parts = []
    for name, idx in keys:
        parts.append(f"P({name})={float(dist[idx]):.6f}")
    return " ".join(parts)


def main() -> int:
    gold = "isu"
    table, lengths = token_byte_tables(TOKENS, use_eot=False)
    marg = byte_probabilities(PROBS, table, lengths, gold, use_eot=False)

    table_eot, lengths_eot = token_byte_tables(TOKENS, use_eot=True)
    gold_eot = list(gold.encode("utf-8")) + [EOT]
    eot = byte_probabilities(
        PROBS, table_eot, lengths_eot, gold_eot, use_eot=True
    )

    i, m, n, s, u, k = (ord(c) for c in "imnsuk")
    print("Marginalize-It")
    print("  B1", _fmt(marg[0], [("i", i), ("m", m), ("n", n)]))
    print("  B2", _fmt(marg[1], [("s", s), ("m", m), ("n", n)]))
    print("  B3", _fmt(marg[2], [("u", u), ("k", k)]))
    print("End-Of-Token")
    print("  B1", _fmt(eot[0], [("i", i), ("m", m), ("n", n)]))
    print("  B2", _fmt(eot[1], [("s", s), ("m", m), ("n", n)]))
    print(
        "  B3",
        _fmt(eot[2], [("u", u), ("k", k), ("EOT", EOT)]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
