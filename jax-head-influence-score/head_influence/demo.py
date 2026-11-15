"""Train the toy classifier, score its heads, then ablate the dominant ones.

    python -m head_influence.demo
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from .ablation import accuracy, all_subset_masks, accuracy_for_masks, head_mask
from .influence import (
    CATEGORY_NAMES,
    batched_influence,
    layer_scores,
    mean_scores_by_category,
    outcome_categories,
    signal_split,
    top_heads,
)
from .model import ModelConfig, batched_logits
from .task import make_batch
from .train import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--eval-size", type=int, default=512)
    ap.add_argument("--n-ablate", type=int, default=3)
    args = ap.parse_args()

    cfg = ModelConfig()
    t0 = time.time()
    params, acc = train(jax.random.key(args.seed), cfg, steps=args.steps)
    print(f"trained {cfg.n_layers}x{cfg.n_heads} heads in {time.time() - t0:.1f}s, "
          f"held-out accuracy {acc:.3f}")

    tokens, labels = make_batch(jax.random.key(args.seed + 100), args.eval_size,
                                cfg.seq_len, cfg.vocab_size)
    inf = batched_influence(cfg, params, tokens)
    mean_score = np.asarray(inf.score.mean(0))
    mean_abs = np.asarray(jnp.abs(inf.score).mean(0))
    mean_delta = np.asarray(inf.delta.mean(0))
    mean_alpha = np.asarray(inf.alpha.mean(0))

    print("\nS_k,h averaged over the eval set (mean |S| in brackets)")
    header = "layer  alpha_k  " + "  ".join(f"     head {h}" for h in range(cfg.n_heads))
    print(header)
    for k in range(cfg.n_layers):
        cells = "  ".join(f"{mean_score[k, h]:+.4f}[{mean_abs[k, h]:.4f}]"
                          for h in range(cfg.n_heads))
        print(f"  {k}    {mean_alpha[k]:.3f}   {cells}")

    per_layer = np.asarray(jax.vmap(layer_scores)(inf.score).mean(0))
    pos, neg = signal_split(mean_score)
    print("\nmean S_k per layer", " ".join(f"{v:+.4f}" for v in per_layer))
    print(f"mean S_final {float(inf.s_final.mean()):+.4f}, "
          f"positive signal {float(pos):.4f}, negative signal {float(neg):.4f}")
    print("dominant heads by |S|:",
          [(f"L{k}H{h}", round(s, 4)) for k, h, s in top_heads(jnp.asarray(mean_abs),
                                                              args.n_ablate, by="abs")])

    preds = batched_logits(cfg, params, tokens, None).argmax(-1)
    cats = outcome_categories(preds, labels)
    for code, name in CATEGORY_NAMES.items():
        n = int((cats == code).sum())
        if n == 0:
            continue
        table = mean_scores_by_category(inf.score, cats, code)
        best = top_heads(table, 1, by="positive")[0]
        worst = top_heads(table, 1, by="negative")[0]
        print(f"{name} (n={n}): top pro-positive L{best[0]}H{best[1]} {best[2]:+.4f}, "
              f"top pro-negative L{worst[0]}H{worst[1]} {worst[2]:+.4f}")

    top = [(k, h) for k, h, _ in top_heads(jnp.asarray(mean_abs), args.n_ablate, by="abs")]
    base = float(accuracy(cfg, params, tokens, labels))
    ablated = float(accuracy(cfg, params, tokens, labels, head_mask(cfg, top)))
    masks, subsets = all_subset_masks(cfg, args.n_ablate)
    all_acc = np.asarray(accuracy_for_masks(cfg, params, tokens, labels, masks))
    damage = base - all_acc
    top_damage = base - ablated
    pct = float((damage < top_damage).mean())
    print(f"\nablating {args.n_ablate} heads: accuracy {base:.3f} -> {ablated:.3f}")
    print(f"drop {top_damage:.3f} against every other {args.n_ablate}-head choice "
          f"({len(subsets)} of them): mean drop {damage.mean():.3f}, "
          f"worst {damage.max():.3f}, percentile {pct:.3f}")

    # How much of the logit difference those heads write directly, which is the
    # part Delta measures and the part the ranking is built on.
    delta = np.asarray(inf.delta)
    carried = np.array([
        np.abs(delta[:, [k for k, _ in sub], [h for _, h in sub]].sum(1)).mean()
        for sub in subsets])
    idx = [i for i, sub in enumerate(subsets) if set(sub) == set(top)][0]
    print(f"direct logit mass carried: {carried[idx]:.3f} against a mean of "
          f"{carried.mean():.3f}, percentile {(carried < carried[idx]).mean():.3f}")


if __name__ == "__main__":
    main()
