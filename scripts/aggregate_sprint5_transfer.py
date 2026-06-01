#!/usr/bin/env python3
"""Aggregate Sprint 5 transfer-eval CSVs into a wide method × (trained_on × eval_on) matrix.

For each (method, trained_on, eval_on) cell we report:
  mean coverage_rate ± 95% bootstrap CI (across the 3 training seeds, where each
  seed contributes 90 episodes -- 30 ep × 3 eval seeds).

Writes:
  results/SPRINT5_TRANSFER_MATRIX.csv -- one row per (method, trained_on, eval_on)
"""
from __future__ import annotations

import glob
import os
import re
import numpy as np
import pandas as pd

RESULTS = "/home/aiteam1/sonthh/MATE/results"
OUT = os.path.join(RESULTS, "SPRINT5_TRANSFER_MATRIX.csv")

PATTERN = re.compile(
    r"sprint5_(?P<method>[a-z]+)_trainedOn_(?P<trained_on>[A-Za-z0-9\-]+)_seed(?P<seed>\d+)_transfer\.csv"
)

def bootstrap_ci(x: np.ndarray, n=2000, alpha=0.05, rng=None) -> tuple[float, float]:
    rng = rng or np.random.default_rng(0)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def main():
    paths = sorted(glob.glob(os.path.join(RESULTS, "sprint5_*_transfer.csv")))
    print(f"Found {len(paths)} sprint5 transfer CSVs")
    rows = []
    for p in paths:
        m = PATTERN.match(os.path.basename(p))
        if not m:
            print(f"  SKIP unrecognized: {p}")
            continue
        method = m.group("method")
        trained_on = m.group("trained_on")
        seed = int(m.group("seed"))
        df = pd.read_csv(p)
        for sc, g in df.groupby("scenario"):
            cov = g["mean_coverage_rate"].to_numpy()
            ret = g["episode_return"].to_numpy()
            rows.append(dict(
                method=method,
                trained_on=trained_on,
                eval_on=sc,
                seed=seed,
                n_eps=len(g),
                mean_cov=float(np.mean(cov)),
                mean_ret=float(np.mean(ret)),
            ))
    long = pd.DataFrame(rows)
    long.to_csv(os.path.join(RESULTS, "SPRINT5_TRANSFER_LONG.csv"), index=False)
    print(f"  wrote LONG ({len(long)} rows)")

    # Aggregate over training seeds (3 seeds → mean ± CI of per-seed cov means)
    agg_rows = []
    for (method, trained_on, eval_on), g in long.groupby(["method", "trained_on", "eval_on"]):
        per_seed_cov = g["mean_cov"].to_numpy()
        per_seed_ret = g["mean_ret"].to_numpy()
        cov_lo, cov_hi = bootstrap_ci(per_seed_cov)
        ret_lo, ret_hi = bootstrap_ci(per_seed_ret)
        agg_rows.append(dict(
            method=method,
            trained_on=trained_on,
            eval_on=eval_on,
            n_seeds=len(g),
            mean_cov=float(per_seed_cov.mean()),
            cov_ci_lo=cov_lo,
            cov_ci_hi=cov_hi,
            mean_ret=float(per_seed_ret.mean()),
            ret_ci_lo=ret_lo,
            ret_ci_hi=ret_hi,
        ))
    agg = pd.DataFrame(agg_rows).sort_values(["method", "trained_on", "eval_on"])
    agg.to_csv(OUT, index=False)
    print(f"  wrote MATRIX ({len(agg)} rows) → {OUT}")
    print()
    print("=" * 80)
    print("EXTENDED TRANSFER MATRIX (Sprint 5)")
    print("=" * 80)
    for method in sorted(agg.method.unique()):
        sub = agg[agg.method == method]
        print(f"\n--- {method.upper()} ---")
        # Wide: trained_on (rows) × eval_on (cols)
        wide = sub.pivot_table(index="trained_on", columns="eval_on", values="mean_cov")
        print(wide.round(3).to_string())
    print()


if __name__ == "__main__":
    main()
