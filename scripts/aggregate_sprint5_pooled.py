#!/usr/bin/env python3
"""Pool all episodes per (method, trained_on, eval_on) and report mean ± SE
in the same format used by the existing EVAL_MATRIX_cross_seed.csv."""
from __future__ import annotations

import glob
import os
import re
import numpy as np
import pandas as pd

RESULTS = "/home/aiteam1/sonthh/MATE/results"
OUT = os.path.join(RESULTS, "SPRINT5_TRANSFER_POOLED.csv")
PATTERN = re.compile(r"sprint5_(?P<method>[a-z]+)_trainedOn_(?P<trained_on>[A-Za-z0-9\-]+)_seed(?P<seed>\d+)_transfer\.csv")

def main():
    paths = sorted(glob.glob(os.path.join(RESULTS, "sprint5_*_transfer.csv")))
    rows = []
    for p in paths:
        m = PATTERN.match(os.path.basename(p))
        method, trained_on, seed = m.group("method"), m.group("trained_on"), int(m.group("seed"))
        df = pd.read_csv(p)
        df["method"] = method
        df["trained_on"] = trained_on
        df["train_seed"] = seed
        rows.append(df)
    big = pd.concat(rows, ignore_index=True)
    print(f"Combined {len(big)} episodes")
    out_rows = []
    for (method, trained_on, eval_on), g in big.groupby(["method", "trained_on", "scenario"]):
        cov = g["mean_coverage_rate"].to_numpy()
        ret = g["episode_return"].to_numpy()
        out_rows.append(dict(
            method=method, trained_on=trained_on, eval_on=eval_on,
            n=len(cov),
            mean_cov=float(cov.mean()),
            se_cov=float(cov.std(ddof=1) / np.sqrt(len(cov))),
            mean_ret=float(ret.mean()),
            se_ret=float(ret.std(ddof=1) / np.sqrt(len(ret))),
        ))
    agg = pd.DataFrame(out_rows).sort_values(["method", "trained_on", "eval_on"])
    agg.to_csv(OUT, index=False)
    print(f"Wrote {OUT}")
    print()
    print("Per-cell pooled coverage (mean ± SE):")
    for method in sorted(agg.method.unique()):
        print(f"\n--- {method.upper()} ---")
        sub = agg[agg.method == method].copy()
        sub["cell"] = sub.apply(lambda r: f"{r['mean_cov']:.3f} ± {r['se_cov']:.3f} (n={r['n']})", axis=1)
        wide = sub.pivot_table(index="trained_on", columns="eval_on", values="cell", aggfunc="first")
        print(wide.to_string())

if __name__ == "__main__":
    main()
