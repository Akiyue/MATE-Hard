"""Unified evaluation matrix.

Reads all CSVs from ``results/`` matching a glob pattern and produces:

  1. A consolidated long-format CSV with columns
     [method, variant, scenario, seed, episode, coverage_rate, real_coverage_rate,
      transport_rate, episode_return, n].
  2. A wide-format summary table (method × scenario → mean ± SE).
  3. A pairwise-significance table (Welch's t, paired bootstrap CI, Cohen's d).

The script assumes the canonical naming conventions used by Sprint 1:

  results/baseline_<inner>{,_<adapter>-MH}{,_<scenario>}.csv
  results/dqn_v8_final_eval.csv                            # legacy DQN seed 0
  results/dqn_seed1_final_eval.csv                         # DQN seed 1
  results/dqn_seed2_final_eval.csv                         # DQN seed 2 (Sprint 2)
  results/tcqmix_v1_step2p6M_eval.csv                      # legacy TC-QMIX seed 0
  results/tcqmix_seed1_eval.csv                            # TC-QMIX seed 1
  results/vdn_seed*_eval.csv                               # VDN seeds 0-2
  results/qmix_vanilla_seed*_eval.csv                      # vanilla QMIX
  results/mappo_seed*_eval.csv                             # MAPPO

For each (method, scenario) cell with multiple seeds the per-episode arrays
from all seeds are concatenated before computing mean ± SE and significance.

Usage:
    uv run python -m mate_marl.scripts.eval_matrix \\
        --results-dir results \\
        --out-long results/EVAL_MATRIX_long.csv \\
        --out-wide results/EVAL_MATRIX_wide.csv \\
        --out-sig  results/EVAL_MATRIX_pairwise.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
from pathlib import Path
from typing import Optional

import numpy as np

from mate_marl.analysis.stats import compare, ComparisonResult


# Each entry maps a (method, scenario) cell to a list of CSV globs whose rows
# we aggregate. We resolve globs lazily.
def discover_csvs(results_dir: Path) -> dict[tuple[str, str], list[Path]]:
    """Discover which CSVs belong to which (method, scenario, variant) cell.

    Method-naming conventions:
      - rule_<inner>             : rule-based on plain MATE
      - rule_<inner>_naiveMH     : rule-based with MATE-Hard naive adapter
      - rule_<inner>_informedMH  : rule-based with MATE-Hard informed adapter
      - DQN_seed<S>              : Double-DQN (Sprint 0/2)
      - TCQMIX_seed<S>           : TC-QMIX (Sprint 0/2)
      - VDN_seed<S>              : VDN (Sprint 2)
      - QMIX_seed<S>             : Vanilla QMIX (Sprint 2)
      - MAPPO_seed<S>            : MAPPO (Sprint 2)
    """
    out: dict[tuple[str, str], list[Path]] = {}

    def add(method: str, scenario: str, p: Path):
        out.setdefault((method, scenario), []).append(p)

    # --- Rule-based plain MATE (16 cells) ---
    plain_inner_map = {
        "MATE-4v8-9": "baseline_{inner}.csv",
        "MATE-2v4-9": "baseline_{inner}_MATE-2v4-9.csv",
        "MATE-4v4-9": "baseline_{inner}_MATE-4v4-9.csv",
        "MATE-8v8-9": "baseline_{inner}_MATE-8v8-9.csv",
    }
    for inner in ("random", "naive", "greedy", "heuristic"):
        for scenario, tmpl in plain_inner_map.items():
            p = results_dir / tmpl.format(inner=inner)
            if p.exists():
                add(f"rule_{inner}", scenario, p)

    # --- Rule-based MATE-Hard naive + informed (32 cells) ---
    for adapter in ("naive", "informed"):
        for inner in ("random", "naive", "greedy", "heuristic"):
            for scenario in ("MATE-2v4-9", "MATE-4v4-9", "MATE-4v8-9", "MATE-8v8-9"):
                p = results_dir / f"baseline_{inner}_{adapter}-MH_{scenario}.csv"
                if p.exists():
                    add(f"rule_{inner}_{adapter}MH", scenario, p)

    # --- Learned-method final-eval CSVs ---
    learned_globs = {
        "DQN_seed0": ["dqn_v8_final_eval.csv"],
        "DQN_seed1": ["dqn_seed1_final_eval.csv"],
        "DQN_seed2": ["dqn_seed2_final_eval.csv"],
        "TCQMIX_seed0": ["tcqmix_v1_step2p6M_eval.csv"],
        "TCQMIX_seed1": ["tcqmix_seed1_eval.csv"],
        "TCQMIX_seed2": ["tcqmix_seed2_eval.csv"],
        "TCQMIX_types_seed0": ["tcqmix_types_seed0_eval.csv"],
        "TCQMIX_types_seed1": ["tcqmix_types_seed1_eval.csv"],
        "TCQMIX_types_seed2": ["tcqmix_types_seed2_eval.csv"],
        "VDN_seed0": ["vdn_seed0_eval.csv"],
        "VDN_seed1": ["vdn_seed1_eval.csv"],
        "VDN_seed2": ["vdn_seed2_eval.csv"],
        "QMIX_seed0": ["qmix_vanilla_seed0_eval.csv"],
        "QMIX_seed1": ["qmix_vanilla_seed1_eval.csv"],
        "QMIX_seed2": ["qmix_vanilla_seed2_eval.csv"],
        "MAPPO_seed0": ["mappo_seed0_eval.csv"],
        "MAPPO_seed1": ["mappo_seed1_eval.csv"],
        "MAPPO_seed2": ["mappo_seed2_eval.csv"],
    }
    for method, patterns in learned_globs.items():
        for pat in patterns:
            for p in results_dir.glob(pat):
                # Learned-method CSVs contain a 'scenario' column; we'll
                # de-aggregate by scenario at load time. Here we just
                # register the file under a sentinel scenario "*".
                add(method, "*", p)

    # Learned-method transfer CSVs (multi-scenario in one file).
    # Each CSV evaluates a 4v8-trained checkpoint on MATE-{2v4,4v4,8v8}.
    transfer_globs = {
        # Legacy seed-0 transfers (n=40 episodes, single eval seed).
        "DQN_seed0_transfer": ["dqn_v8_transfer_eval.csv"],
        "TCQMIX_seed0_transfer": ["tcqmix_v1_transfer_eval.csv"],
        # Sprint 3 transfers (n=90 episodes = 3 eval seeds × 30 episodes).
        "DQN_seed1_transfer": ["DQN_seed1_transfer.csv"],
        "DQN_seed2_transfer": ["DQN_seed2_transfer.csv"],
        "TCQMIX_seed1_transfer": ["TCQMIX_seed1_transfer.csv"],
        "TCQMIX_seed2_transfer": ["TCQMIX_seed2_transfer.csv"],
        "TCQMIX_types_seed0_transfer": ["TCQMIX_types_seed0_transfer.csv"],
        "TCQMIX_types_seed1_transfer": ["TCQMIX_types_seed1_transfer.csv"],
        "TCQMIX_types_seed2_transfer": ["TCQMIX_types_seed2_transfer.csv"],
        "QMIX_seed0_transfer": ["QMIX_vanilla_seed0_transfer.csv"],
        "QMIX_seed1_transfer": ["QMIX_vanilla_seed1_transfer.csv"],
        "QMIX_seed2_transfer": ["QMIX_vanilla_seed2_transfer.csv"],
        "VDN_seed0_transfer": ["VDN_seed0_transfer.csv"],
        "VDN_seed1_transfer": ["VDN_seed1_transfer.csv"],
        "VDN_seed2_transfer": ["VDN_seed2_transfer.csv"],
        "MAPPO_seed0_transfer": ["MAPPO_seed0_transfer.csv"],
        "MAPPO_seed1_transfer": ["MAPPO_seed1_transfer.csv"],
        "MAPPO_seed2_transfer": ["MAPPO_seed2_transfer.csv"],
    }
    for method, patterns in transfer_globs.items():
        for pat in patterns:
            for p in results_dir.glob(pat):
                add(method, "*", p)

    return out


def load_csv_rows(p: Path, scenario_filter: Optional[str] = None) -> list[dict]:
    """Load rows from a CSV, filtering by 'scenario' column if specified."""
    rows: list[dict] = []
    with p.open() as f:
        for r in csv.DictReader(f):
            if scenario_filter and r.get("scenario") and r["scenario"] != scenario_filter:
                continue
            rows.append(r)
    return rows


def aggregate_long(discovered: dict[tuple[str, str], list[Path]]) -> list[dict]:
    """Produce one record per episode in long format."""
    out: list[dict] = []
    scenarios = ("MATE-2v4-9", "MATE-4v4-9", "MATE-4v8-9", "MATE-8v8-9")
    for (method, scenario), paths in discovered.items():
        # If the registered scenario is "*", the CSV itself has a scenario
        # column we need to demultiplex.
        if scenario == "*":
            for p in paths:
                with p.open() as f:
                    for r in csv.DictReader(f):
                        s = r.get("scenario", "?")
                        try:
                            cov = float(r.get("mean_coverage_rate", "nan"))
                            rcov = float(r.get("mean_real_coverage_rate", "nan"))
                            tr = float(r.get("mean_mean_transport_rate", "nan"))
                            er = float(r.get("episode_return", "nan"))
                            seed = int(r.get("seed", "0"))
                            ep = int(r.get("episode", "0"))
                        except ValueError:
                            continue
                        out.append({
                            "method": method, "scenario": s, "seed": seed,
                            "episode": ep, "coverage_rate": cov,
                            "real_coverage_rate": rcov, "transport_rate": tr,
                            "episode_return": er, "source": str(p),
                        })
        else:
            for p in paths:
                rows = load_csv_rows(p)
                for r in rows:
                    try:
                        cov = float(r.get("mean_coverage_rate", "nan"))
                        rcov = float(r.get("mean_real_coverage_rate", "nan"))
                        tr = float(r.get("mean_mean_transport_rate", "nan"))
                        er = float(r.get("episode_return", "nan"))
                        seed = int(r.get("seed", "0"))
                        ep = int(r.get("episode", "0"))
                    except ValueError:
                        continue
                    out.append({
                        "method": method, "scenario": scenario, "seed": seed,
                        "episode": ep, "coverage_rate": cov,
                        "real_coverage_rate": rcov, "transport_rate": tr,
                        "episode_return": er, "source": str(p),
                    })
    return out


def wide_table(long_rows: list[dict], metric: str = "coverage_rate") -> list[dict]:
    """Aggregate long-format rows into (method, scenario) → mean ± SE."""
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in long_rows:
        v = r.get(metric)
        if v is None or v != v:  # nan check
            continue
        grouped[(r["method"], r["scenario"])].append(float(v))

    out: list[dict] = []
    for (method, scenario), values in sorted(grouped.items()):
        arr = np.array(values)
        n = arr.size
        if n == 0:
            continue
        out.append({
            "method": method,
            "scenario": scenario,
            "metric": metric,
            "n": n,
            "mean": float(arr.mean()),
            "se": float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            "median": float(np.median(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
        })
    return out


def collapse_seeds(method: str) -> str:
    """Map per-seed method names to their base method for cross-seed aggregation.

    Examples:
      - DQN_seed0 / DQN_seed1 / DQN_seed2          → DQN
      - DQN_seed0_transfer / ..._seed2_transfer    → DQN_transfer
      - rule_greedy_naiveMH                        → rule_greedy_naiveMH (unchanged)
    """
    return re.sub(r"_seed\d+(?=$|_transfer$)", "", method)


def cross_seed_table(long_rows: list[dict], metric: str = "coverage_rate") -> list[dict]:
    """Same as wide_table but with seeds collapsed (e.g. DQN combines seeds 0,1,2)."""
    out_rows = [
        {**r, "method": collapse_seeds(r["method"])} for r in long_rows
    ]
    return wide_table(out_rows, metric=metric)


def pairwise_significance(
    long_rows: list[dict],
    *,
    metric: str = "coverage_rate",
    scenario: str = "MATE-4v8-9",
) -> list[ComparisonResult]:
    """All pairwise comparisons among methods on a given scenario."""
    from collections import defaultdict
    samples = defaultdict(list)
    for r in long_rows:
        if r["scenario"] != scenario:
            continue
        v = r.get(metric)
        if v is None or v != v:
            continue
        samples[collapse_seeds(r["method"])].append(float(v))

    sample_arrays = {k: np.array(v) for k, v in samples.items() if v}
    if len(sample_arrays) < 2:
        return []
    rng = np.random.default_rng(0)
    from mate_marl.analysis.stats import pairwise_table
    return pairwise_table(sample_arrays, metric=metric, num_resamples=10_000, rng=rng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-long", default="results/EVAL_MATRIX_long.csv")
    ap.add_argument("--out-wide", default="results/EVAL_MATRIX_wide.csv")
    ap.add_argument("--out-cross-seed", default="results/EVAL_MATRIX_cross_seed.csv")
    ap.add_argument("--out-sig", default="results/EVAL_MATRIX_pairwise.csv")
    ap.add_argument("--scenario-for-sig", default="MATE-4v8-9")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    discovered = discover_csvs(results_dir)
    print(f"Discovered {len(discovered)} (method, scenario) cells across {sum(len(v) for v in discovered.values())} CSV files")

    long_rows = aggregate_long(discovered)
    print(f"Long-format: {len(long_rows)} episode rows")

    # --- LONG csv ---
    if long_rows:
        with Path(args.out_long).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted(long_rows[0]))
            w.writeheader()
            w.writerows(long_rows)
        print(f"  wrote {args.out_long}")

    # --- WIDE csv (per-seed) ---
    wide = wide_table(long_rows)
    with Path(args.out_wide).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "scenario", "metric", "n", "mean", "se", "median", "p25", "p75"])
        w.writeheader()
        w.writerows(wide)
    print(f"  wrote {args.out_wide} ({len(wide)} rows)")

    # --- CROSS-SEED csv ---
    cross = cross_seed_table(long_rows)
    with Path(args.out_cross_seed).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "scenario", "metric", "n", "mean", "se", "median", "p25", "p75"])
        w.writeheader()
        w.writerows(cross)
    print(f"  wrote {args.out_cross_seed} ({len(cross)} rows)")

    # --- PAIRWISE significance ---
    sig = pairwise_significance(long_rows, scenario=args.scenario_for_sig)
    if sig:
        with Path(args.out_sig).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sig[0].as_dict().keys()))
            w.writeheader()
            w.writerows(r.as_dict() for r in sig)
        print(f"  wrote {args.out_sig} ({len(sig)} pairwise comparisons on {args.scenario_for_sig})")

    # --- Friendly summary print ---
    print("\n=== Cross-seed summary (coverage_rate on MATE-4v8-9) ===")
    for r in cross:
        if r["scenario"] != args.scenario_for_sig:
            continue
        print(f"  {r['method']:30s} {r['mean']:.4f} ± {r['se']:.4f}  (n={r['n']})")


if __name__ == "__main__":
    main()
