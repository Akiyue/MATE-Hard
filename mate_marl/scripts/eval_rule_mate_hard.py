"""Run rule-based agents on MATE-Hard scenarios with naive/informed adapter.

This script is the MATE-Hard counterpart of ``mate_marl/scripts/eval.py``,
which runs rule-based agents on PLAIN MATE. Output CSV schema matches the
plain-MATE eval so the two can be concatenated downstream.

Usage:
    uv run python -m mate_marl.scripts.eval_rule_mate_hard \
        --inner greedy --adapter informed \
        --scenarios MATE-4v8-9 \
        --num-episodes 30 --seeds 0 1 2 \
        --max-steps 5000 \
        --out results/baseline_greedy_informed_MATE-4v8-9.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import mate
from mate.agents import (
    GreedyCameraAgent,
    HeuristicCameraAgent,
    NaiveCameraAgent,
    RandomCameraAgent,
)

from mate_marl.scripts.make_env import make_mate_hard_baseline_env
from mate_marl.baselines.mate_hard_adapter import (
    RuleBasedNaiveMateHard,
    RuleBasedInformedMateHard,
)


METRIC_KEYS = (
    "coverage_rate",
    "real_coverage_rate",
    "mean_transport_rate",
    "num_delivered_cargoes",
)

INNER_MAP = {
    "random": RandomCameraAgent,
    "naive": NaiveCameraAgent,
    "greedy": GreedyCameraAgent,
    "heuristic": HeuristicCameraAgent,
}


def run_episode(env, agents, max_steps):
    obs, _info = env.reset()
    mate.group_reset(agents, obs)
    metric_acc = {k: [] for k in METRIC_KEYS}
    ep_return = 0.0
    last_info = None
    for t in range(max_steps):
        actions = mate.group_step(env.unwrapped, agents, obs, last_info)
        obs, reward, term, trunc, infos = env.step(np.asarray(actions))
        last_info = infos if isinstance(infos, list) else None
        ep_return += float(np.asarray(reward).mean()) if not np.isscalar(reward) else float(reward)
        if isinstance(infos, list):
            for d in infos:
                if isinstance(d, dict):
                    for k in METRIC_KEYS:
                        if k in d:
                            metric_acc[k].extend(np.atleast_1d(d[k]).tolist())
        elif isinstance(infos, dict):
            for k in METRIC_KEYS:
                if k in infos:
                    metric_acc[k].extend(np.atleast_1d(infos[k]).tolist())
        done = (np.asarray(term).any() if hasattr(term, "any") else term) or \
               (np.asarray(trunc).any() if hasattr(trunc, "any") else trunc)
        if done:
            break
    return {
        "episode_return": ep_return,
        "episode_length": t + 1,
        **{f"mean_{k}": float(np.mean(metric_acc[k])) if metric_acc[k] else float("nan")
           for k in METRIC_KEYS},
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inner", choices=list(INNER_MAP), required=True,
                   help="Base MATE rule-based agent to wrap")
    p.add_argument("--adapter", choices=("naive", "informed"), required=True,
                   help="MATE-Hard adapter strategy")
    p.add_argument("--scenarios", nargs="+", required=True)
    p.add_argument("--num-episodes", type=int, default=30)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--num-fogs", type=int, default=4)
    p.add_argument("--no-heterogeneous", action="store_true")
    p.add_argument("--no-energy", action="store_true")
    p.add_argument("--no-fog", action="store_true")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    AdapterCls = (
        RuleBasedInformedMateHard if args.adapter == "informed"
        else RuleBasedNaiveMateHard
    )
    InnerCls = INNER_MAP[args.inner]

    rows = []
    for scenario in args.scenarios:
        for seed in args.seeds:
            print(f"=== {scenario} seed={seed} ({args.inner}/{args.adapter}) ===", flush=True)
            env = make_mate_hard_baseline_env(
                mate_config=f"{scenario}.yaml",
                num_fogs=args.num_fogs,
                enable_heterogeneous=not args.no_heterogeneous,
                enable_energy=not args.no_energy,
                enable_fog=not args.no_fog,
                seed=seed,
            )
            agents = AdapterCls(InnerCls(), num_fogs=args.num_fogs).spawn(
                env.unwrapped.num_cameras
            )
            for a in agents:
                a.seed(seed)
            for ep in range(args.num_episodes):
                metrics = run_episode(env, agents, args.max_steps)
                rows.append({
                    "inner": args.inner, "adapter": args.adapter,
                    "scenario": scenario, "seed": seed, "episode": ep,
                    **metrics,
                })
                if (ep + 1) % 10 == 0:
                    cov_so_far = np.mean(
                        [r["mean_coverage_rate"] for r in rows[-(ep + 1):]]
                    )
                    print(f"  ep {ep+1}/{args.num_episodes}: rolling cov={cov_so_far:.3f}", flush=True)
            env.close()

    fieldnames = sorted(set().union(*(r.keys() for r in rows)))
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    cov = np.array([r["mean_coverage_rate"] for r in rows])
    n = len(rows)
    print(f"\n[SUMMARY] {args.inner}/{args.adapter}: n={n}, cov={cov.mean():.3f} ± {cov.std()/n**0.5:.3f}")
    print(f"  written to {out_path}")


if __name__ == "__main__":
    main()
