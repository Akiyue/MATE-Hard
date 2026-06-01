"""Eval a MAPPO checkpoint on MATE scenarios (continuous action, no shaping)."""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import numpy as np
import torch

from mate_marl.scripts.make_env import make_marl_env
from mate_marl.trainers import MAPPO, MAPPOConfig
from mate_marl.nets.encoder import EntitySetEncoderConfig

METRIC_KEYS = ("coverage_rate", "real_coverage_rate", "mean_transport_rate", "num_delivered_cargoes")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--scenarios", nargs="+", required=True)
    p.add_argument("--num-episodes", type=int, default=30)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--embed-dim", type=int, default=96)
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-experts", type=int, default=4)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    matches = sorted(glob.glob(args.checkpoint))
    if not matches:
        raise SystemExit(f"No ckpt: {args.checkpoint}")
    ckpt = matches[-1]
    print(f"loading: {ckpt}", flush=True)
    rows = []
    for scenario in args.scenarios:
        for seed in args.seeds:
            print(f"=== {scenario} seed={seed} ===", flush=True)

            def env_factory(seed=seed):
                return make_marl_env(f"{scenario}.yaml", reward_shaping=False, seed=seed)

            probe = env_factory()
            nC = probe.unwrapped.num_cameras
            A = probe.single_action_space.shape[0]
            probe.close()

            cfg = MAPPOConfig(
                num_envs=1, n_steps=8, batch_size=8, n_epochs=1,
                encoder=EntitySetEncoderConfig(
                    embed_dim=args.embed_dim, num_blocks=args.num_blocks,
                    num_heads=args.num_heads, num_experts=args.num_experts, top_k=args.top_k,
                ),
                device=args.device, seed=seed,
            )
            agent = MAPPO(env_factory, num_cameras=nC, action_dim=A, config=cfg)
            try:
                agent.load(ckpt)
            except Exception as e:
                print(f"[warn] load: {e}", flush=True)
                d = torch.load(ckpt, map_location=agent.device)
                if "policy_state_dict" in d:
                    agent.policy.load_state_dict(d["policy_state_dict"], strict=False)
            agent.policy.eval()

            env = env_factory()
            for ep in range(args.num_episodes):
                obs, _ = env.reset()
                acc = {k: [] for k in METRIC_KEYS}
                ret = 0.0
                el = 0
                for t in range(args.max_steps):
                    obs_t = {k: torch.as_tensor(v, device=agent.device) for k, v in obs.items()}
                    with torch.no_grad():
                        mean = agent.policy.forward_actor(obs_t)
                        # deterministic action = mean (already tanh-bounded)
                        a = mean.cpu().numpy()
                    # Action must be in (num_cameras, action_dim)
                    obs, r, term, trunc, info = env.step(a)
                    ret += float(np.asarray(r).mean()) if not np.isscalar(r) else float(r)
                    el += 1
                    if isinstance(info, dict):
                        for k in METRIC_KEYS:
                            if k in info:
                                acc[k].extend(np.atleast_1d(info[k]).tolist())
                    if (np.asarray(term).any() if hasattr(term, "any") else term) or \
                       (np.asarray(trunc).any() if hasattr(trunc, "any") else trunc):
                        break
                row = {
                    "checkpoint": ckpt, "scenario": scenario, "seed": seed, "episode": ep,
                    "episode_return": ret, "episode_length": el,
                }
                for k in METRIC_KEYS:
                    row[f"mean_{k}"] = float(np.mean(acc[k])) if acc[k] else float("nan")
                rows.append(row)
                if (ep + 1) % 5 == 0:
                    cov = np.mean([r["mean_coverage_rate"] for r in rows[-(ep+1):]])
                    print(f"  ep {ep+1}/{args.num_episodes}: rolling cov={cov:.3f}", flush=True)
            env.close()
            agent.envs.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
        w.writeheader()
        w.writerows(rows)
    cov = np.array([r["mean_coverage_rate"] for r in rows])
    n = len(rows)
    print(f"\n[SUMMARY] {n} episodes\n  coverage      = {cov.mean():.3f} ± {cov.std()/n**.5:.3f}\n  written to {out}")


if __name__ == "__main__":
    main()
