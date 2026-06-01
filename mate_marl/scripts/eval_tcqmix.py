"""Evaluate a trained TC-QMIX checkpoint on MATE scenarios (no reward shaping)."""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import numpy as np
import torch

from mate_marl.scripts.make_env import make_marl_env_discrete
from mate_marl.trainers import TCQMIX, TCQMIXConfig
from mate_marl.nets.encoder import EntitySetEncoderConfig
from mate_marl.wrappers.heterogeneous_cameras import NUM_CAMERA_TYPES


METRIC_KEYS = (
    "coverage_rate",
    "real_coverage_rate",
    "mean_transport_rate",
    "num_delivered_cargoes",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--scenarios", nargs="+", required=True)
    p.add_argument("--num-episodes", type=int, default=30)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--levels", type=int, default=5)
    p.add_argument("--embed-dim", type=int, default=96)
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-experts", type=int, default=4)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--mixer-embed-dim", type=int, default=64)
    p.add_argument("--mixer-hyper-hidden", type=int, default=64)
    p.add_argument("--mixer-hyper-input", default="state+types")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    matches = sorted(glob.glob(args.checkpoint))
    if not matches:
        raise SystemExit(f"No checkpoint found: {args.checkpoint}")
    ckpt_path = matches[-1]
    print(f"loading checkpoint: {ckpt_path}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for scenario in args.scenarios:
        for seed in args.seeds:
            print(f"=== {scenario} seed={seed} ===", flush=True)

            def env_factory(seed=seed):
                return make_marl_env_discrete(
                    mate_config=f"{scenario}.yaml",
                    levels=args.levels,
                    reward_shaping=False,
                    seed=seed,
                )

            probe = env_factory()
            nC = probe.unwrapped.num_cameras
            nA = probe.single_action_space.n
            probe.close()

            cfg = TCQMIXConfig(
                num_envs=1, num_agents=nC, n_actions=nA, num_types=NUM_CAMERA_TYPES,
                buffer_size=10, batch_size=4, learning_starts=10_000,
                encoder=EntitySetEncoderConfig(
                    embed_dim=args.embed_dim, num_blocks=args.num_blocks,
                    num_heads=args.num_heads, num_experts=args.num_experts,
                    top_k=args.top_k,
                ),
                mixer_embed_dim=args.mixer_embed_dim,
                mixer_hyper_hidden=args.mixer_hyper_hidden,
                mixer_hyper_input=args.mixer_hyper_input,
                device=args.device,
            )
            agent = TCQMIX(env_factory, num_cameras=nC, n_actions=nA, config=cfg)
            try:
                agent.load(ckpt_path)
            except Exception as e:
                print(f"[warn] load failed (likely num_agents mismatch for transfer): {e}", flush=True)
                # For transfer to scenarios with a different number of cameras,
                # we just zero-init the mixer; the per-agent Q-net is still
                # loaded. The mixer is unused for greedy ACTION selection.
                ckpt = torch.load(ckpt_path, map_location=agent.device)
                agent.q_net.load_state_dict(ckpt["q_net"], strict=False)
                agent.q_target.load_state_dict(ckpt["q_target"], strict=False)
            agent.q_net.eval()

            env = env_factory()
            for ep in range(args.num_episodes):
                obs, _ = env.reset()
                metric_acc = {k: [] for k in METRIC_KEYS}
                ep_return = 0.0
                ep_len = 0
                for t in range(args.max_steps):
                    obs_t = {}
                    for k, v in obs.items():
                        ten = torch.as_tensor(v, device=agent.device)
                        if v.ndim > 1:
                            ten = ten.reshape((-1,) + v.shape[1:])
                        obs_t[k] = ten
                    with torch.no_grad():
                        q, _ = agent.q_net(obs_t)
                        a = q.argmax(dim=-1).cpu().numpy()
                    obs, reward, term, trunc, info = env.step(a)
                    ep_return += float(np.asarray(reward).mean())
                    ep_len += 1
                    for k in METRIC_KEYS:
                        if isinstance(info, dict) and k in info:
                            metric_acc[k].extend(np.atleast_1d(info[k]).tolist())
                    if (np.asarray(term).any() if hasattr(term, "any") else term) or \
                       (np.asarray(trunc).any() if hasattr(trunc, "any") else trunc):
                        break
                row = {
                    "checkpoint": ckpt_path,
                    "scenario": scenario,
                    "seed": seed,
                    "episode": ep,
                    "episode_return": ep_return,
                    "episode_length": ep_len,
                }
                for k in METRIC_KEYS:
                    row[f"mean_{k}"] = float(np.mean(metric_acc[k])) if metric_acc[k] else float("nan")
                rows.append(row)
                if (ep + 1) % 5 == 0:
                    cov_so_far = np.mean([r["mean_coverage_rate"] for r in rows[-(ep+1):]])
                    print(f"  ep {ep+1}/{args.num_episodes}: rolling cov={cov_so_far:.3f}", flush=True)
            env.close()
            agent.envs.close()

    fieldnames = sorted(set().union(*(r.keys() for r in rows)))
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    cov = np.array([r["mean_coverage_rate"] for r in rows])
    rcov = np.array([r["mean_real_coverage_rate"] for r in rows])
    tr = np.array([r["mean_mean_transport_rate"] for r in rows])
    n = len(rows)
    print(f"\n[SUMMARY] {n} episodes")
    print(f"  coverage      = {cov.mean():.3f} ± {cov.std()/n**.5:.3f}")
    print(f"  real_coverage = {rcov.mean():.3f} ± {rcov.std()/n**.5:.3f}")
    print(f"  transport     = {tr.mean():.3f} ± {tr.std()/n**.5:.3f}")
    print(f"  written to {out_path}")


if __name__ == "__main__":
    main()
