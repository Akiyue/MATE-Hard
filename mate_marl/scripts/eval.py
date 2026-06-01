"""Evaluation harness.

Loads a trained MAPPO checkpoint, rolls out N episodes per scenario, and
aggregates the canonical MATE metrics into a CSV row per (checkpoint,
scenario, seed).

Tracked metrics (already produced by MATE in info[i]):
  - coverage_rate
  - real_coverage_rate
  - mean_transport_rate
  - num_delivered_cargoes
Plus episode-level: episode_return, episode_length, mean_energy_remaining.

Usage:
    uv run python -m mate_marl.scripts.eval \
        --checkpoint checkpoints/.../mappo_xxx.pt \
        --scenarios MATE-4v8-9 MATE-2v4-9 MATE-4v4-9 MATE-8v8-9 \
        --num-episodes 50 --seeds 0 1 2 3 4 \
        --out results/eval.csv

Also supports rule-based baselines via --baseline {random,naive,greedy,heuristic}
(no checkpoint required).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import mate
from mate.agents import (
    GreedyCameraAgent,
    HeuristicCameraAgent,
    NaiveCameraAgent,
    RandomCameraAgent,
)

from mate_marl.scripts.make_env import make_marl_env
from mate_marl.trainers import MAPPO, MAPPOConfig
from mate_marl.nets.encoder import EntitySetEncoderConfig


METRIC_KEYS = (
    "coverage_rate",
    "real_coverage_rate",
    "mean_transport_rate",
    "num_delivered_cargoes",
)


# ---------- baselines ----------


_BASELINE_MAP = {
    "random": RandomCameraAgent,
    "naive": NaiveCameraAgent,
    "greedy": GreedyCameraAgent,
    "heuristic": HeuristicCameraAgent,
}


def run_baseline_episode(
    env,
    agents: list,
    max_steps: int,
) -> dict[str, float]:
    """Run one episode of a rule-based camera agent set."""
    obs, _ = env.reset()
    mate.group_reset(agents, obs)

    ep_return = 0.0
    last_info: list[dict] = []
    metric_acc = {k: [] for k in METRIC_KEYS}
    energy_acc: list[float] = []
    for t in range(max_steps):
        action = mate.group_step(env.unwrapped, agents, obs, last_info if last_info else None)
        # MATE expects np.ndarray of shape (num_cameras, 2).
        action = np.asarray(action)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_return += float(np.asarray(reward).mean())
        last_info = info if isinstance(info, list) else []
        for d in last_info:
            for k in METRIC_KEYS:
                if k in d:
                    metric_acc[k].append(float(d[k]))
            if "energy" in d:
                energy_acc.append(float(d["energy"]))
        if (
            (np.asarray(terminated).any() if hasattr(terminated, "any") else terminated)
            or (np.asarray(truncated).any() if hasattr(truncated, "any") else truncated)
        ):
            break

    out = {f"mean_{k}": float(np.mean(v)) if v else float("nan") for k, v in metric_acc.items()}
    out["episode_return"] = ep_return
    out["episode_length"] = t + 1
    out["mean_energy_remaining"] = float(np.mean(energy_acc)) if energy_acc else float("nan")
    return out


def make_eval_env(scenario: str, seed: Optional[int], **wrapper_kwargs):
    """Wrap with the same flagship stack used at training time."""
    return make_marl_env(mate_config=f"{scenario}.yaml", seed=seed, **wrapper_kwargs)


def make_baseline_env(scenario: str, seed: Optional[int]):
    """Build a MATE env wrapped with MultiCamera (no MARL extensions). Used by
    the rule-based baselines that ship with MATE."""
    import gymnasium as gym

    from mate.agents import GreedyTargetAgent

    base = gym.make("MultiAgentTracking-v0", config=f"{scenario}.yaml")
    return mate.MultiCamera.make(base, target_agent=GreedyTargetAgent())


# ---------- learned policy ----------


def run_learned_episode(
    agent: MAPPO,
    env,
    max_steps: int,
    deterministic: bool = True,
) -> dict[str, float]:
    obs, _ = env.reset()
    nC = env.unwrapped.num_cameras

    ep_return = 0.0
    metric_acc = {k: [] for k in METRIC_KEYS}
    energy_acc: list[float] = []
    last_info_list: list[dict] = []

    for t in range(max_steps):
        # Treat (num_cameras, ...) as the batch dim for the shared policy.
        obs_t = {k: torch.as_tensor(v, device=agent.device) for k, v in obs.items()}
        with torch.no_grad():
            mean, _ = agent.policy(obs_t)
            if deterministic:
                action_t = mean
            else:
                std = agent.policy.log_std.exp().expand_as(mean)
                action_t = torch.distributions.Normal(mean, std).sample()
        action = action_t.cpu().numpy()
        obs, reward, term, trunc, info = env.step(action)
        ep_return += float(np.asarray(reward).mean())
        # info is a dict (FlattenAgentsForPPO aggregates it).
        if isinstance(info, dict):
            for k in METRIC_KEYS:
                if k in info:
                    v = info[k]
                    metric_acc[k].extend(np.atleast_1d(v).tolist())
            if "energy" in info:
                energy_acc.extend(np.atleast_1d(info["energy"]).tolist())
        if (
            (np.asarray(term).any() if hasattr(term, "any") else term)
            or (np.asarray(trunc).any() if hasattr(trunc, "any") else trunc)
        ):
            break

    out = {f"mean_{k}": float(np.mean(v)) if v else float("nan") for k, v in metric_acc.items()}
    out["episode_return"] = ep_return
    out["episode_length"] = t + 1
    out["mean_energy_remaining"] = float(np.mean(energy_acc)) if energy_acc else float("nan")
    return out


def load_agent(checkpoint: Path, env_factory, device: str) -> MAPPO:
    probe = env_factory()
    nC = probe.unwrapped.num_cameras
    A = probe.single_action_space.shape[0]
    probe.close()

    # Use same encoder config as training default.
    cfg = MAPPOConfig(
        num_envs=1,  # we don't use the trainer's vec_env here
        n_steps=1,
        batch_size=1,
        device=device,
    )
    agent = MAPPO(env_factory, num_cameras=nC, action_dim=A, config=cfg)
    agent.load(checkpoint)
    return agent


# ---------- main ----------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None,
                   help="path to .pt checkpoint produced by MAPPO.fit")
    p.add_argument("--baseline", type=str, default=None,
                   choices=list(_BASELINE_MAP.keys()))
    p.add_argument("--scenarios", nargs="+", required=True,
                   help="scenario names without .yaml, e.g. MATE-4v8-9")
    p.add_argument("--num-episodes", type=int, default=50)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--max-steps", type=int, default=10_000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)

    # MARL-extension toggles (must match training).
    p.add_argument("--no-heterogeneous", action="store_true")
    p.add_argument("--no-energy", action="store_true")
    p.add_argument("--no-fog", action="store_true")
    p.add_argument("--num-fogs", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if (args.checkpoint is None) == (args.baseline is None):
        raise SystemExit("Provide exactly one of --checkpoint or --baseline.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for scenario in args.scenarios:
        for seed in args.seeds:
            print(f"=== {scenario} seed={seed} ===", flush=True)

            if args.checkpoint is not None:
                # Learned policy uses the full flagship stack.
                def env_factory(seed=seed):
                    return make_eval_env(
                        scenario, seed=seed,
                        enable_heterogeneous=not args.no_heterogeneous,
                        enable_energy=not args.no_energy,
                        enable_fog=not args.no_fog,
                        num_fogs=args.num_fogs,
                    )
                agent = load_agent(Path(args.checkpoint), env_factory, args.device)
                agent.policy.eval()
                env = env_factory()
                for ep in range(args.num_episodes):
                    metrics = run_learned_episode(agent, env, args.max_steps)
                    rows.append({
                        "checkpoint": args.checkpoint, "baseline": "",
                        "scenario": scenario, "seed": seed, "episode": ep,
                        **metrics,
                    })
                env.close()
                agent.envs.close()
            else:
                # Rule-based baseline uses raw MATE (no extensions) — extensions
                # don't apply because the built-in agents don't observe them.
                env = make_baseline_env(scenario, seed=seed)
                BaselineCls = _BASELINE_MAP[args.baseline]
                num_cameras = env.unwrapped.num_cameras
                agents = BaselineCls().spawn(num_cameras)
                for ep in range(args.num_episodes):
                    metrics = run_baseline_episode(env, agents, args.max_steps)
                    rows.append({
                        "checkpoint": "", "baseline": args.baseline,
                        "scenario": scenario, "seed": seed, "episode": ep,
                        **metrics,
                    })
                env.close()

    fieldnames = sorted(set().union(*(r.keys() for r in rows)))
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
