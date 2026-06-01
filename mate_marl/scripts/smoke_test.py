"""Minimal CPU smoke test for the MARL flagship stack.

Exercises:
  1. Env stack construction (Heterogeneous + Energy + Fog + DictObs + Flatten).
  2. Encoder forward pass with mixed-type entity tokens.
  3. MAPPO rollout collection (n_steps=8) + one PPO update (n_epochs=1).
  4. Save → load round trip.

Runs in 1-2 minutes on CPU. Expected: no crashes, value/policy losses finite.

Usage:
    uv run python -m mate_marl.scripts.smoke_test
"""

from __future__ import annotations

import sys
import traceback
import tempfile
from pathlib import Path

import numpy as np
import torch

from mate_marl.scripts.make_env import make_marl_env
from mate_marl.trainers import MAPPO, MAPPOConfig
from mate_marl.nets.encoder import EntitySetEncoderConfig


def step1_env_stack() -> None:
    print("[1/4] Building env stack...", flush=True)
    env = make_marl_env(mate_config="MATE-4v2-9.yaml", seed=0)
    obs, info = env.reset(seed=0)
    print("  obs keys:", sorted(obs.keys()))
    for k, v in obs.items():
        print(f"    {k:10s} : shape={v.shape} dtype={v.dtype}")
    print("  action_space:", env.action_space)
    print("  num_cameras:", env.unwrapped.num_cameras)

    # Take 5 random steps.
    for t in range(5):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        if term.any() or trunc.any():
            obs, info = env.reset()
    print("  ✓ 5 random steps OK")
    env.close()


def step2_encoder_forward() -> None:
    print("[2/4] Encoder forward pass...", flush=True)
    from mate_marl.nets import MARLActorCritic, MARLActorCriticConfig

    env = make_marl_env(mate_config="MATE-4v2-9.yaml", seed=0)
    obs, _ = env.reset(seed=0)
    nC = env.unwrapped.num_cameras

    cfg = MARLActorCriticConfig(
        encoder=EntitySetEncoderConfig(
            embed_dim=32, d_ff=64, num_blocks=1, num_heads=2,
            num_experts=2, top_k=1, use_shared_expert=False,
        ),
        action_dim=2,
        centralized_critic=True,
    )
    net = MARLActorCritic(num_cameras=nC, config=cfg)

    # Convert dict np arrays → torch and treat (num_cameras,) as batch dim.
    obs_t = {k: torch.as_tensor(v) for k, v in obs.items()}
    mean, value = net(obs_t)
    print(f"  mean shape: {tuple(mean.shape)}, value shape: {tuple(value.shape)}")
    assert mean.shape == (nC, 2), f"unexpected mean shape {mean.shape}"
    assert value.shape == (nC,), f"unexpected value shape {value.shape}"
    print("  ✓ forward pass OK")
    env.close()


def step3_mappo_one_update() -> None:
    print("[3/4] MAPPO collect + update (n_steps=8)...", flush=True)

    def env_factory():
        return make_marl_env(mate_config="MATE-4v2-9.yaml", seed=None)

    # Probe num_cameras + action_dim.
    probe = env_factory()
    nC = probe.unwrapped.num_cameras
    A = probe.single_action_space.shape[0]
    probe.close()

    cfg = MAPPOConfig(
        num_envs=2,
        n_steps=8,
        batch_size=16,
        n_epochs=1,
        encoder=EntitySetEncoderConfig(
            embed_dim=32, d_ff=64, num_blocks=1, num_heads=2,
            num_experts=2, top_k=1, use_shared_expert=False,
        ),
        device="cpu",
        seed=0,
    )
    agent = MAPPO(env_factory, num_cameras=nC, action_dim=A, config=cfg)
    agent.collect_rollout()
    stats = agent.update()
    print(f"  stats: {stats}")
    assert all(np.isfinite(v) for v in stats.values()), "non-finite loss"
    print("  ✓ rollout + update OK")
    agent.envs.close()


def step4_save_load() -> None:
    print("[4/4] Save / load round trip...", flush=True)

    def env_factory():
        return make_marl_env(mate_config="MATE-4v2-9.yaml", seed=None)

    probe = env_factory()
    nC = probe.unwrapped.num_cameras
    A = probe.single_action_space.shape[0]
    probe.close()

    cfg = MAPPOConfig(
        num_envs=2, n_steps=4, batch_size=8, n_epochs=1,
        encoder=EntitySetEncoderConfig(
            embed_dim=16, d_ff=32, num_blocks=1, num_heads=2,
            num_experts=2, top_k=1, use_shared_expert=False,
        ),
        device="cpu", seed=0,
    )
    agent = MAPPO(env_factory, num_cameras=nC, action_dim=A, config=cfg)
    agent.collect_rollout()
    agent.update()
    with tempfile.TemporaryDirectory() as d:
        path = agent.save(d)
        agent2 = MAPPO(env_factory, num_cameras=nC, action_dim=A, config=cfg)
        agent2.load(path)
        # Compare a few parameter tensors.
        for (n1, p1), (n2, p2) in zip(
            agent.policy.named_parameters(), agent2.policy.named_parameters()
        ):
            assert torch.allclose(p1, p2), f"param mismatch at {n1}"
    print("  ✓ save/load OK")
    agent.envs.close()
    agent2.envs.close()


def main() -> int:
    steps = [step1_env_stack, step2_encoder_forward, step3_mappo_one_update, step4_save_load]
    for step in steps:
        try:
            step()
        except Exception:
            traceback.print_exc()
            return 1
    print("\nAll smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
