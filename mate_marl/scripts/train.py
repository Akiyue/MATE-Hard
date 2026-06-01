"""Real-training entrypoint for the MATE-Hard MARL flagship.

Usage:
    uv run python -m mate_marl.scripts.train \
        --mate-config MATE-4v8-9.yaml \
        --num-envs 8 --n-steps 512 --total-timesteps 10000000 \
        --device cuda --seed 0

This is a thin CLI around mate_marl.trainers.MAPPO. For ablations, toggle
--no-heterogeneous, --no-energy, --no-fog independently.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mate_marl.scripts.make_env import make_marl_env
from mate_marl.trainers import MAPPO, MAPPOConfig
from mate_marl.nets.encoder import EntitySetEncoderConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mate-config", default="MATE-4v8-9.yaml")
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--total-timesteps", type=int, default=10_000_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--moe-loss-coef", type=float, default=0.01)
    p.add_argument("--target-kl", type=float, default=0.05,
                   help="Early-stop a PPO update when approx KL > 1.5 * this. "
                        "Set to a large value (e.g. 1.0) to effectively disable.")
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--no-reward-norm", action="store_true",
                   help="Disable SB3-VecNormalize-style reward RMS normalization. "
                        "Useful when reward shaping already provides per-step variance.")

    # Encoder.
    p.add_argument("--encoder", choices=("set-moe", "set", "deepset", "mlp"),
                   default="set-moe",
                   help="Architecture: set-moe (flagship, attention+MoE), "
                        "set (attention only), deepset, mlp (flat baseline).")
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--d-ff", type=int, default=256)
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-experts", type=int, default=4)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--no-shared-expert", action="store_true")
    p.add_argument("--no-moe", action="store_true")
    p.add_argument("--no-type-conditioning", action="store_true")
    p.add_argument("--decentralized-critic", action="store_true")

    # Env ablations.
    p.add_argument("--no-heterogeneous", action="store_true")
    p.add_argument("--no-energy", action="store_true")
    p.add_argument("--no-fog", action="store_true")
    p.add_argument("--num-fogs", type=int, default=4)
    p.add_argument("--type-assignment", default="round_robin")
    p.add_argument("--no-reward-shaping", action="store_true",
                   help="Disable AuxiliaryCameraRewards-style coverage shaping. "
                        "Use for the 'vanilla reward' ablation row in the paper.")
    p.add_argument("--reward-shaping", action="store_true",
                   help="Enable per-camera soft_coverage_score shaping. "
                        "Recommended for training; disable for 'vanilla' baseline.")

    # Misc.
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-dir", default="checkpoints/mappo")
    p.add_argument("--tb-log-dir", default=None,
                   help="If set, write TensorBoard scalars here. e.g. tb/run1")
    p.add_argument("--run-name", default=None,
                   help="Subdir under save-dir/tb-log-dir for this run.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    def env_factory():
        return make_marl_env(
            mate_config=args.mate_config,
            type_assignment=args.type_assignment,
            enable_heterogeneous=not args.no_heterogeneous,
            enable_energy=not args.no_energy,
            enable_fog=not args.no_fog,
            num_fogs=args.num_fogs,
            reward_shaping=args.reward_shaping and not args.no_reward_shaping,
            seed=args.seed,
        )

    probe = env_factory()
    num_cameras = probe.unwrapped.num_cameras
    action_dim = probe.single_action_space.shape[0]
    probe.close()

    encoder_cfg = EntitySetEncoderConfig(
        embed_dim=args.embed_dim,
        d_ff=args.d_ff,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        num_experts=args.num_experts,
        top_k=args.top_k,
        use_shared_expert=not args.no_shared_expert,
        moe=not args.no_moe and args.encoder == "set-moe",
        type_conditioning=not args.no_type_conditioning,
    )

    # Build encoder for the chosen architecture.
    encoder = None
    if args.encoder == "set-moe" or args.encoder == "set":
        encoder = None  # MARLActorCritic builds the EntitySetEncoder internally
    elif args.encoder == "deepset":
        from mate_marl.nets.baselines import DeepSetEncoder, DeepSetEncoderConfig
        encoder = DeepSetEncoder(DeepSetEncoderConfig(embed_dim=args.embed_dim))
    elif args.encoder == "mlp":
        from mate_marl.nets.baselines import MLPEncoder, MLPEncoderConfig
        # Probe one env to learn token counts.
        probe2 = env_factory()
        u = probe2.unwrapped
        nT, nO = u.num_targets, u.num_obstacles
        nF = probe2.observation_space.spaces["fog"].shape[1]
        probe2.close()
        encoder = MLPEncoder(MLPEncoderConfig(
            embed_dim=args.embed_dim,
            num_cameras=num_cameras, num_targets=nT,
            num_obstacles=nO, num_fogs=nF,
        ))
    cfg = MAPPOConfig(
        num_envs=args.num_envs,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        entropy_coef=args.entropy_coef,
        moe_loss_coef=args.moe_loss_coef,
        target_kl=args.target_kl,
        vf_coef=args.vf_coef,
        normalize_reward=not args.no_reward_norm,
        lr=args.lr,
        encoder=encoder_cfg,
        centralized_critic=not args.decentralized_critic,
        device=args.device,
        seed=args.seed,
    )

    agent = MAPPO(
        env_factory, num_cameras=num_cameras, action_dim=action_dim,
        config=cfg, encoder=encoder,
    )
    sub = args.run_name or args.mate_config.replace(".yaml", "")
    save_dir = Path(args.save_dir) / sub
    tb_log_dir = Path(args.tb_log_dir) / sub if args.tb_log_dir else None
    agent.fit(
        total_timesteps=args.total_timesteps,
        save_dir=save_dir,
        progress_bar=True,
        tb_log_dir=tb_log_dir,
    )


if __name__ == "__main__":
    main()
