"""DQN-MARL training entrypoint (param-shared DQN over discretized actions)."""

from __future__ import annotations

import argparse
from pathlib import Path

from mate_marl.scripts.make_env import make_marl_env_discrete
from mate_marl.trainers import DQN, DQNConfig
from mate_marl.nets.encoder import EntitySetEncoderConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mate-config", default="MATE-4v8-9.yaml")
    p.add_argument("--levels", type=int, default=5)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--total-timesteps", type=int, default=5_000_000)

    # DQN.
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-starts", type=int, default=10_000)
    p.add_argument("--train-freq", type=int, default=4)
    p.add_argument("--target-tau", type=float, default=5e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-grad-norm", type=float, default=10.0)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay-steps", type=int, default=200_000)
    p.add_argument("--no-double-dqn", action="store_true",
                   help="Disable Double DQN (use vanilla DQN). "
                        "Default behaviour uses Double DQN.")

    # Encoder.
    p.add_argument("--encoder", choices=("set-moe", "set", "deepset", "mlp"), default="set-moe")
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--d-ff", type=int, default=256)
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-experts", type=int, default=4)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--no-shared-expert", action="store_true")
    p.add_argument("--no-moe", action="store_true")
    p.add_argument("--no-type-conditioning", action="store_true")
    p.add_argument("--head-hidden", type=int, default=128)

    # Env ablations.
    p.add_argument("--no-heterogeneous", action="store_true")
    p.add_argument("--no-energy", action="store_true")
    p.add_argument("--no-fog", action="store_true")
    p.add_argument("--num-fogs", type=int, default=4)
    p.add_argument("--reward-shaping", action="store_true")
    p.add_argument("--shaping-weight", type=float, default=None,
                   help="Override soft_coverage_score weight in reward shaping. "
                        "Default 0.02 (in FlattenAgentsForPPO.DEFAULT_REWARD_WEIGHTS).")

    # IO.
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-dir", default="checkpoints/dqn")
    p.add_argument("--tb-log-dir", default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--log-interval", type=int, default=2000)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    def env_factory():
        return make_marl_env_discrete(
            mate_config=args.mate_config, levels=args.levels,
            enable_heterogeneous=not args.no_heterogeneous,
            enable_energy=not args.no_energy,
            enable_fog=not args.no_fog,
            num_fogs=args.num_fogs,
            reward_shaping=args.reward_shaping,
            shaping_weight=args.shaping_weight,
            seed=args.seed,
        )

    probe = env_factory()
    nC = probe.unwrapped.num_cameras
    nA = probe.single_action_space.n
    probe.close()

    encoder_cfg = EntitySetEncoderConfig(
        embed_dim=args.embed_dim, d_ff=args.d_ff, num_blocks=args.num_blocks,
        num_heads=args.num_heads, num_experts=args.num_experts, top_k=args.top_k,
        use_shared_expert=not args.no_shared_expert,
        moe=not args.no_moe and args.encoder == "set-moe",
        type_conditioning=not args.no_type_conditioning,
    )
    cfg = DQNConfig(
        num_envs=args.num_envs, n_actions=nA,
        buffer_size=args.buffer_size, batch_size=args.batch_size,
        learning_starts=args.learning_starts, train_freq=args.train_freq,
        target_update_tau=args.target_tau, gamma=args.gamma, lr=args.lr,
        max_grad_norm=args.max_grad_norm,
        eps_start=args.eps_start, eps_end=args.eps_end,
        eps_decay_steps=args.eps_decay_steps,
        double_dqn=not args.no_double_dqn,
        encoder=encoder_cfg, head_hidden=args.head_hidden,
        device=args.device, seed=args.seed,
    )

    encoder = None
    if args.encoder in ("set-moe", "set"):
        encoder = None
    elif args.encoder == "deepset":
        from mate_marl.nets.baselines import DeepSetEncoder, DeepSetEncoderConfig
        encoder = DeepSetEncoder(DeepSetEncoderConfig(embed_dim=args.embed_dim))
    elif args.encoder == "mlp":
        from mate_marl.nets.baselines import MLPEncoder, MLPEncoderConfig
        probe2 = env_factory()
        u = probe2.unwrapped
        nT, nO = u.num_targets, u.num_obstacles
        nF = probe2.observation_space.spaces["fog"].shape[1]
        probe2.close()
        encoder = MLPEncoder(MLPEncoderConfig(
            embed_dim=args.embed_dim,
            num_cameras=nC, num_targets=nT, num_obstacles=nO, num_fogs=nF,
        ))

    agent = DQN(env_factory, num_cameras=nC, n_actions=nA, config=cfg, encoder=encoder)

    sub = args.run_name or args.mate_config.replace(".yaml", "")
    save_dir = Path(args.save_dir) / sub
    tb_log_dir = Path(args.tb_log_dir) / sub if args.tb_log_dir else None
    agent.fit(
        total_timesteps=args.total_timesteps,
        log_interval=args.log_interval,
        save_dir=save_dir,
        progress_bar=True,
        tb_log_dir=tb_log_dir,
    )


if __name__ == "__main__":
    main()
