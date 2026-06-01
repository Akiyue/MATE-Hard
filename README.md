# MATE-Hard

A strict superset of the **Multi-Agent Tracking Environment (MATE)** that adds three orthogonal sources of difficulty via composable Gym wrappers:

1. **Heterogeneous cameras** — three camera types (wide / tele / fisheye) with distinct field-of-view, range, slew speed, and observation noise.
2. **Energy budgets** — every camera carries a finite battery that drains with slew/zoom action; cameras must opportunistically visit charging stations.
3. **Dynamic fog** — drifting Gaussian occluder cloud blocks otherwise-valid lines of sight.

Disabling all three wrappers recovers vanilla MATE, so prior results remain reproducible.

The repository also ships a complete reference framework: a Set-Transformer + Type-Conditioned Mixture-of-Experts encoder, six cooperative-MARL trainers (Double-DQN, MAPPO, VDN, vanilla QMIX, TC-QMIX, TC-QMIX types-only), a unified evaluation matrix script, and statistical-significance utilities (paired bootstrap, Welch's *t*-test, Cohen's *d*).

## Installation

Requires Python 3.13 and [uv](https://github.com/astral-sh/uv).

```bash
pip install uv          # or: pip3 install uv
uv sync                 # install / sync dependencies from uv.lock
source .venv/bin/activate
```

## How MATE-Hard is constructed

MATE-Hard is **not** a separate environment or a new YAML — it is a runtime
stack of three Gym wrappers applied on top of any plain MATE scenario. The
three wrappers live in `mate_marl/wrappers/`:

- `mate_marl.wrappers.HeterogeneousCameras` — assigns each camera a type (wide / tele / fisheye) with distinct FoV, range, slew speed, and noise.
- `mate_marl.wrappers.EnergyConstraint` — adds a per-camera battery that drains with slew/zoom; cameras recharge inside warehouse circles.
- `mate_marl.wrappers.DynamicFog` — spawns drifting Gaussian fog patches that occlude target observations.

The factory `mate_marl.scripts.make_env.make_marl_env` composes the full stack
on top of MATE's `MultiCamera` adapter:

```python
from mate_marl.scripts.make_env import make_marl_env

# Full MATE-Hard stack on the headline 4v8 scenario:
env = make_marl_env(
    mate_config="MATE-4v8-9.yaml",
    enable_heterogeneous=True,
    enable_energy=True,
    enable_fog=True,
)

# Plain MATE (all three wrappers disabled — exactly reproduces vanilla MATE):
env = make_marl_env(
    mate_config="MATE-4v8-9.yaml",
    enable_heterogeneous=False,
    enable_energy=False,
    enable_fog=False,
)
```

The full wrapper stack assembled by `make_marl_env` (outermost first) is:

```
FlattenAgentsForPPO
  MateMARLDictObs                       # pack flat obs into typed entity-token dict
    DynamicFog                          # MATE-Hard wrapper (3)
      EnergyConstraint                  # MATE-Hard wrapper (2)
        HeterogeneousCameras            # MATE-Hard wrapper (1)
          mate.MultiCamera              # MATE single-team adapter
            MultiAgentTracking-v0       # MATE base env
```

## Quick start

Run a sample MATE-Hard episode with pygame rendering:

```bash
uv run python example.py
```

Train Double-DQN on `MATE-Hard-4v8-9` (3 seeds, 3M steps):

```bash
uv run python -m mate_marl.scripts.train_dqn \
    --mate-config MATE-4v8-9.yaml \
    --seed 0 --run-name dqn_4v8_seed0 \
    --save-dir checkpoints/dqn \
    --reward-shaping --shaping-weight 0.02 \
    --num-envs 8 --buffer-size 50000 \
    --batch-size 128 --learning-starts 5000 \
    --train-freq 4 --target-tau 5e-3 \
    --total-timesteps 3000000 \
    --lr 3e-4 --gamma 0.99 \
    --eps-start 1.0 --eps-end 0.05 --eps-decay-steps 400000 \
    --embed-dim 96 --num-blocks 2 --num-heads 4 \
    --num-experts 4 --top-k 2
```

Train TC-QMIX (type-conditioned mixer):

```bash
uv run python -m mate_marl.scripts.train_tcqmix \
    --mate-config MATE-4v8-9.yaml \
    --seed 0 --run-name tcqmix_4v8_seed0 \
    --save-dir checkpoints/tcqmix \
    --mixer-hyper-input state+types \
    --reward-shaping --shaping-weight 0.02 \
    --total-timesteps 3000000
```

Evaluate a checkpoint on multiple scenarios (zero-shot transfer matrix):

```bash
uv run python -m mate_marl.scripts.eval_dqn \
    --checkpoint checkpoints/dqn/dqn_4v8_seed0/dqn_<TS>.pt \
    --scenarios MATE-2v4-9 MATE-4v4-9 MATE-4v8-9 MATE-8v8-9 \
    --seeds 0 1 2 --num-episodes 30 \
    --out results/dqn_seed0_transfer.csv
```

Driver scripts that schedule full multi-seed sweeps across two GPUs live in `scripts/`.

## Repository layout

```
mate/             # MATE environment + MATE-Hard wrappers, agents, asset YAMLs
mate_marl/        # Trainers, evaluators, baselines, network blocks, wrappers
gym_agent/        # Standalone RL framework (PPO / A2C / DQN, vec-envs, buffers)
common_net/       # Reusable PyTorch building blocks (attention, MoE, MLPs)
scripts/          # Multi-seed training / transfer-eval driver scripts
example.py        # Sample MATE-Hard episode with pygame rendering
wrapper.py        # Dict-observation wrapper demo
```

### Key modules

- `mate_marl.wrappers.HeterogeneousCameras`, `mate_marl.wrappers.EnergyConstraint`, `mate_marl.wrappers.DynamicFog` — the three MATE-Hard wrappers (see *How MATE-Hard is constructed* above).
- `mate_marl.scripts.make_env.make_marl_env` — factory that composes the full MATE-Hard wrapper stack; per-wrapper toggles enable ablations.
- `mate_marl.nets` — Set-Transformer + Type-Conditioned MoE encoder.
- `mate_marl.trainers` — six trainers (Double-DQN, MAPPO, VDN, vanilla QMIX, TC-QMIX, TC-QMIX types-only); TC-QMIX uses a hypernet-based type-conditioned mixer.
- `mate_marl.scripts.eval_matrix` — unified evaluation harness used to build the transfer matrix.
- `mate_marl.analysis` — statistical-significance utilities (paired bootstrap CI, Welch's *t*-test, Cohen's *d*).

## Scenarios

Built-in scenario YAMLs (in `mate/assets/`):

| Scenario        | Cameras | Targets | Notes |
|-----------------|---------|---------|-------|
| `MATE-2v4-9`    | 2       | 4       | Sparse cameras |
| `MATE-4v4-9`    | 4       | 4       | Balanced |
| `MATE-4v8-9`    | 4       | 8       | Default training scenario |
| `MATE-8v8-9`    | 8       | 8       | Dense cameras |
| `MATE-Navigation` | varies | 0     | Pure navigation |

The MATE-Hard wrappers are scenario-agnostic.

## License

Code in this repository is released for academic use; see the upcoming preprint for citation information.
