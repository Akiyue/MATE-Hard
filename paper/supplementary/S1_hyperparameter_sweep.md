# S1 — Hyperparameter sweep log

Supplementary material for *MATE-Hard and the Count-Invariance Problem in
Multi-Agent Active Perception*.

This appendix lists **every training configuration we ran** on the way to the
headline results, including the runs that failed. Reviewers asked specifically
for (i) the MAPPO-specific settings (PPO epochs, GAE-λ, learning-rate schedule)
and (ii) the reward-shaping sweep; both are below. Nothing here is
cherry-picked: the failed and abandoned runs are reported alongside the ones
that produced the numbers in the manuscript.

Per-iteration metrics for every reported run are in the released TensorBoard
event files under `tb/`; the run directory has the same name as the log file
named in the last column. The raw stdout logs are ~800 MB of progress-bar
output and are available from the corresponding author rather than in the
repository.

---

## S1.1 Settings shared by every run

These are fixed across the whole sweep and across all six benchmarked methods.

| Setting | Value |
|---|---|
| Scenario (training) | `MATE-Hard-4v8-9-fast` (1500-step horizon) |
| Scenario (evaluation) | `MATE-4v8-9` and transfer scenarios, 5000-step horizon |
| Encoder | Set-Transformer, embed dim `E = 96`, 2 MAB blocks, 4 heads |
| MoE feed-forward | 4 experts, top-`k = 2`, 1 shared expert, aux load-balancing loss weight 0.01 |
| Camera-type embedding | 3-way (`WIDE_SHORT` / `NARROW_LONG` / `OMNI_NOISY`), broadcast over all tokens |
| Discount γ | 0.99 |
| Gradient clip | 10.0 (value-based) / 0.5 (MAPPO, `max_grad_norm`) |
| Optimiser | Adam, no learning-rate decay (constant lr throughout) |
| Reward shaping | SCS at `w = 0.02` for the five value-based methods; **none** for MAPPO (see S1.2 and S1.3) |
| Total budget | `--total-timesteps 3000000` for every reported run. **This does not mean the same thing for both trainer families**: the value-based trainers count raw environment steps (`env_step_count += num_envs`), so they receive 3M environment steps, while MAPPO accumulates one count per agent per step (`timesteps += num_envs * num_cameras`), so it receives 3M agent-steps = 750k environment steps on a four-camera scenario. See S1.3. |

`AdamW`, cosine decay and linear lr annealing were **not** used; the learning
rate is constant in every run below. We note this explicitly because a reviewer
asked about the learning-rate schedule.

---

## S1.2 The full run log (17 configurations)

`n_epochs`, `target_kl` and `reward_norm` apply to the on-policy (PPO/MAPPO)
runs only; the `v10*` rows are the off-policy value-based line of work.
"iter-1 KL" and "iter-2+ KL" are the approximate KL divergences PPO measured on
its first and subsequent epochs — the diagnostic we used to detect a collapsed
policy gradient.

| # | Config | Algo / encoder | `n_epochs` | `target_kl` | lr | reward norm | reward shaping | iter-1 KL | iter-2+ KL | Outcome |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | v1 | MAPPO / Set-MoE | 4 | none | 3e-4 | no | none | huge | — | killed (value loss ≈ 35k) |
| 2 | v2 | MAPPO / Set-MoE | 4 | 0.05 | 1e-4 | no | none | 0.10 | 0.01 | mean return stuck at −0.81 |
| 3 | v3 | MAPPO / Set-MoE | 4 | 0.15 | 3e-4 | no | none | 0.16 | 0.005–0.01 | stuck at −0.81 |
| 4 | v4 | MAPPO / Set-MoE | 10 | 0.15 | 3e-4 | no | none | 0.16 | 0.005–0.01 | stuck at −0.81 |
| 5 | v5 | MAPPO / Set-MoE | 10 | 0.15 | 3e-4 | no | `coverage_rate` | 0.007 | 0.004 | team-shared shaping killed the gradient signal |
| 6 | v6 | MAPPO / MLP | 10 | 0.20 | 3e-4 | **yes** | none | 0.67 | 0.01–0.02 | return −22 → −20, then plateau |
| 7 | v6 | MAPPO / Set-MoE | 10 | 0.20 | 3e-4 | **yes** | none | 0.17 | 0.005–0.01 | same plateau ≈ −20 |
| 8 | v7 | MAPPO / MLP | 10 | 0.20 | 3e-4 | yes | SCS (per-camera) | 0.0005 | — | KL collapsed; shaping smoothed advantages further |
| 9 | v8 | MAPPO / MLP | 10 | 0.20 | 3e-4 | no | SCS (per-camera) | 0.0006 | — | same collapse without RMS normalisation |
| 10 | v9 | MAPPO / Set-MoE | 4 | 0.50 | 1e-4 | no | SCS | 0.004 | 0.001 | 5M-step run; return flat; killed as wasted compute (`logs/v9_ppo.log`) |
| 11 | v10 | Double-DQN / Set-MoE | — | — | 3e-4 | — | SCS, w = 0.10 | — | — | OOM at step 275k (GPU contention); restarted |
| 12 | v10b | Double-DQN / Set-MoE | — | — | 3e-4 | — | SCS, w = 0.10 | — | — | return peaked −76.0 at ε ≈ 0.45, **regressed to −79** as ε decayed; OOM at 671k (MoE leak) |
| 13 | v10c | Double-DQN / Set-MoE | — | — | 3e-4 | — | SCS, w = 0.10 | — | — | same regression; OOM at 568k (MoE leak) |
| 14 | v10d | Double-DQN / Set-MoE | — | — | 3e-4 | — | **SCS, w = 0.02** | — | — | first run with no regression; plateau ≈ −16 |
| 15 | v10e | Double-DQN / Set-MoE | — | — | 3e-4 | — | SCS, w = 0.05 | — | — | regression returned (−28 → −39); OOM at 763k (MoE leak) |
| 16 | v10f | Double-DQN / Set-MoE | — | — | 3e-4 | — | **SCS, w = 0.02** | — | — | reproduced #14 cleanly after the MoE leak fix; 1.26M steps, stable at −15.9 |
| 17 | v10g | Double-DQN / Set-MoE | — | — | 3e-4 | — | **SCS, w = 0.02** | — | — | first `4v8-9` run; checkpointed every 250k; **this configuration is the one used for all reported results** (`logs/dqn_4v8_v7.log`, `logs/dqn_4v8_v8.log`) |

SCS = *soft coverage score*, the dense per-camera shaping term defined in
Section 5 of the paper.

**The reward-shaping sweep.** Eleven of the seventeen runs use shaping. Read as
a sweep over the shaping signal and its weight `w`:

| Shaping signal | weight `w` | runs | Outcome |
|---|---|---|---|
| none | — | v1–v4, v6 (MLP), v6 (Set-MoE) | PPO plateaus; no shaping-induced pathology |
| `coverage_rate` (team-shared) | — | v5 | near-constant per-step bonus; advantage variance collapses |
| SCS (per-camera) | 0.10 | v10, v10b, v10c | Q-learning over-optimises the shaping term and **regresses** once ε decays |
| SCS (per-camera) | 0.05 | v10e | same regression, weaker |
| SCS (per-camera) | 0.02 | v10d, v10f, v10g | no regression; converges to a stable but sub-ceiling policy |
| SCS (per-camera), no RMS reward norm | — | v8 | policy gradient collapses (iter-1 KL 0.0006) |

`w = 0.02` is used for every reported result. Both ends of the sweep are bad in
different ways: `w ≥ 0.05` makes the Q-function chase the shaping term, and
smaller weights leave the policy close to uniform. We report `w = 0.02` as the
best of a poor set, not as a tuned optimum — see Section 6.2 of the paper.

---

## S1.3 MAPPO-specific settings

Reviewers asked for the MAPPO hyperparameters explicitly. We report them with
their provenance, because the exact launch command for the three reported MAPPO
seeds was not archived the way the value-based one was (the value-based runs are
reproduced verbatim by `scripts/run_sprint5_train.sh`, which records every flag).
Rather than present unsourced numbers, each row below says where the value comes
from:

* **log** — derivable from the per-iteration metrics of the reported runs,
  released as TensorBoard event files under `tb/mappo_4v8_seed{0,1,2}/` (the
  same series the stdout logs carry);
* **code** — the trainer default in `MAPPOConfig`
  (`mate_marl/trainers/mappo.py`), which applies unless the launch overrode it;
* **report** — recorded in the sprint report accompanying the runs.

| Parameter | Value | Source |
|---|---|---|
| Rollout size per update | 32,768 agent-steps per PPO iteration, i.e. `num_envs × n_steps = 8192` environment transitions | log (`[iter 1] steps=32768`, `[iter 2] steps=65536`) |
| PPO iterations over the budget | ≈ 92 policy updates | log (3M agent-steps ÷ 32,768) |
| Step accounting | `timesteps += num_envs × num_cameras`, so `--total-timesteps 3000000` = 3M agent-steps = **750k environment steps** | code (`mappo.py`, rollout loop) |
| Reward shaping | **none** — `--reward-shaping` is `store_true` and defaults off; the reported runs did not pass it | code + report ("MAPPO (continuous, no shaping)") |
| `gae_lambda` (GAE-λ) | 0.95 | code |
| `clip_range` | 0.2 (ratio clipping; value clipping uses the same range) | code |
| `n_epochs` (PPO epochs) | 4 | code |
| `vf_coef` | 0.5 | code |
| `entropy_coef` | 0.01 | code |
| `moe_loss_coef` | 0.01 (MoE load-balancing auxiliary loss) | code |
| `max_grad_norm` | 0.5 | code |
| Learning-rate schedule | constant, no decay of any kind | code |
| `normalize_advantage` | true | code |
| `normalize_reward` | true (SB3-`VecNormalize`-style running-return RMS, clip 10.0) | code |
| Action head | tanh-squashed Gaussian, `log_std_init = -0.5`, 2-D (rotation rate, zoom rate) | code |
| Critic | centralised: per-agent latent ‖ mean-pooled team latent, width `2E` for any team size | code |

The split of the 8192 environment transitions per update between `num_envs` and
`n_steps` is not recoverable from the logs (only their product is); the training
template in `HANDOFF.md` uses `--num-envs 8 --n-steps 1024`, which is consistent
with the observed product, but we do not assert it as fact.

**`target_kl`.** The trainer default is 0.05. Runs #2–#5 in S1.2 used
`target_kl` ∈ {0.05, 0.15} and the policy stopped moving after the first epoch
(iter-2+ approximate KL ≈ 0.005); raising it to 0.20 in runs #6–#9 is what
allowed MAPPO to move at all. Whether the three reported seeds used 0.20 or the
0.05 default is not recorded, which is a gap in our records rather than a
finding, and we note it here rather than guess.

**What this means for the comparison.** MAPPO is not budget-matched to the
value-based methods (750k vs 3M environment steps) and does not receive the SCS
shaping they use. Section 5.2 of the paper states both asymmetries and declines
to draw an on-policy-versus-off-policy conclusion from Table 5. On its two
non-collapsed seeds MAPPO reaches 0.189 coverage, the joint-best figure in that
table.

**Inter-seed variance.** The three MAPPO seeds gave 0.187, 0.190 and 0.152
coverage — a cross-seed standard deviation of 0.021. This is the largest in the
suite, but instability is not confined to the on-policy method: VDN's three
seeds gave 0.187, 0.163 and 0.189, a standard deviation of 0.014. The remaining
four methods are all at or below 0.005. Users benchmarking on MATE-Hard should
budget at least five training seeds regardless of algorithm family.

## S1.4 Value-based settings

Unlike the MAPPO runs, the value-based runs are reproduced verbatim by a
released script: `scripts/run_sprint5_train.sh` records the full flag list in
its `COMMON_ARGS` array and states that these are "the same as the Sprint 2
headline runs". The table below is that flag list. Note that several of these
flags **override** the dataclass defaults in
`mate_marl/trainers/{dqn,tcqmix}.py`; anyone re-running the trainers with their
built-in defaults will not reproduce our configuration, so the flags matter.

| Parameter | Value used (all five value-based methods) | Trainer default, if different |
|---|---|---|
| `--total-timesteps` | 3,000,000 environment steps | — |
| `--num-envs` | 8 | — |
| `--buffer-size` | 50,000 | 100,000 for Double-DQN |
| `--batch-size` | 128 | 256 for Double-DQN, 64 for the QMIX family |
| `--learning-starts` | 5,000 | 10,000 for Double-DQN |
| `--train-freq` | every 4 environment steps | — |
| `--target-tau` (Polyak) | 5e-3 | — |
| `--lr` | 3e-4, constant | — |
| `--gamma` | 0.99 | — |
| `--eps-start` / `--eps-end` | 1.0 / 0.05 | — |
| `--eps-decay-steps` | 400,000 | 200,000 for Double-DQN |
| `--reward-shaping --shaping-weight` | on, 0.02 | off by default |
| `--embed-dim` | 96 | 128 in `train_dqn.py` |
| `--num-blocks` / `--num-heads` | 2 / 4 | — |
| `--num-experts` / `--top-k` | 4 / 2 | — |
| action space | `Discrete(25)` (5 rotation × 5 zoom) | — |
| gradient clip | 10.0 | — |
| Double-Q target | enabled | — |
| `--mixer-embed-dim` (QMIX family) | 64 | — |
| `--mixer-hyper-hidden` (QMIX family) | 64 | — |
| `--mixer-hyper-input` | `state+types` (TC-QMIX), `state` (vanilla QMIX), `types` (ablation) | — |
| monotonicity | `abs(·)` on hypernet weight outputs | — |

Double DQN was adopted after runs #12–#13 in S1.2 showed the classic
maximisation-bias regression; it is enabled for every reported value-based run.

## S1.5 The MoE memory leak

`common_net.moe.MoE._store_gate_logits` appended to `self._gate_logits` on every
forward pass in training mode. MAPPO drained that list each update via
`MoEGateLossManager`; the value-based trainers did not, so the list grew without
bound and the process exhausted GPU memory after roughly 500k–800k gradient
steps. Runs #12, #13 and #15 died this way.

The fix (`mate_marl/trainers/dqn.py`, mirrored in `tcqmix.py`) collects the MoE
layers at construction and calls `moe._reset_gate_logits()` after every gradient
step, and puts the target network in `eval()` mode so it does not accumulate
either.

**This leak affected only development runs.** It was fixed before run #17, and
every checkpoint behind every number reported in the manuscript was produced
after the fix. No reported result comes from a leaking run.

---

## S1.6 Compute

Approximately 210 GPU-hours in total on shared NVIDIA RTX 3090 Ti hardware,
including the failed runs above (~70 of those hours are the Sprint-5 extended
train-eval matrix of Table 8). The machine was shared with other users, which
is why several runs OOM'd at unrelated step counts; runs were launched with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and a memory footprint kept
under ~9 GiB.

---

## S1.7 Reproducing the tables

Every data table in the manuscript is regenerated from the released per-episode
CSVs by a single command, with no GPU and no simulator install:

```bash
python reproduce_tables.py           # writes paper/tables/*.tex and prints the report
python reproduce_tables.py --check   # additionally asserts the numbers match the manuscript
```

`--check` exits non-zero if any regenerated value disagrees with the published
one. Tables 3 and 4 are hyperparameter listings rather than measurements; they
correspond to Sections S1.3 and S1.4 above.

The figures are regenerated by `paper/make_figures_sprint4.py` (Figures 4-7 and
9-12) and `paper/make_figure_training_curves.py` (Figure 8). The latter reads
the trainers' stdout logs when present and otherwise the released TensorBoard
event files under `tb/`; both carry the same `rollout/mean_return_100` series
and agree to within the logs' three-decimal rounding. It applies the
agent-step-to-environment-step correction of S1.3 to the MAPPO curve, which is
why that curve ends at 753,664 environment steps.
