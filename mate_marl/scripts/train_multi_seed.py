"""Multi-seed training driver.

Runs an existing single-seed training script (train_dqn / train_tcqmix /
train_vdn / train) once per seed, in sequence on the same GPU, writing
checkpoints to per-seed subdirectories.

Why a *separate script* and not a flag on the existing scripts:

  - The existing per-method trainers are debugged single-process loops.
    Adding per-seed multi-process spawning inside them is risky.
  - Sequential per-seed runs avoid GPU contention on the shared machine.
  - Each per-seed run is fully reproducible from its own command line.

Usage:
    uv run python -m mate_marl.scripts.train_multi_seed \\
        --method dqn \\
        --seeds 1 2 \\
        --mate-config MATE-4v8-9-fast.yaml \\
        --run-name dqn_4v8 \\
        -- \\
        <any other args, forwarded to the underlying train_dqn script>
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


METHOD_SCRIPTS = {
    "dqn": "mate_marl.scripts.train_dqn",
    "tcqmix": "mate_marl.scripts.train_tcqmix",
    "vdn": "mate_marl.scripts.train_vdn",
    "mappo": "mate_marl.scripts.train",
}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=list(METHOD_SCRIPTS))
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--mate-config", required=True)
    p.add_argument("--run-name", required=True,
                   help="base run name; per-seed runs are written to "
                        "checkpoints/<method>/<run-name>_seed<S>/")
    p.add_argument("--save-dir", default=None,
                   help="root checkpoints directory; default checkpoints/<method>")
    p.add_argument("--tb-log-dir", default="tb",
                   help="root TensorBoard directory; per-seed subdir appended")
    p.add_argument("--cuda-device", default=None,
                   help="value for CUDA_VISIBLE_DEVICES (e.g. '0' or '1'); "
                        "unset means inherit env")
    return p.parse_known_args()


def main() -> None:
    args, forward = parse_args()
    save_root = Path(args.save_dir) if args.save_dir else Path("checkpoints") / args.method

    script = METHOD_SCRIPTS[args.method]
    for seed in args.seeds:
        run_name = f"{args.run_name}_seed{seed}"
        save_dir = save_root  # underlying script writes to save_dir/<run-name>
        tb_log_dir = args.tb_log_dir if args.tb_log_dir else None

        cmd = [
            sys.executable, "-u", "-m", script,
            "--mate-config", args.mate_config,
            "--seed", str(seed),
            "--run-name", run_name,
            "--save-dir", str(save_dir),
        ] + (["--tb-log-dir", tb_log_dir] if tb_log_dir else []) + list(forward)

        env = dict(os.environ)
        if args.cuda_device is not None:
            env["CUDA_VISIBLE_DEVICES"] = args.cuda_device
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        print(f"\n========================================================")
        print(f"[multi-seed] method={args.method} seed={seed}")
        print(f"[multi-seed] cmd: {' '.join(cmd)}")
        print(f"========================================================\n", flush=True)

        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            print(f"[multi-seed] seed {seed} FAILED with exit code {result.returncode}",
                  flush=True)
            sys.exit(result.returncode)


if __name__ == "__main__":
    main()
