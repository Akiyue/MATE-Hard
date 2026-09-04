#!/usr/bin/env python3
"""Training-curves figure for the six headline methods on MATE-Hard-4v8-9.

This script reads the per-iteration metrics that the trainers print to stdout
(``logs/``) and falls back to the released TensorBoard event files (``tb/``)
when a log is unavailable.  Both sources carry the same
``rollout/mean_return_100`` series and give identical curves, so the figure can
be regenerated from the public repository as it stands.

Two things this figure has to get right, both of which the earlier
TensorBoard-based version did not:

1. **A common horizontal axis.**  The value-based trainers advance their step
   counter by ``num_envs`` per environment step and log ``[step N]`` with N in
   *environment steps*.  MAPPO advances by ``num_envs * num_cameras`` and logs
   ``steps=N`` in *agent-steps*.  A nominal 3M budget is therefore 3M
   environment steps for the value-based methods and 3M / num_cameras =
   750k environment steps for MAPPO.  We divide MAPPO's counter by the number
   of cameras so every curve is plotted against environment steps; MAPPO's
   curve consequently ends at ~754k, which is the truth.

2. **An honest y-axis caveat.**  All six trainers log ``mean_ret_100``, the
   100-episode rolling mean of the episode return, computed from the raw
   environment reward.  For the five value-based methods that reward includes
   the SCS shaping term at w = 0.02; MAPPO trains without shaping, so its
   returns exclude it.  The offset is small at this weight but it is not zero,
   and the caption says so.
"""
from __future__ import annotations

import os
import re

import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOGS = os.path.join(ROOT, "logs")
TB = os.path.join(ROOT, "tb")
SCALAR = "rollout/mean_return_100"
OUT = os.path.join(HERE, "figures", "training_curves.pdf")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

NUM_CAMERAS = 4          # MATE-Hard-4v8-9
MAX_ENV_STEPS = 3_000_000
GRID = np.linspace(0, MAX_ENV_STEPS, 400)
SMOOTH_WIN = 25

# (label, [seed0, seed1, seed2] log stems, colour)
METHODS = [
    ("TC-QMIX (Ours)", ["tcqmix_4v8_v1", "tcqmix_4v8_seed1", "tcqmix_4v8_seed2"], "#1F77B4"),
    ("Double-DQN",     ["dqn_4v8_v8", "dqn_4v8_seed1", "dqn_4v8_seed2"], "#FF7F0E"),
    ("QMIX (vanilla)", ["qmix_vanilla_4v8_seed0", "qmix_vanilla_4v8_seed1",
                        "qmix_vanilla_4v8_seed2"], "#2CA02C"),
    ("TC-QMIX (types)", ["tcqmix_types_4v8_seed0", "tcqmix_types_4v8_seed1",
                         "tcqmix_types_4v8_seed2"], "#9467BD"),
    ("VDN",            ["vdn_4v8_seed0", "vdn_4v8_seed1", "vdn_4v8_seed2"], "#8C564B"),
    ("MAPPO",          ["mappo_4v8_seed0", "mappo_4v8_seed1", "mappo_4v8_seed2"], "#D62728"),
]

# Value-based trainers: "[step   15000] mean_ret_100=-4.419 ..."
STEP_RE = re.compile(r"\[step\s+(\d+)\]\s+mean_ret_100=(-?[\d.]+|nan)")
# MAPPO:                "[iter    1] steps=32768 mean_ret_100=-6.161 ..."
ITER_RE = re.compile(r"\[iter\s+\d+\]\s+steps=(\d+)\s+mean_ret_100=(-?[\d.]+|nan)")


def _from_log(stem: str) -> tuple[list[tuple[int, float]], float]:
    """Parse one stdout log. Returns (points, step_scale)."""
    path = os.path.join(LOGS, stem + ".log")
    if not os.path.exists(path):
        return [], 1.0
    text = open(path, errors="replace").read().replace("\r", "\n")
    pts = [(int(a), float(b)) for a, b in STEP_RE.findall(text) if b != "nan"]
    if pts:                                   # already environment steps
        return pts, 1.0
    pts = [(int(a), float(b)) for a, b in ITER_RE.findall(text) if b != "nan"]
    return pts, 1.0 / NUM_CAMERAS             # MAPPO logs agent-steps


def _from_tensorboard(stem: str) -> tuple[list[tuple[int, float]], float]:
    """Fall back to the released TensorBoard event files under ``tb/``.

    The same step-counter asymmetry applies: MAPPO writes ``self.timesteps``
    (agent-steps) as its TensorBoard step, the value-based trainers write
    ``env_step_count`` (environment steps).
    """
    run_dir = os.path.join(TB, stem)
    if not os.path.isdir(run_dir):
        return [], 1.0
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        return [], 1.0
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    if SCALAR not in ea.Tags()["scalars"]:
        return [], 1.0
    pts = [(int(e.step), float(e.value)) for e in ea.Scalars(SCALAR)]
    scale = 1.0 / NUM_CAMERAS if stem.startswith("mappo") else 1.0
    return pts, scale


def load_curve(stem: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (environment_steps, mean_ret_100) for one run.

    Prefers the stdout logs in ``logs/``; falls back to the TensorBoard event
    files in ``tb/``, which are the ones shipped in the public repository.
    Both give identical curves.
    """
    pts, scale = _from_log(stem)
    if not pts:
        pts, scale = _from_tensorboard(stem)
    if not pts:
        return np.array([]), np.array([])

    steps = np.array([p[0] for p in pts], dtype=float) * scale
    vals = np.array([p[1] for p in pts], dtype=float)
    keep = (steps <= MAX_ENV_STEPS) & np.isfinite(vals)
    return steps[keep], vals[keep]


def smooth(y: np.ndarray, w: int) -> np.ndarray:
    """Exponentially weighted moving average."""
    if len(y) < 3:
        return y
    alpha = 2.0 / (min(w, len(y)) + 1)
    out = np.empty_like(y)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1 - alpha) * out[i - 1]
    return out


def interp(steps: np.ndarray, vals: np.ndarray) -> np.ndarray:
    if len(steps) == 0:
        return np.full_like(GRID, np.nan)
    return np.interp(GRID, steps, vals, left=np.nan, right=np.nan)


def main() -> None:
    # Past MAPPO's 754k-step horizon its seed slices are all-NaN by
    # construction; nanmean/nanstd warn about that and we expect it.
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    mappo_end = None

    for label, stems, colour in METHODS:
        curves = []
        for stem in stems:
            st, va = load_curve(stem)
            if len(st) == 0:
                print(f"[warn] no data for {stem}")
                continue
            curves.append(interp(st, smooth(va, SMOOTH_WIN)))
            if label == "MAPPO":
                mappo_end = max(mappo_end or 0.0, float(st.max()))
        if not curves:
            continue
        arr = np.vstack(curves)
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(arr, axis=0)
            n = np.sum(~np.isnan(arr), axis=0)
            se = np.nanstd(arr, axis=0, ddof=0) / np.sqrt(np.maximum(n, 1))
        ax.plot(GRID, mean, color=colour, lw=1.6, label=label)
        ax.fill_between(GRID, mean - se, mean + se, color=colour, alpha=0.15, lw=0)

    if mappo_end:
        ax.axvline(mappo_end, color="#D62728", ls=":", lw=1.1)
        ax.annotate(f"MAPPO budget ends\n({mappo_end/1e3:.0f}k env steps)",
                    xy=(mappo_end, ax.get_ylim()[0]),
                    xytext=(mappo_end + 90_000, ax.get_ylim()[0] + 0.45),
                    color="#D62728", fontsize=7.5, style="italic")

    ax.set_xlabel("environment steps  (MAPPO's agent-step counter divided by "
                  f"{NUM_CAMERAS})")
    ax.set_ylabel("mean_ret_100  (100-episode rolling mean of episode return)")
    ax.set_title("Training curves on MATE-Hard-4v8-9  (mean ± 1 SE over 3 seeds)")
    ax.grid(alpha=0.3, ls=":")
    ax.legend(loc="lower right", fontsize=8, ncol=2, framealpha=0.9)
    ax.set_xlim(0, MAX_ENV_STEPS)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
