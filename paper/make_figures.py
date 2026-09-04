"""Generate all figures for the paper from experiment CSVs.

Outputs:
  paper/figures/baseline_matrix.pdf    - coverage by scenario × baseline
  paper/figures/dqn_training_curve.pdf - mean_ret & q_mean over 3M steps
  paper/figures/transfer_failure.pdf   - zero-shot transfer vs random floor
  paper/figures/env_overview.pdf       - schematic of MATE-Hard wrappers
  paper/figures/architecture.pdf       - schematic of encoder + actor-critic
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D


HERE = Path(__file__).parent
RESULTS = HERE.parent / "results"
LOGS = HERE.parent / "logs"
OUT = HERE / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Common style: nice serif font like papers, color-blind-friendly palette.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "DejaVu Serif", "Times New Roman"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.5,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})
PALETTE = {
    "random":    "#7f7f7f",
    "naive":     "#bcbd22",
    "greedy":    "#1f77b4",
    "heuristic": "#ff7f0e",
    "dqn":       "#d62728",
}


# ---------- helpers ----------

def stat(rows, key):
    vals = np.array([float(r[key]) for r in rows])
    return vals.mean(), vals.std(ddof=1) / np.sqrt(len(vals))


def read_baseline(baseline, scenario):
    if scenario == "MATE-4v8-9":
        p = RESULTS / f"baseline_{baseline}.csv"
    else:
        p = RESULTS / f"baseline_{baseline}_{scenario}.csv"
    if not p.exists():
        return None
    with p.open() as f:
        return list(csv.DictReader(f))


# ---------- 1. baseline matrix ----------

def fig_baseline_matrix():
    scenarios = ["MATE-2v4-9", "MATE-4v4-9", "MATE-4v8-9", "MATE-8v8-9"]
    baselines = ["random", "naive", "greedy", "heuristic"]
    x = np.arange(len(scenarios))
    bar_w = 0.18

    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    for i, b in enumerate(baselines):
        means = []
        errs = []
        for s in scenarios:
            rows = read_baseline(b, s)
            if rows is None:
                means.append(0); errs.append(0)
            else:
                m, e = stat(rows, "mean_coverage_rate")
                means.append(m); errs.append(e)
        ax.bar(x + (i - 1.5) * bar_w, means, bar_w,
               yerr=errs, capsize=2.2, label=b.capitalize(),
               color=PALETTE[b], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("MATE-", "").replace("-9", "") for s in scenarios])
    ax.set_xlabel("Scenario (cameras vs targets)")
    ax.set_ylabel("Coverage rate")
    ax.set_ylim(0, 0.85)
    ax.set_title("Rule-based baseline coverage across MATE scenarios")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)
    fig.savefig(OUT / "baseline_matrix.pdf")
    plt.close(fig)


# ---------- 2. DQN training curve from logs ----------

LOG_PAT = re.compile(
    r"\[step\s+(\d+)\]\s+mean_ret_100=(-?\d+\.\d+|nan)\s+loss=(-?\d+\.\d+)"
    r"\s+q_mean=(-?\d+\.\d+)\s+eps=(\d+\.\d+)"
)


def parse_dqn_log(path: Path):
    if not path.exists():
        return None
    raw = path.read_text(errors="ignore").replace("\r", "\n")
    rows = []
    for line in raw.splitlines():
        m = LOG_PAT.search(line)
        if not m:
            continue
        try:
            step = int(m.group(1))
            mr = float(m.group(2)) if m.group(2) != "nan" else np.nan
            ls = float(m.group(3))
            q = float(m.group(4))
            ep = float(m.group(5))
            rows.append((step, mr, ls, q, ep))
        except ValueError:
            pass
    return np.array(rows)


def smooth(y, w=10):
    if len(y) < w:
        return y
    # nan-safe smoothing: replace nan with previous valid value
    arr = np.array(y, dtype=float)
    last = arr[0] if not np.isnan(arr[0]) else 0.0
    for i in range(len(arr)):
        if np.isnan(arr[i]):
            arr[i] = last
        else:
            last = arr[i]
    return np.convolve(arr, np.ones(w) / w, mode="valid")


def fig_dqn_training_curve():
    data = parse_dqn_log(LOGS / "dqn_4v8_v8.log")
    if data is None or len(data) < 5:
        # placeholder
        fig, ax = plt.subplots(figsize=(5.2, 3.0))
        ax.text(0.5, 0.5, "(no v8 log found)", ha="center", va="center")
        fig.savefig(OUT / "dqn_training_curve.pdf")
        plt.close(fig)
        return
    steps, mr, ls, q, ep = data.T
    # Smooth
    s_steps = steps[len(steps) - len(smooth(mr)):]
    smr = smooth(mr)
    sq = smooth(q)

    fig, ax1 = plt.subplots(figsize=(5.2, 2.9))
    ax1.plot(s_steps / 1e6, smr, color=PALETTE["dqn"], linewidth=1.4,
             label=r"$\overline{R}_{100}$ (shaped scale)")
    ax1.set_xlabel("Environment steps (millions)")
    ax1.set_ylabel("Mean episode return ($R$, shaped)", color=PALETTE["dqn"])
    ax1.tick_params(axis="y", labelcolor=PALETTE["dqn"])
    ax1.set_ylim(-9, -3)

    ax2 = ax1.twinx()
    ax2.plot(s_steps / 1e6, sq, color="#1f77b4", linewidth=1.2, linestyle="--",
             label="Q-value mean")
    ax2.plot(steps / 1e6, ep, color="#7f7f7f", linewidth=0.8, alpha=0.7,
             label=r"$\varepsilon$ schedule")
    ax2.set_ylabel("Q mean / $\\varepsilon$", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.set_ylim(-1.5, 1.2)
    ax2.grid(False)

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower right", frameon=True, framealpha=0.9)

    ax1.set_title("Double-DQN training on MATE-4v8-9-fast")
    fig.savefig(OUT / "dqn_training_curve.pdf")
    plt.close(fig)


# ---------- 3. transfer failure bar chart ----------

def fig_transfer_failure():
    # Load DQN v8 final eval (4v8) and transfer eval (other scenarios)
    transfer = list(csv.DictReader(open(RESULTS / "dqn_v8_transfer_eval.csv")))
    final_4v8 = list(csv.DictReader(open(RESULTS / "dqn_v8_final_eval.csv")))

    scenarios = ["MATE-2v4-9", "MATE-4v4-9", "MATE-4v8-9", "MATE-8v8-9"]
    dqn_means, dqn_errs = [], []
    rand_means, rand_errs = [], []
    for s in scenarios:
        if s == "MATE-4v8-9":
            rs = final_4v8
        else:
            rs = [r for r in transfer if r["scenario"] == s]
        m, e = stat(rs, "mean_coverage_rate")
        dqn_means.append(m); dqn_errs.append(e)

        rb = read_baseline("random", s)
        m2, e2 = stat(rb, "mean_coverage_rate")
        rand_means.append(m2); rand_errs.append(e2)

    x = np.arange(len(scenarios))
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ax.bar(x - 0.18, rand_means, 0.34, yerr=rand_errs, capsize=2.5,
           label="Random (rule baseline, same scenario)",
           color=PALETTE["random"], edgecolor="black", linewidth=0.4)
    ax.bar(x + 0.18, dqn_means, 0.34, yerr=dqn_errs, capsize=2.5,
           label="Double-DQN-MARL (trained on 4v8-9)",
           color=PALETTE["dqn"], edgecolor="black", linewidth=0.4)
    # Annotate in-distribution scenario
    in_dist_idx = scenarios.index("MATE-4v8-9")
    ax.annotate("in-distribution", xy=(in_dist_idx + 0.18, dqn_means[in_dist_idx] + 0.02),
                xytext=(in_dist_idx + 0.5, dqn_means[in_dist_idx] + 0.20),
                fontsize=7.5, ha="left",
                arrowprops=dict(arrowstyle="-", color="black", lw=0.5))
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("MATE-", "").replace("-9", "") for s in scenarios])
    ax.set_xlabel("Eval scenario (training scenario: 4v8)")
    ax.set_ylabel("Coverage rate")
    ax.set_ylim(0, 0.45)
    ax.set_title("Zero-shot transfer: DQN-MARL collapses below random")
    ax.legend(loc="upper left", frameon=False)
    fig.savefig(OUT / "transfer_failure.pdf")
    plt.close(fig)


# ---------- 4. env overview schematic ----------

def fig_env_overview():
    fig, ax = plt.subplots(figsize=(5.2, 2.7))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")

    def box(x, y, w, h, label, color, edge="black"):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                           linewidth=0.8, edgecolor=edge, facecolor=color)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=8.5)

    def arrow(x1, y1, x2, y2):
        a = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle="->", mutation_scale=10, lw=0.9, color="black")
        ax.add_patch(a)

    # Stack from bottom to top: env -> wrappers -> trainer
    box(0.5, 0.4, 2.0, 0.8, "MultiAgentTracking\n(base MATE env)", "#e8e8e8")
    box(3.0, 0.4, 2.0, 0.8, "MultiCamera\n(single-team)", "#e8e8e8")
    box(5.5, 0.4, 1.8, 0.8, "DiscreteCamera\n(actions × 5)", "#fff2cc")
    box(0.5, 1.6, 1.8, 0.8, "Heterogeneous\nCameras", "#d5e8d4")
    box(2.5, 1.6, 1.8, 0.8, "Energy\nConstraint", "#d5e8d4")
    box(4.5, 1.6, 1.8, 0.8, "Dynamic\nFog", "#d5e8d4")
    box(6.5, 1.6, 1.6, 0.8, "Soft-Cov\nShaping", "#fff2cc")
    box(0.5, 2.8, 4.0, 0.8, "MateMARLDictObs\n(typed entity tokens)", "#dae8fc")
    box(4.7, 2.8, 3.0, 0.8, "DiscreteFlattenForDQN", "#dae8fc")
    box(0.5, 4.0, 7.2, 0.8, "Param-shared Q-net = Set-Transformer + Type-Conditioned MoE",
        "#f8cecc", edge="#990000")

    # arrows
    arrow(2.5, 0.8, 3.0, 0.8)
    arrow(5.0, 0.8, 5.5, 0.8)
    arrow(4.4, 1.2, 4.4, 1.5)  # MATE -> wrappers
    arrow(4.4, 2.4, 4.4, 2.7)  # wrappers -> dictobs
    arrow(4.4, 3.6, 4.4, 3.9)  # dictobs -> qnet
    arrow(7.4, 1.2, 7.4, 1.5)

    # legend
    ax.text(8.5, 3.2, "MATE-Hard\nwrappers", fontsize=8, color="#005000", ha="center")
    ax.text(8.5, 0.8, "Discretize\n+ shape", fontsize=8, color="#806000", ha="center")

    ax.set_title("MATE-Hard environment wrapper stack and Q-network")
    fig.savefig(OUT / "env_overview.pdf")
    plt.close(fig)


# ---------- 5. architecture schematic ----------

def fig_architecture():
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    def box(x, y, w, h, label, color="#f3f3f3", fontsize=8, edge="black"):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                           linewidth=0.8, edgecolor=edge, facecolor=color)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fontsize)

    def arrow(x1, y1, x2, y2, lw=0.9):
        a = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle="->", mutation_scale=8, lw=lw, color="black")
        ax.add_patch(a)

    # ----- input row (token types) -----
    tokens = [
        ("self\n(C,F)", "#dae8fc"),
        ("teammate\n(C,C-1,F)", "#dae8fc"),
        ("target\n(C,T,F)", "#dae8fc"),
        ("obstacle\n(C,O,F)", "#dae8fc"),
        ("fog\n(C,F_f,F)", "#dae8fc"),
        ("preserved\n(C,F)", "#dae8fc"),
    ]
    for i, (lbl, c) in enumerate(tokens):
        box(0.2 + 1.6 * i, 0.2, 1.4, 0.7, lbl, c, fontsize=6.5)
    # per-type linear projection
    for i in range(6):
        arrow(0.9 + 1.6 * i, 0.95, 0.9 + 1.6 * i, 1.3)
    box(0.2, 1.3, 9.6, 0.55, "Per-entity-type linear projection  +  entity-type embedding  +  camera-type embedding",
        "#fff2cc", fontsize=7.5)
    arrow(5.0, 1.9, 5.0, 2.2)
    # mask * concat
    box(0.2, 2.2, 9.6, 0.55, "Validity-mask multiply  →  concatenate to a single token set  (B, N_total, E)",
        "#fff2cc", fontsize=7.5)
    arrow(5.0, 2.8, 5.0, 3.1)
    # MAB block 1
    box(0.4, 3.1, 9.2, 0.7, "MAB block 1: MHA  +  Top-K MoE FF  (4 experts, k=2)",
        "#f8cecc", fontsize=7.5)
    arrow(5.0, 3.85, 5.0, 4.05)
    # MAB block 2
    box(0.4, 4.05, 9.2, 0.7, "MAB block 2: MHA  +  Top-K MoE FF  (4 experts, k=2)",
        "#f8cecc", fontsize=7.5)
    arrow(5.0, 4.8, 5.0, 5.0)
    # pool + heads
    box(0.4, 5.0, 4.4, 0.6, "Pool SELF token  →  Latent (B, E)", "#d5e8d4", fontsize=7.5)
    box(5.2, 5.0, 2.0, 0.6, "Actor MLP", "#d5e8d4", fontsize=7.5)
    box(7.4, 5.0, 2.2, 0.6, "Q-head / Value head", "#d5e8d4", fontsize=7.5)
    arrow(4.8, 5.3, 5.2, 5.3)
    arrow(7.2, 5.3, 7.4, 5.3)

    ax.set_title("Set-Transformer + Type-Conditioned MoE encoder + actor/critic heads",
                 pad=2)
    fig.savefig(OUT / "architecture.pdf")
    plt.close(fig)


# ---------- main ----------

def main():
    fig_baseline_matrix();   print("OK baseline_matrix")
    fig_dqn_training_curve(); print("OK dqn_training_curve")
    fig_transfer_failure();   print("OK transfer_failure")
    fig_env_overview();       print("OK env_overview")
    fig_architecture();       print("OK architecture")


if __name__ == "__main__":
    main()
