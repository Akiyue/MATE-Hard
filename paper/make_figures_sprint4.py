"""Sprint-4 paper figures.

  paper/figures/env_overview.pdf            - redesigned wrapper-stack pipeline
  paper/figures/method_comparison.pdf       - 6-method bar chart vs rule ceiling
  paper/figures/transfer_failure.pdf        - 6 methods x 4 scenarios bar chart
  paper/figures/rule_degrade.pdf            - plain MATE vs MATE-Hard rule degradation
  paper/figures/ablation.pdf                - TC-QMIX mixer-hyper-input ablation
  paper/figures/pairwise_heatmap.pdf        - pairwise Cohen's d heatmap on 4v8-9
  paper/figures/count_scaling.pdf           - coverage vs team-size scaling (transfer)
  paper/figures/seed_variance.pdf           - per-seed dot plot per method

Reads from results/EVAL_MATRIX_*.csv (Sprint 2/3 aggregated).
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# data loaders
# --------------------------------------------------------------------------

def load_cross_seed() -> dict[tuple[str, str], tuple[float, float]]:
    """Returns {(method, scenario): (mean, se)} for coverage_rate rows."""
    out: dict[tuple[str, str], tuple[float, float]] = {}
    with (RESULTS / "EVAL_MATRIX_cross_seed.csv").open() as f:
        for r in csv.DictReader(f):
            if r["metric"] != "coverage_rate":
                continue
            out[(r["method"], r["scenario"])] = (float(r["mean"]), float(r["se"]))
    return out


def rule_mh(data, inner: str, scenario: str) -> float:
    """Mean coverage of a rule agent on MATE-Hard, best of the two adapters.

    The learned checkpoints are evaluated with the heterogeneous-camera, energy
    and fog wrappers enabled, so plain-MATE rule numbers are not a like-for-like
    reference for them.
    """
    return max(data[(f"rule_{inner}_{a}MH", scenario)][0] for a in ("naive", "informed"))


def load_long() -> list[dict]:
    out = []
    with (RESULTS / "EVAL_MATRIX_long.csv").open() as f:
        for r in csv.DictReader(f):
            out.append(r)
    return out


def load_pairwise() -> list[dict]:
    out = []
    with (RESULTS / "EVAL_MATRIX_pairwise.csv").open() as f:
        for r in csv.DictReader(f):
            out.append(r)
    return out


# --------------------------------------------------------------------------
# 1. Redesigned environment / pipeline overview
# --------------------------------------------------------------------------

def fig_env_overview() -> None:
    """MATE-Hard pipeline: each tier is a single full-width box; vertical arrows
    between tiers. Categories shown as left-margin tags. Trainers fan out at top."""

    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    def box(x, y, w, h, label, color, edge="#222", fontsize=9.0,
            fontweight="normal", text_color="black"):
        b = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.4,rounding_size=1.6",
                           linewidth=0.9, edgecolor=edge, facecolor=color)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, fontweight=fontweight, color=text_color)

    def arrow(x1, y1, x2, y2, lw=1.1, color="#333"):
        a = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle="-|>", mutation_scale=12,
                            lw=lw, color=color, shrinkA=2, shrinkB=2)
        ax.add_patch(a)

    # Layout: 6 stacked tiers; left tag column 0-16, content column 18-92.
    CL, CR = 18, 92      # content column bounds
    CX = (CL + CR) / 2
    TIER_H = 8

    tiers = [
        # (y_center, tag, tag_color, box_color, edge, label, fontweight)
        (10, "base env",           "#555", "#ececec", "#444",
         "MultiAgentTracking  (two-team MATE base env, MultiCamera adapter)", "normal"),
        (26, "MATE-Hard\nwrappers", "#005000", "#d5e8d4", "#246224",
         "HeterogeneousCameras  →  EnergyConstraint  →  DynamicFog", "normal"),
        (42, "action +\nshaping",  "#806000", "#fff2cc", "#806000",
         "DiscreteCamera  (5×5 levels)  →  soft_coverage_score shaping  (w = 0.02)", "normal"),
        (58, "observation",        "#1c4e80", "#dae8fc", "#1c4e80",
         "MateMARLDictObs  →  typed entity-token dict\n"
         "{self, teammate, target, obstacle, fog, preserved}", "normal"),
        (74, "encoder",            "#990000", "#f8cecc", "#990000",
         "Set-Transformer  +  Type-Conditioned Top-K MoE", "bold"),
    ]
    for y, tag, tagc, fill, edge, label, fw in tiers:
        # left-margin tag
        ax.text(CL - 2, y + TIER_H / 2, tag, ha="right", va="center",
                fontsize=8.5, color=tagc, fontweight="bold")
        box(CL, y, CR - CL, TIER_H, label, fill, edge=edge,
            fontsize=9 if fw == "normal" else 9.5, fontweight=fw)

    # Vertical arrows between tiers (centered)
    for i in range(len(tiers) - 1):
        y_from = tiers[i][0] + TIER_H
        y_to   = tiers[i + 1][0]
        arrow(CX, y_from, CX, y_to)

    # ---- six trainers row at the top
    Y_TRAIN = 90
    trainers = [
        ("Double-DQN",           "#ffd6a5"),
        ("MAPPO",                "#ffadad"),
        ("VDN",                  "#caffbf"),
        ("QMIX",                 "#9bf6ff"),
        ("TC-QMIX\n(Ours)",      "#bdb2ff"),
        ("TC-QMIX\n(types only)","#fdffb6"),
    ]
    n = len(trainers)
    gap = 1.2
    tw = (CR - CL - (n - 1) * gap) / n
    for i, (lbl, col) in enumerate(trainers):
        x = CL + i * (tw + gap)
        fw = "bold" if "Ours" in lbl else "normal"
        box(x, Y_TRAIN, tw, 7.5, lbl, col, fontsize=8, fontweight=fw)

    # left tag for trainers row
    ax.text(CL - 2, Y_TRAIN + 3.75, "trainers", ha="right", va="center",
            fontsize=8.5, color="#333", fontweight="bold")

    # Encoder-to-trainer arrows: a short fan from encoder top up to each trainer base
    y_enc_top = tiers[-1][0] + TIER_H
    for i in range(n):
        x = CL + i * (tw + gap) + tw / 2
        arrow(x, y_enc_top, x, Y_TRAIN)

    fig.tight_layout(pad=0.5)
    fig.savefig(OUT / "env_overview.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'env_overview.pdf'}")


def fig_architecture() -> None:
    """Per-camera encoder.

    Layout: single vertical flow up to MAB block 2; then a Y-split into two
    branches. The LEFT branch (per-camera latent from pooled SELF token) feeds
    the actor head and the per-agent Q-head. The RIGHT branch (team state from
    mean-pooling across cameras) feeds the centralised critic and the mixing
    hypernet. The split is drawn with explicit junction lines so that each
    head visibly receives its source.
    """

    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    def box(x, y, w, h, label, color, edge="#222", fontsize=9.0,
            fontweight="normal"):
        b = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.4,rounding_size=1.6",
                           linewidth=0.9, edgecolor=edge, facecolor=color)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, fontweight=fontweight)

    def arrow(x1, y1, x2, y2, lw=1.0, color="#333"):
        a = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle="-|>", mutation_scale=11,
                            lw=lw, color=color, shrinkA=1.5, shrinkB=1.5)
        ax.add_patch(a)

    def line(x1, y1, x2, y2, lw=1.0, color="#333"):
        ax.plot([x1, x2], [y1, y2], color=color, lw=lw, solid_capstyle="round",
                zorder=1)

    XL, XR = 6, 94
    CX = (XL + XR) / 2

    # =====================================================================
    # Vertical encoder column (bottom -> middle)
    # =====================================================================

    # ---- input token tiles (row at bottom)
    tokens = ["self", "teammate", "target", "obstacle", "fog", "preserved"]
    n = len(tokens)
    gap = 1.6
    tw = (XR - XL - (n - 1) * gap) / n
    Y_TOK = 4
    for i, t in enumerate(tokens):
        x = XL + i * (tw + gap)
        box(x, Y_TOK, tw, 6, t, "#dae8fc", edge="#1c4e80", fontsize=8.5)
    # italic caption under tokens
    ax.text(CX, Y_TOK - 3.0,
            "typed entity tokens from MateMARLDictObs",
            ha="center", fontsize=8, style="italic", color="#1c4e80")

    # ---- linear projection + embeddings band
    Y_PROJ = 18
    box(XL, Y_PROJ, XR - XL, 7,
        "per-entity linear projection  +  entity-type embedding  "
        "+  camera-type embedding",
        "#fff2cc", edge="#806000", fontsize=9)
    for i in range(n):
        x = XL + i * (tw + gap) + tw / 2
        arrow(x, Y_TOK + 6, x, Y_PROJ)

    # ---- validity mask + concat
    Y_CONCAT = 30
    box(XL, Y_CONCAT, XR - XL, 6,
        "validity mask  ⊙  concatenate  →  single token set",
        "#fff2cc", edge="#806000", fontsize=9)
    arrow(CX, Y_PROJ + 7, CX, Y_CONCAT)

    # ---- MAB block 1
    Y_MAB1 = 40
    box(XL, Y_MAB1, XR - XL, 8,
        "MAB block 1:  multi-head self-attention  +  Top-K MoE FFN  "
        "(4 experts, k=2)",
        "#f8cecc", edge="#990000", fontsize=9)
    arrow(CX, Y_CONCAT + 6, CX, Y_MAB1)

    # ---- MAB block 2
    Y_MAB2 = 52
    box(XL, Y_MAB2, XR - XL, 8,
        "MAB block 2:  multi-head self-attention  +  Top-K MoE FFN  "
        "(4 experts, k=2)",
        "#f8cecc", edge="#990000", fontsize=9)
    arrow(CX, Y_MAB1 + 8, CX, Y_MAB2)

    # =====================================================================
    # Y-split into per-camera branch (LEFT) and team branch (RIGHT)
    # =====================================================================

    Y_SPLIT  = 64   # junction y after MAB2
    Y_POOL   = 70   # pooling boxes
    Y_HEAD_BAR = 84
    Y_HEAD   = 88

    # branch centerlines
    LX, RX = 25, 75      # branch column centers

    # Y-split: plain lines for the junction (no intermediate arrowheads),
    # arrowheads only where the signal arrives at a box.
    line(CX, Y_MAB2 + 8, CX, Y_SPLIT)        # central stub
    line(LX, Y_SPLIT, RX, Y_SPLIT)           # horizontal junction bar
    arrow(LX, Y_SPLIT, LX, Y_POOL)           # left drop into pool
    arrow(RX, Y_SPLIT, RX, Y_POOL)           # right drop into pool

    # ---- pool boxes (per-camera + team)
    pool_w = 38
    box(LX - pool_w / 2, Y_POOL, pool_w, 7,
        "pool SELF token\nper-camera latent",
        "#d5e8d4", edge="#246224", fontsize=8.5)
    box(RX - pool_w / 2, Y_POOL, pool_w, 7,
        "mean-pool across cameras\nteam state",
        "#d5e8d4", edge="#246224", fontsize=8.5)

    # =====================================================================
    # Head row: 2 heads on each branch
    # =====================================================================
    head_w = 17
    head_h = 8
    head_gap = 3
    branches = [
        (LX, [
            ("actor head\n(MAPPO)",            "#ffadad"),
            ("per-agent\nQ-head",              "#bdb2ff"),
        ]),
        (RX, [
            ("centralised critic\n(MAPPO)",    "#ffd6a5"),
            ("mixing hypernet\n(QMIX / TC-QMIX)", "#9bf6ff"),
        ]),
    ]
    for bx, heads in branches:
        # branch local "bar" connecting the two heads
        total = 2 * head_w + head_gap
        bar_left  = bx - total / 2
        bar_right = bx + total / 2
        # plain stub up from pool to bar; arrowheads only on the final hops
        line(bx, Y_POOL + 7, bx, Y_HEAD_BAR)
        line(bar_left + head_w / 2, Y_HEAD_BAR,
             bar_right - head_w / 2, Y_HEAD_BAR)
        # two head boxes + their arrows
        for i, (lbl, col) in enumerate(heads):
            hx = bar_left + i * (head_w + head_gap)
            arrow(hx + head_w / 2, Y_HEAD_BAR, hx + head_w / 2, Y_HEAD)
            box(hx, Y_HEAD, head_w, head_h, lbl, col, fontsize=7.5)

    # =====================================================================
    # Branch annotations: place between the junction bar and the pool boxes
    # so they label each downward arrow without crowding the bar itself.
    # =====================================================================
    ax.text(LX + 1, (Y_SPLIT + Y_POOL) / 2, "per-camera path",
            ha="left", va="center", fontsize=7.5, color="#246224", style="italic")
    ax.text(RX + 1, (Y_SPLIT + Y_POOL) / 2, "team-state path",
            ha="left", va="center", fontsize=7.5, color="#246224", style="italic")

    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "architecture.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'architecture.pdf'}")


# --------------------------------------------------------------------------
# 2. Method-comparison bar chart on MATE-Hard-4v8-9
# --------------------------------------------------------------------------

def fig_method_comparison() -> None:
    data = load_cross_seed()
    sc = "MATE-4v8-9"

    methods = [
        ("Rule Greedy\n(informed-MH)", "rule_greedy_informedMH", "#2ca02c"),
        ("TC-QMIX\n(Ours)",            "TCQMIX",                 "#1f77b4"),
        ("Double-DQN",                  "DQN",                    "#ff7f0e"),
        ("QMIX",                        "QMIX",                   "#d62728"),
        ("TC-QMIX\n(types-only)",       "TCQMIX_types",           "#9467bd"),
        ("VDN",                         "VDN",                    "#8c564b"),
        ("MAPPO",                       "MAPPO",                  "#e377c2"),
    ]
    means = [data[(m, sc)][0] for _, m, _ in methods]
    ses   = [data[(m, sc)][1] for _, m, _ in methods]

    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    xs = np.arange(len(methods))
    colors = [c for _, _, c in methods]
    bars = ax.bar(xs, means, yerr=[1.96 * s for s in ses], capsize=4,
                  color=colors, edgecolor="black", linewidth=0.4,
                  error_kw=dict(elinewidth=0.8, ecolor="black"))
    for i, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                f"{means[i]:.3f}", ha="center", va="bottom", fontsize=7)

    ceil = data[("rule_greedy_informedMH", sc)][0]
    ax.axhline(ceil, ls="--", color="#2ca02c", lw=0.8, alpha=0.7)
    ax.text(len(methods) - 0.1, ceil + 0.003, "rule ceiling",
            color="#2ca02c", fontsize=7, ha="right")

    ax.set_xticks(xs)
    ax.set_xticklabels([n for n, _, _ in methods], fontsize=7.5)
    ax.set_ylabel("coverage_rate  (mean ± 95% CI)")
    ax.set_title("Final coverage on MATE-Hard-4v8-9  (3 seeds, n = 270 per method)")
    ax.set_ylim(0, 0.27)
    ax.grid(axis="y", alpha=0.3, ls=":")
    fig.tight_layout()
    fig.savefig(OUT / "method_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'method_comparison.pdf'}")


# --------------------------------------------------------------------------
# 3. Transfer matrix: 6 methods x 4 scenarios
# --------------------------------------------------------------------------

def fig_transfer_matrix() -> None:
    data = load_cross_seed()
    scenarios = ["MATE-2v4-9", "MATE-4v4-9", "MATE-4v8-9", "MATE-8v8-9"]
    scenario_labels = ["2v4-9\n(transfer)", "4v4-9\n(transfer)",
                       "4v8-9\n(in-dist.)", "8v8-9\n(transfer)"]

    learned = [
        ("Double-DQN",          "DQN",          "#ff7f0e"),
        ("TC-QMIX (Ours)",      "TCQMIX",       "#1f77b4"),
        ("QMIX",                "QMIX",         "#d62728"),
        ("TC-QMIX (types)",     "TCQMIX_types", "#9467bd"),
        ("VDN",                 "VDN",          "#8c564b"),
        ("MAPPO",               "MAPPO",        "#e377c2"),
    ]

    def get(method_base: str, scenario: str):
        if scenario == "MATE-4v8-9":
            return data[(method_base, scenario)]
        return data[(method_base + "_transfer", scenario)]

    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    width = 0.12
    xs = np.arange(len(scenarios))

    for i, (lbl, base, col) in enumerate(learned):
        means = [get(base, s)[0] for s in scenarios]
        ses   = [get(base, s)[1] for s in scenarios]
        offset = (i - (len(learned) - 1) / 2) * width
        ax.bar(xs + offset, means, width=width * 0.95,
               yerr=[1.96 * s for s in ses], capsize=2,
               label=lbl, color=col, edgecolor="black", linewidth=0.3,
               error_kw=dict(elinewidth=0.6, ecolor="black"))

    for j, s in enumerate(scenarios):
        ceil = rule_mh(data, "greedy", s)
        ax.hlines(ceil, xs[j] - 0.42, xs[j] + 0.42,
                  colors="#2ca02c", linestyles="--", linewidth=1.2)
        ax.text(xs[j], ceil + 0.010,
                f"Greedy: {ceil:.2f}", ha="center", color="#2ca02c", fontsize=7)
        floor = rule_mh(data, "random", s)
        ax.hlines(floor, xs[j] - 0.42, xs[j] + 0.42,
                  colors="gray", linestyles=":", linewidth=1.0)
        ax.text(xs[j], max(floor - 0.028, 0.002),
                f"Random: {floor:.2f}", ha="center", color="gray", fontsize=6.5)

    ax.set_xticks(xs)
    ax.set_xticklabels(scenario_labels, fontsize=8)
    ax.set_ylabel("coverage_rate  (mean ± 95% CI)")
    ax.set_title("Zero-shot transfer of 4v8-trained checkpoints, all on MATE-Hard\n"
                 "(n ≥ 220 per learned cell; references are MATE-Hard rule agents)")
    ax.set_ylim(0, 0.42)
    ax.grid(axis="y", alpha=0.3, ls=":")
    ax.legend(loc="upper left", ncol=3, fontsize=7, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "transfer_failure.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'transfer_failure.pdf'}")


# --------------------------------------------------------------------------
# 4. NEW: plain MATE vs MATE-Hard rule-based degradation
# --------------------------------------------------------------------------

def fig_rule_degrade() -> None:
    data = load_cross_seed()
    scenarios = ["MATE-2v4-9", "MATE-4v4-9", "MATE-4v8-9", "MATE-8v8-9"]
    scenario_labels = ["2v4-9", "4v4-9", "4v8-9", "8v8-9"]

    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    width = 0.20
    xs = np.arange(len(scenarios))

    # Plain MATE: greedy + heuristic
    g_plain  = [data[("rule_greedy",    s)][0] for s in scenarios]
    h_plain  = [data[("rule_heuristic", s)][0] for s in scenarios]
    # MATE-Hard: best of the naive / informed adapter, matching the "best
    # MATE-Hard entry" definition used for the Delta-% column of Table 2.
    g_mh = [max(data[("rule_greedy_naiveMH",    s)][0],
                data[("rule_greedy_informedMH", s)][0]) for s in scenarios]
    h_mh = [max(data[("rule_heuristic_naiveMH",    s)][0],
                data[("rule_heuristic_informedMH", s)][0]) for s in scenarios]

    ax.bar(xs - 1.5*width, g_plain, width, label="Greedy / plain MATE",
           color="#2ca02c", edgecolor="black", linewidth=0.3)
    ax.bar(xs - 0.5*width, h_plain, width, label="Heuristic / plain MATE",
           color="#7fcf7f", edgecolor="black", linewidth=0.3)
    ax.bar(xs + 0.5*width, g_mh,    width, label="Greedy / MATE-Hard (best adapter)",
           color="#d62728", edgecolor="black", linewidth=0.3)
    ax.bar(xs + 1.5*width, h_mh,    width, label="Heuristic / MATE-Hard (best adapter)",
           color="#ff9d9c", edgecolor="black", linewidth=0.3)

    # annotate degradation % above plain-MATE Greedy bars
    for j, s in enumerate(scenarios):
        drop = 100 * (1 - max(g_mh[j], h_mh[j]) / max(g_plain[j], h_plain[j]))
        ax.text(xs[j], max(g_plain[j], h_plain[j]) + 0.02,
                f"−{drop:.0f}%", ha="center", color="#990000",
                fontsize=8, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels(scenario_labels)
    ax.set_ylabel("coverage_rate  (mean over 90 episodes)")
    ax.set_title("Rule-based agents on plain MATE vs MATE-Hard  (Finding 1)")
    ax.set_ylim(0, 0.85)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, ls=":")
    fig.tight_layout()
    fig.savefig(OUT / "rule_degrade.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'rule_degrade.pdf'}")


# --------------------------------------------------------------------------
# 5. NEW: TC-QMIX mixer-hyper-input ablation
# --------------------------------------------------------------------------

def fig_ablation() -> None:
    data = load_cross_seed()
    sc = "MATE-4v8-9"

    variants = [
        ("state +\ntypes\n(TC-QMIX)",  "TCQMIX",       "#1f77b4"),
        ("state only\n(vanilla QMIX)", "QMIX",         "#d62728"),
        ("types only",                  "TCQMIX_types", "#9467bd"),
    ]
    means = [data[(m, sc)][0] for _, m, _ in variants]
    ses   = [data[(m, sc)][1] for _, m, _ in variants]

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    xs = np.arange(len(variants))
    bars = ax.bar(xs, means, yerr=[1.96 * s for s in ses], capsize=5,
                  color=[c for _, _, c in variants],
                  edgecolor="black", linewidth=0.4,
                  error_kw=dict(elinewidth=0.8, ecolor="black"))
    for i, b in enumerate(bars):
        # Place value labels INSIDE the bar top, white text on the dark fill.
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() - 0.006,
                f"{means[i]:.3f}", ha="center", va="top", fontsize=9,
                color="white",
                fontweight="bold" if i == 0 else "normal")

    # Significance brackets vs TC-QMIX (state+types), well above error bars
    def bracket(x0, x1, y, label):
        ax.plot([x0, x0, x1, x1], [y, y + 0.003, y + 0.003, y],
                color="black", lw=0.8)
        ax.text((x0 + x1) / 2, y + 0.005, label,
                ha="center", fontsize=8.5)

    bracket(0, 1, 0.210, "p = 0.034 *")
    bracket(0, 2, 0.222, "p = 0.0012 **")

    ax.set_xticks(xs)
    ax.set_xticklabels([n for n, _, _ in variants], fontsize=8.5)
    ax.set_ylabel("coverage_rate  (mean ± 95% CI)")
    ax.set_title("TC-QMIX mixer hyper-input ablation  (n = 270 / variant)")
    ax.set_ylim(0.16, 0.240)
    ax.grid(axis="y", alpha=0.3, ls=":")
    fig.tight_layout()
    fig.savefig(OUT / "ablation.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'ablation.pdf'}")


# --------------------------------------------------------------------------
# 6. NEW: pairwise Cohen's d heatmap on MATE-4v8-9
# --------------------------------------------------------------------------

def fig_pairwise_heatmap() -> None:
    pairs = load_pairwise()
    methods = ["TCQMIX", "DQN", "QMIX", "TCQMIX_types", "VDN", "MAPPO"]
    labels  = ["TC-QMIX", "DQN", "QMIX", "TC-QMIX\n(types)", "VDN", "MAPPO"]
    n = len(methods)
    d_mat = np.full((n, n), np.nan)
    p_mat = np.full((n, n), np.nan)

    idx = {m: i for i, m in enumerate(methods)}
    for r in pairs:
        a, b = r["method_a"], r["method_b"]
        if a in idx and b in idx:
            ia, ib = idx[a], idx[b]
            d_mat[ia, ib] = float(r["cohen_d"])
            d_mat[ib, ia] = -float(r["cohen_d"])
            p_mat[ia, ib] = float(r["welch_p"])
            p_mat[ib, ia] = float(r["welch_p"])
    np.fill_diagonal(d_mat, 0.0)
    np.fill_diagonal(p_mat, 1.0)

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(d_mat, cmap="RdBu_r", vmin=-0.5, vmax=0.5)

    for i in range(n):
        for j in range(n):
            if i == j:
                txt = "—"
            else:
                d = d_mat[i, j]
                p = p_mat[i, j]
                sig = ("***" if p < 0.001 else
                       "**"  if p < 0.01  else
                       "*"   if p < 0.05  else
                       "ns")
                txt = f"d={d:+.2f}\n{sig}"
            color = "white" if abs(d_mat[i, j]) > 0.3 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=7, color=color)

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Pairwise effect size (Cohen's d) — row vs column\n"
                 "MATE-Hard-4v8-9, n = 270 per method", fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Cohen's d  (row − column)")
    fig.tight_layout()
    fig.savefig(OUT / "pairwise_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'pairwise_heatmap.pdf'}")


# --------------------------------------------------------------------------
# 7. NEW: coverage vs team-size scaling
# --------------------------------------------------------------------------

def fig_count_scaling() -> None:
    data = load_cross_seed()
    # Team size = N_cam from each scenario id
    scenarios = ["MATE-2v4-9", "MATE-4v4-9", "MATE-4v8-9", "MATE-8v8-9"]
    n_cam = [2, 4, 4, 8]

    learned = [
        ("Double-DQN",          "DQN",          "#ff7f0e", "o"),
        ("TC-QMIX (Ours)",      "TCQMIX",       "#1f77b4", "s"),
        ("QMIX",                "QMIX",         "#d62728", "^"),
        ("TC-QMIX (types)",     "TCQMIX_types", "#9467bd", "D"),
        ("VDN",                 "VDN",          "#8c564b", "v"),
        ("MAPPO",               "MAPPO",        "#e377c2", "P"),
    ]

    def get(base, sc):
        if sc == "MATE-4v8-9":
            return data[(base, sc)]
        return data[(base + "_transfer", sc)]

    fig, ax = plt.subplots(figsize=(7.4, 4.0))

    # Reference lines: MATE-Hard Rule Greedy and Random per-scenario.  The
    # learned curves are MATE-Hard evaluations, so the references must be too.
    greedy = [rule_mh(data, "greedy", s) for s in scenarios]
    rand   = [rule_mh(data, "random", s) for s in scenarios]
    ax.plot(range(len(scenarios)), greedy, "--", color="#2ca02c", lw=1.5,
            marker="*", ms=10, label="Rule Greedy, MATE-Hard (ceiling)")
    ax.plot(range(len(scenarios)), rand, ":", color="gray", lw=1.2,
            marker="x", ms=7, label="Rule Random, MATE-Hard (floor)")

    for lbl, base, col, mk in learned:
        ys = [get(base, s)[0] for s in scenarios]
        es = [1.96 * get(base, s)[1] for s in scenarios]
        ax.errorbar(range(len(scenarios)), ys, yerr=es,
                    marker=mk, ms=6, lw=1.2, color=col,
                    label=lbl, capsize=2)

    # Annotate 4v8 (in-distribution) zone
    ax.axvspan(1.5, 2.5, color="yellow", alpha=0.08)
    ax.text(2, 0.365, "training\nscenario", ha="center",
            color="#806000", fontsize=8, style="italic")

    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([f"{s.replace('MATE-','').replace('-9','')}\n(N_cam={n})"
                        for s, n in zip(scenarios, n_cam)], fontsize=8)
    ax.set_xlabel("scenario  (camera×target)")
    ax.set_ylabel("coverage_rate  (mean ± 95% CI)")
    ax.set_title("Coverage vs team size — every learned method collapses on 2v4")
    ax.set_ylim(0, 0.40)   # MATE-Hard references top out at ~0.35
    ax.grid(alpha=0.3, ls=":")
    ax.legend(loc="upper left", ncol=2, fontsize=7.5, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "count_scaling.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'count_scaling.pdf'}")


# --------------------------------------------------------------------------
# 8. NEW: per-seed scatter — shows MAPPO seed-2 outlier
# --------------------------------------------------------------------------

def fig_seed_variance() -> None:
    rows = load_long()
    methods = [
        ("Double-DQN",      "DQN",          "#ff7f0e"),
        ("TC-QMIX (Ours)",  "TCQMIX",       "#1f77b4"),
        ("QMIX",            "QMIX",         "#d62728"),
        ("TC-QMIX (types)", "TCQMIX_types", "#9467bd"),
        ("VDN",             "VDN",          "#8c564b"),
        ("MAPPO",           "MAPPO",        "#e377c2"),
    ]
    # Per-training-seed mean (we collapse 3 eval seeds into the per-train-seed cell).
    # The long CSV's "method" field is collapsed; per-seed info is in "source".
    # Use the wide CSV instead.
    per_train_seed: dict[str, list[float]] = {m: [] for _, m, _ in methods}
    with (RESULTS / "EVAL_MATRIX_wide.csv").open() as f:
        for r in csv.DictReader(f):
            if r["scenario"] != "MATE-4v8-9":
                continue
            m = r["method"]
            # method is like "DQN_seed0"; map back to the base
            if "_seed" in m and not m.endswith("_transfer"):
                base = m.rsplit("_seed", 1)[0]
                if base in per_train_seed:
                    per_train_seed[base].append(float(r["mean"]))

    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    xs = np.arange(len(methods))
    for i, (lbl, base, col) in enumerate(methods):
        ys = per_train_seed.get(base, [])
        if not ys:
            continue
        # individual seed dots
        ax.scatter([i] * len(ys), ys, s=80, color=col, edgecolor="black",
                   linewidth=0.5, zorder=3)
        # mean horizontal bar
        mean = np.mean(ys)
        ax.hlines(mean, i - 0.25, i + 0.25, color=col, lw=2.4, zorder=2)
        # range
        ax.vlines(i, np.min(ys), np.max(ys), color=col, alpha=0.35, lw=1.5, zorder=1)
        # annotate seed numbers (alternate sides to avoid label overlap)
        for j, y in enumerate(sorted(ys)):
            ax.text(i + 0.18, y, f"s{j}", fontsize=7, va="center")

    # Highlight MAPPO seed-2 collapse (annotate inside the plot, not below x-axis)
    if "MAPPO" in per_train_seed and per_train_seed["MAPPO"]:
        low = min(per_train_seed["MAPPO"])
        ax.annotate("MAPPO seed-2\ncollapse",
                    xy=(5, low), xytext=(4.0, 0.155),
                    fontsize=8.5, color="#990000", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#990000", lw=0.9))

    ax.set_xticks(xs)
    ax.set_xticklabels([n for n, _, _ in methods], fontsize=8.5)
    ax.set_ylabel("coverage_rate  (per-seed mean, 90 ep)")
    ax.set_title("Per-seed coverage on MATE-Hard-4v8-9 — MAPPO is the only high-variance method")
    ax.grid(axis="y", alpha=0.3, ls=":")
    ax.set_ylim(0.145, 0.200)
    ax.set_xlim(-0.6, len(methods) - 0.4)
    fig.tight_layout()
    fig.savefig(OUT / "seed_variance.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'seed_variance.pdf'}")


# --------------------------------------------------------------------------

if __name__ == "__main__":
    fig_env_overview()
    fig_architecture()
    fig_method_comparison()
    fig_transfer_matrix()
    fig_rule_degrade()
    fig_ablation()
    fig_pairwise_heatmap()
    fig_count_scaling()
    fig_seed_variance()
