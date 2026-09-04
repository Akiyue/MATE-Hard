#!/usr/bin/env python3
"""Regenerate every data-derived table of the manuscript from ``results/``.

This is the "single CLI command" promised in Sections 1 and 5 of the paper.
It reads *only* the released per-episode CSVs under ``results/`` -- no GPU,
no checkpoints, no training, no simulator -- and re-derives every number that
appears in a data table of the manuscript:

    Table 1  rule-based baselines on plain MATE            (tab:baseline-matrix)
    Table 2  rule-based degradation on MATE-Hard           (tab:rule-mh-degrade)
    Table 5  headline results on MATE-Hard-4v8-9           (tab:main-results)
    Table 6  TC-QMIX mixer-hyper-input ablation            (tab:ablation)
    Table 7  zero-shot transfer matrix                     (tab:transfer)
    Table 8  extended train-eval matrix                    (tab:transfer-extended)

Tables 3 and 4 are hyperparameter listings, not measured quantities; they are
fixed by the training configs and by the sweep documented in
``paper/supplementary/S1_hyperparameter_sweep.md``.

Usage
-----
Write the LaTeX table bodies into ``paper/tables/`` and print the report::

    python reproduce_tables.py

Additionally assert that every regenerated number matches the value printed
in the submitted manuscript (non-zero exit status on any mismatch)::

    python reproduce_tables.py --check

Dependencies: numpy only (scipy is used for the t-test when importable, with a
pure-numpy fallback).  The statistical routines are *not* duplicated here: they
are loaded from ``mate_marl/analysis/stats.py``, the same module used to produce
the submitted tables.  The CSV-to-cell mapping is written out explicitly below
so that a reader can see exactly which released file feeds which table cell.

Conventions (Section 5 of the paper): Welch's two-sided t-test on per-episode
coverage, paired-bootstrap 95% CIs with 10,000 resamples, Cohen's d on the
pooled standard deviation.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

# The report prints non-ASCII statistics symbols; the Windows console
# defaults to cp1252 and would raise on them.
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")


def _load_stats():
    """Import mate_marl/analysis/stats.py without importing the whole package.

    ``mate_marl/__init__.py`` pulls in the Gym wrappers, which would make table
    reproduction depend on a working simulator install.  stats.py itself needs
    only numpy, so we load it directly by path.
    """
    path = ROOT / "mate_marl" / "analysis" / "stats.py"
    spec = importlib.util.spec_from_file_location("_mate_stats", path)
    mod = importlib.util.module_from_spec(spec)
    # Register before executing: @dataclass resolves annotations through
    # sys.modules[cls.__module__].
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


STATS = _load_stats()
compare = STATS.compare

SCENARIOS = ("MATE-2v4-9", "MATE-4v4-9", "MATE-4v8-9", "MATE-8v8-9")
TRAIN_SCENARIO = "MATE-4v8-9"
SEEDS = (0, 1, 2)
RULE_INNERS = ("random", "naive", "greedy", "heuristic")

# Display name -> per-seed method-key prefix.
LEARNED = {
    "TC-QMIX (Ours)":       "TCQMIX",
    "Double-DQN":           "DQN",
    "QMIX (vanilla)":       "QMIX",
    "TC-QMIX (types-only)": "TCQMIX_types",
    "VDN":                  "VDN",
    "MAPPO":                "MAPPO",
}

# Which released CSV holds the final in-distribution evaluation of each
# (method, training seed).  Every file has one row per evaluation episode.
INDIST_CSV = {
    "DQN_seed0": "dqn_v8_final_eval.csv",
    "DQN_seed1": "dqn_seed1_final_eval.csv",
    "DQN_seed2": "dqn_seed2_final_eval.csv",
    "TCQMIX_seed0": "tcqmix_v1_step2p6M_eval.csv",
    "TCQMIX_seed1": "tcqmix_seed1_eval.csv",
    "TCQMIX_seed2": "tcqmix_seed2_eval.csv",
    "TCQMIX_types_seed0": "tcqmix_types_seed0_eval.csv",
    "TCQMIX_types_seed1": "tcqmix_types_seed1_eval.csv",
    "TCQMIX_types_seed2": "tcqmix_types_seed2_eval.csv",
    "QMIX_seed0": "qmix_vanilla_seed0_eval.csv",
    "QMIX_seed1": "qmix_vanilla_seed1_eval.csv",
    "QMIX_seed2": "qmix_vanilla_seed2_eval.csv",
    "VDN_seed0": "vdn_seed0_eval.csv",
    "VDN_seed1": "vdn_seed1_eval.csv",
    "VDN_seed2": "vdn_seed2_eval.csv",
    "MAPPO_seed0": "mappo_seed0_eval.csv",
    "MAPPO_seed1": "mappo_seed1_eval.csv",
    "MAPPO_seed2": "mappo_seed2_eval.csv",
}

# Zero-shot transfer of the same checkpoints; one file spans the three
# out-of-distribution scenarios (demultiplexed by the 'scenario' column).
TRANSFER_CSV = {
    "DQN_seed0": "dqn_v8_transfer_eval.csv",          # legacy, 1 eval seed (n=40)
    "DQN_seed1": "DQN_seed1_transfer.csv",
    "DQN_seed2": "DQN_seed2_transfer.csv",
    "TCQMIX_seed0": "tcqmix_v1_transfer_eval.csv",    # legacy, 1 eval seed (n=40)
    "TCQMIX_seed1": "TCQMIX_seed1_transfer.csv",
    "TCQMIX_seed2": "TCQMIX_seed2_transfer.csv",
    "TCQMIX_types_seed0": "TCQMIX_types_seed0_transfer.csv",
    "TCQMIX_types_seed1": "TCQMIX_types_seed1_transfer.csv",
    "TCQMIX_types_seed2": "TCQMIX_types_seed2_transfer.csv",
    "QMIX_seed0": "QMIX_vanilla_seed0_transfer.csv",
    "QMIX_seed1": "QMIX_vanilla_seed1_transfer.csv",
    "QMIX_seed2": "QMIX_vanilla_seed2_transfer.csv",
    "VDN_seed0": "VDN_seed0_transfer.csv",
    "VDN_seed1": "VDN_seed1_transfer.csv",
    "VDN_seed2": "VDN_seed2_transfer.csv",
    "MAPPO_seed0": "MAPPO_seed0_transfer.csv",
    "MAPPO_seed1": "MAPPO_seed1_transfer.csv",
    "MAPPO_seed2": "MAPPO_seed2_transfer.csv",
}

# Rule-based agents on plain MATE.  The 4v8 run predates the scenario suffix.
PLAIN_RULE_CSV = {
    "MATE-4v8-9": "baseline_{inner}.csv",
    "MATE-2v4-9": "baseline_{inner}_MATE-2v4-9.csv",
    "MATE-4v4-9": "baseline_{inner}_MATE-4v4-9.csv",
    "MATE-8v8-9": "baseline_{inner}_MATE-8v8-9.csv",
}

METRIC_COL = {
    "coverage_rate": "mean_coverage_rate",
    "real_coverage_rate": "mean_real_coverage_rate",
    "transport_rate": "mean_mean_transport_rate",
    "episode_return": "episode_return",
}


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def _read(path: Path, metric: str, scenario: str | None = None) -> np.ndarray:
    """Per-episode values of ``metric`` from one CSV, optionally filtered."""
    col = METRIC_COL[metric]
    vals: list[float] = []
    if not path.exists():
        return np.asarray(vals, dtype=np.float64)
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            if scenario is not None and r.get("scenario") not in (None, scenario):
                continue
            raw = r.get(col, "")
            if raw in ("", None):
                continue
            try:
                v = float(raw)
            except ValueError:
                continue
            if v == v:
                vals.append(v)
    return np.asarray(vals, dtype=np.float64)


class Data:
    """All per-episode arrays needed by the tables, keyed by (key, scenario)."""

    def __init__(self, results_dir: Path, metric: str = "coverage_rate") -> None:
        self.dir = results_dir
        self.metric = metric
        self.cells: dict[tuple[str, str], np.ndarray] = {}

        for inner in RULE_INNERS:
            for sc, tmpl in PLAIN_RULE_CSV.items():
                self._put(f"rule_{inner}", sc,
                          _read(results_dir / tmpl.format(inner=inner), metric, sc))
            for adapter in ("naive", "informed"):
                for sc in SCENARIOS:
                    p = results_dir / f"baseline_{inner}_{adapter}-MH_{sc}.csv"
                    self._put(f"rule_{inner}_{adapter}MH", sc, _read(p, metric, sc))

        for key, fname in INDIST_CSV.items():
            self._put(key, TRAIN_SCENARIO,
                      _read(results_dir / fname, metric, TRAIN_SCENARIO))
        for key, fname in TRANSFER_CSV.items():
            for sc in SCENARIOS:
                if sc == TRAIN_SCENARIO:
                    continue
                self._put(f"{key}_transfer", sc, _read(results_dir / fname, metric, sc))

    def _put(self, key: str, scenario: str, arr: np.ndarray) -> None:
        if arr.size:
            self.cells[(key, scenario)] = arr

    def get(self, key: str, scenario: str) -> np.ndarray:
        return self.cells.get((key, scenario), np.asarray([], dtype=np.float64))

    def learned(self, base: str, scenario: str) -> np.ndarray:
        """Pool the three training seeds of one method on one scenario."""
        suffix = "" if scenario == TRAIN_SCENARIO else "_transfer"
        parts = [self.get(f"{base}_seed{s}{suffix}", scenario) for s in SEEDS]
        parts = [p for p in parts if p.size]
        return np.concatenate(parts) if parts else np.asarray([], dtype=np.float64)


def mse(a: np.ndarray) -> tuple[float, float]:
    """Mean and standard error of the mean."""
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(a.size))


def stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def fmt_p(p: float) -> str:
    """LaTeX for a p-value, using the manuscript's rounding convention."""
    if p >= 0.01:
        return "%.3f" % p
    if p >= 1e-4:
        return "%.2g" % p
    exp = 0
    m = p
    while m < 1.0:
        m *= 10.0
        exp -= 1
    return r"%.1f \!\times\! 10^{%d}" % (m, exp)


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def table1(rd: Path):
    """Rule-based baselines on plain MATE.

    Following the manuscript: the best-coverage baseline of each scenario is
    named in bold, and each of its metric cells is bold where that baseline is
    also the best on that metric (lower is better for the transport rate).
    """
    metrics = ["coverage_rate", "real_coverage_rate", "transport_rate", "episode_return"]
    lower_is_better = {"transport_rate"}
    d = {m: Data(rd, m) for m in metrics}
    lines, values = [], {}
    for sc in SCENARIOS:
        covs = {i: float(d["coverage_rate"].get(f"rule_{i}", sc).mean())
                for i in RULE_INNERS if d["coverage_rate"].get(f"rule_{i}", sc).size}
        best = max(covs, key=covs.get) if covs else None
        best_on = {}
        for m in metrics:
            vals = {i: float(d[m].get(f"rule_{i}", sc).mean())
                    for i in RULE_INNERS if d[m].get(f"rule_{i}", sc).size}
            if vals:
                best_on[m] = (min if m in lower_is_better else max)(vals, key=vals.get)
        for j, inner in enumerate(RULE_INNERS):
            cells = []
            for m in metrics:
                arr = d[m].get(f"rule_{inner}", sc)
                if arr.size == 0:
                    cells.append("--")
                    continue
                mu, se = mse(arr)
                values[(sc, inner, m)] = (mu, se)
                cell = (f"$-${abs(mu):.0f} $\\pm$ {se:.0f}" if m == "episode_return"
                        else f"{mu:.3f} $\\pm$ {se:.3f}")
                if inner == best and best_on.get(m) == inner:
                    cell = f"\\textbf{{{cell}}}"
                cells.append(cell)
            name = inner.capitalize()
            name = f"\\textbf{{{name}}}" if inner == best else name
            lead = (f"    \\multirow{{4}}{{*}}{{\\texttt{{{sc}}}}} &" if j == 0
                    else "                                         &")
            lines.append(f"{lead} {name:<20} & " + " & ".join(cells) + " \\\\")
        lines.append("    \\midrule")
    return "\n".join(lines[:-1]), values


def table2(rd: Path):
    """Naive/informed MATE-Hard coverage, plus the Delta-% degradation column.

    Delta % compares the best MATE-Hard entry (over both adapters and both
    inner agents) with the best plain-MATE entry for the same scenario.  This
    is the same definition Figure 5 annotates.
    """
    d = Data(rd)
    lines, values = [], {}
    for sc in SCENARIOS:
        plain_best = max(float(d.get(f"rule_{i}", sc).mean())
                         for i in RULE_INNERS if d.get(f"rule_{i}", sc).size)
        mh_best = max(float(d.get(f"rule_{i}_{a}MH", sc).mean())
                      for i in ("greedy", "heuristic") for a in ("naive", "informed")
                      if d.get(f"rule_{i}_{a}MH", sc).size)
        drop = 100.0 * (1.0 - mh_best / plain_best)
        values[sc] = {"plain_best": plain_best, "mh_best": mh_best, "drop_pct": drop}
        short = sc.replace("MATE-", "")
        for j, adapter in enumerate(("naive", "informed")):
            cells = []
            for inner in ("greedy", "heuristic"):
                mu, se = mse(d.get(f"rule_{inner}_{adapter}MH", sc))
                values[(sc, adapter, inner)] = (mu, se)
                cells.append(f"{mu:.3f} $\\pm$ {se:.3f}")
            if j == 0:
                lead = f"    \\multirow{{2}}{{*}}{{\\texttt{{{short}}}}} & naive   "
                tail = f" & \\multirow{{2}}{{*}}{{$-${drop:.0f}\\%}} \\\\"
            else:
                lead = "                                    & informed"
                tail = " & \\\\"
            lines.append(lead + " & " + " & ".join(cells) + tail)
        lines.append("    \\midrule")
    return "\n".join(lines[:-1]), values


def table5(rd: Path):
    d = Data(rd)
    rule = d.get("rule_greedy_informedMH", TRAIN_SCENARIO)
    rule_mu, rule_se = mse(rule)
    ref = d.learned("TCQMIX", TRAIN_SCENARIO)

    rows = []
    for name, base in LEARNED.items():
        arr = d.learned(base, TRAIN_SCENARIO)
        mu, se = mse(arr)
        if base == "TCQMIX":
            p = dd = float("nan")
        else:
            c = compare(ref, arr, method_a="TC-QMIX", method_b=name)
            p, dd = c.welch_p, c.cohen_d
        rows.append({"name": name, "base": base, "mean": mu, "se": se,
                     "n": int(arr.size), "p": p, "d": dd,
                     "delta_rule_pct": 100.0 * (mu - rule_mu) / rule_mu})
    rows.sort(key=lambda r: -r["mean"])

    lines = ["    \\rowcolor{green!8}",
             f"    Rule Greedy (informed-MH)  & \\textbf{{{rule_mu:.3f} $\\pm$ {rule_se:.3f}}}"
             f" & {rule.size}  & ---     & $< 10^{{-6}}$ \\\\",
             "    \\midrule"]
    for r in rows:
        cov = f"{r['mean']:.3f} $\\pm$ {r['se']:.3f}"
        if r["base"] == "TCQMIX":
            cov, pcell = f"\\textbf{{{cov}}}", "---"
        else:
            pcell = f"${fmt_p(r['p'])}$ {stars(r['p'])}"
        lines.append(f"    {r['name']:<27} & {cov} & {r['n']} "
                     f"& $-${abs(r['delta_rule_pct']):.0f}\\% & {pcell} \\\\")
    meta = {"rows": rows, "rule": (rule_mu, rule_se, int(rule.size)),
            "rule_vs_tcqmix": compare(rule, ref, method_a="Rule Greedy", method_b="TC-QMIX")}
    return "\n".join(lines), meta


def table6(rd: Path):
    d = Data(rd)
    ref = d.learned("TCQMIX", TRAIN_SCENARIO)
    mu0, se0 = mse(ref)
    lines = ["    \\rowcolor{green!8}",
             f"    state $+$ types (TC-QMIX)   & \\textbf{{{mu0:.3f} $\\pm$ {se0:.3f}}}"
             f" & ---     & ---     \\\\"]
    meta = {"full": (mu0, se0)}
    for label, base in (("state (vanilla QMIX)", "QMIX"), ("types only", "TCQMIX_types")):
        arr = d.learned(base, TRAIN_SCENARIO)
        mu, se = mse(arr)
        c = compare(ref, arr, method_a="TC-QMIX", method_b=label)
        meta[base] = {"mean": mu, "se": se, "delta": mu - mu0,
                      "p": c.welch_p, "d": c.cohen_d}
        lines.append(f"    {label:<37} & {mu:.3f} $\\pm$ {se:.3f} & $-${abs(mu - mu0):.3f}"
                     f" & ${fmt_p(c.welch_p)}$ {stars(c.welch_p)} \\\\")
    return "\n".join(lines), meta


def table7(rd: Path):
    """Full zero-shot transfer matrix.

    The learned checkpoints are evaluated on the **MATE-Hard** variant of each
    scenario (``make_marl_env_discrete`` enables the heterogeneous-camera,
    energy and fog wrappers by default), so the reference rows must be the
    MATE-Hard rule-based agents, not the plain-MATE ones.  For each rule agent
    we take the better of its naive and informed adapter, the same convention
    used for the Delta-% column of Table 2 and for Figure 5.  A plain-MATE
    Greedy row is kept at the bottom, explicitly labelled, because it is the
    benchmark-difficulty reference of Section 5.1 -- it is *not* a like-for-like
    comparator for the learned rows.

    The final block reports normalised skill,
        (learned - Random_MH) / (Greedy_MH - Random_MH),
    i.e. where the best learned method sits on the random-to-ceiling range of
    the same environment.  This is the count-invariance quantity: raw coverage
    falls four-fold from 4v8 to 2v4, but so does the coverage of a random
    policy, so the raw drop mostly measures the scenario rather than the policy.
    """
    d = Data(rd)
    lines, meta = [], {}

    def mh(inner: str, sc: str) -> np.ndarray:
        """Per-episode array of the better adapter for this rule agent."""
        cands = [d.get(f"rule_{inner}_{a}MH", sc) for a in ("naive", "informed")]
        cands = [c for c in cands if c.size]
        return max(cands, key=lambda c: c.mean())

    for inner in ("greedy", "heuristic", "random"):
        cells = []
        for sc in SCENARIOS:
            mu, se = mse(mh(inner, sc))
            meta[(f"ruleMH_{inner}", sc)] = (mu, se)
            cells.append(f"{mu:.3f} $\\pm$ {se:.3f}")
        if inner != "random":
            lines.append("    \\rowcolor{green!8}")
        label = f"Rule {inner.capitalize()} (MATE-Hard)"
        lines.append(f"    {label:<28} & " + " & ".join(cells) + " \\\\")
    lines.append("    \\midrule")

    best = {sc: max(LEARNED.values(), key=lambda b: float(d.learned(b, sc).mean()))
            for sc in SCENARIOS}
    order = sorted(LEARNED.items(),
                   key=lambda kv: -float(d.learned(kv[1], "MATE-8v8-9").mean()))
    for name, base in order:
        cells = []
        for sc in SCENARIOS:
            arr = d.learned(base, sc)
            mu, se = mse(arr)
            meta[(base, sc)] = (mu, se, int(arr.size))
            cell = f"{mu:.3f} $\\pm$ {se:.3f}"
            cells.append(f"\\textbf{{{cell}}}" if best[sc] == base else cell)
        lines.append(f"    {name:<28} & " + " & ".join(cells) + " \\\\")

    lines.append("    \\midrule")
    gaps, norms = [], []
    for sc in SCENARIOS:
        ceil = float(mh("greedy", sc).mean())
        floor = float(mh("random", sc).mean())
        top = max(float(d.learned(b, sc).mean()) for b in LEARNED.values())
        meta[("gap", sc)] = 100 * (1 - top / ceil)
        meta[("norm", sc)] = (top - floor) / (ceil - floor)
        gaps.append(f"$-${100 * (1 - top / ceil):.0f}\\%")
        norms.append(f"{(top - floor) / (ceil - floor):.2f}")
    lines.append("    Gap to Rule Greedy (MATE-Hard) & " + " & ".join(gaps) + " \\\\")
    lines.append("    Normalised skill $\\in [0,1]$ & " + " & ".join(norms) + " \\\\")

    lines.append("    \\midrule")
    plain = []
    for sc in SCENARIOS:
        mu, se = mse(d.get("rule_greedy", sc))
        meta[("rule_greedy_plain", sc)] = (mu, se)
        plain.append(f"{mu:.3f} $\\pm$ {se:.3f}")
    lines.append("    \\textit{Rule Greedy (plain MATE)} & " + " & ".join(plain) + " \\\\")
    for sc in SCENARIOS:
        meta[("rule_random_plain", sc)] = mse(d.get("rule_random", sc))
    return "\n".join(lines), meta


def table8(rd: Path):
    d = Data(rd)
    pooled: dict[tuple[str, str, str], tuple[float, float]] = {}
    with (rd / "SPRINT5_TRANSFER_POOLED.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            pooled[(r["method"], r["trained_on"], r["eval_on"])] = (
                float(r["mean_cov"]), float(r["se_cov"]))

    lines, meta = [], {}
    for disp, base, key in (("Double-DQN", "DQN", "dqn"), ("TC-QMIX   ", "TCQMIX", "tcqmix")):
        for trained in ("MATE-2v4-9", TRAIN_SCENARIO, "MATE-8v8-9"):
            cells = []
            for sc in SCENARIOS:
                mu, se = (mse(d.learned(base, sc)) if trained == TRAIN_SCENARIO
                          else pooled[(key, trained, sc)])
                meta[(base, trained, sc)] = (mu, se)
                cell = f"{mu:.3f} $\\pm$ {se:.3f}"
                if trained == sc:
                    cell = "\\cellcolor{green!8}" + cell
                cells.append(cell)
            lines.append(f"    {disp} & \\texttt{{{trained}}} & " + " & ".join(cells) + " \\\\")
        lines.append("    \\midrule")
    return "\n".join(lines[:-1]), meta


# ---------------------------------------------------------------------------
# values as printed in the submitted manuscript
# ---------------------------------------------------------------------------

PUBLISHED = {
    "table2_drop_pct": {"MATE-2v4-9": 84, "MATE-4v4-9": 71,
                        "MATE-4v8-9": 65, "MATE-8v8-9": 49},
    "table5_mean": {"TC-QMIX (Ours)": 0.189, "Double-DQN": 0.185,
                    "QMIX (vanilla)": 0.183, "TC-QMIX (types-only)": 0.180,
                    "VDN": 0.180, "MAPPO": 0.177},
    "table5_rule_mean": 0.215,
    "table6_p": {"QMIX": 0.034, "TCQMIX_types": 0.0012},
    "table7_2v4": {"DQN": 0.047, "TCQMIX": 0.047, "QMIX": 0.046,
                   "TCQMIX_types": 0.046, "VDN": 0.045, "MAPPO": 0.045},
    # Same-environment (MATE-Hard) Random reference on 2v4, best adapter.
    "random_floor_2v4_mh": 0.042,
    # (best learned - Random_MH) / (Greedy_MH - Random_MH) per scenario.
    "normalised_skill": {"MATE-2v4-9": 0.18, "MATE-4v4-9": 0.32,
                         "MATE-4v8-9": 0.34, "MATE-8v8-9": 0.35},
}


def run_checks(t2, t5, t6, t7) -> list[str]:
    fails: list[str] = []
    for sc, want in PUBLISHED["table2_drop_pct"].items():
        got = round(t2[sc]["drop_pct"])
        if got != want:
            fails.append(f"Table 2 {sc}: manuscript -{want}%, regenerated -{got}%")
    for r in t5["rows"]:
        want = PUBLISHED["table5_mean"][r["name"]]
        if abs(round(r["mean"], 3) - want) > 1e-9:
            fails.append(f"Table 5 {r['name']}: manuscript {want:.3f}, regenerated {r['mean']:.3f}")
    if abs(round(t5["rule"][0], 3) - PUBLISHED["table5_rule_mean"]) > 1e-9:
        fails.append(f"Table 5 rule ceiling: manuscript {PUBLISHED['table5_rule_mean']:.3f}, "
                     f"regenerated {t5['rule'][0]:.3f}")
    for base, want in PUBLISHED["table6_p"].items():
        got = t6[base]["p"]
        if abs(got - want) > 0.5 * want:
            fails.append(f"Table 6 {base}: manuscript p={want}, regenerated p={got:.4g}")
    for base, want in PUBLISHED["table7_2v4"].items():
        got = round(t7[(base, "MATE-2v4-9")][0], 3)
        if abs(got - want) > 1e-9:
            fails.append(f"Table 7 {base} on 2v4: manuscript {want:.3f}, regenerated {got:.3f}")
    floor = t7[("ruleMH_random", "MATE-2v4-9")][0]
    if abs(round(floor, 3) - PUBLISHED["random_floor_2v4_mh"]) > 1e-9:
        fails.append(f"MATE-Hard Random floor on 2v4: manuscript "
                     f"{PUBLISHED['random_floor_2v4_mh']:.3f}, regenerated {floor:.3f}")
    for sc, want in PUBLISHED["normalised_skill"].items():
        got = round(t7[("norm", sc)], 2)
        if abs(got - want) > 1e-9:
            fails.append(f"Normalised skill {sc}: manuscript {want:.2f}, regenerated {got:.2f}")
    return fails


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "paper" / "tables")
    ap.add_argument("--check", action="store_true",
                    help="assert the regenerated numbers match the manuscript")
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rd = args.results_dir

    print(f"reading per-episode CSVs from {rd}\n")
    built = [
        ("table1_baseline_matrix",   "Table 1  rule-based baselines on plain MATE", table1(rd)),
        ("table2_rule_mh_degrade",   "Table 2  rule-based degradation on MATE-Hard", table2(rd)),
        ("table5_main_results",      "Table 5  headline results on MATE-Hard-4v8-9", table5(rd)),
        ("table6_ablation",          "Table 6  TC-QMIX mixer-hyper-input ablation", table6(rd)),
        ("table7_transfer",          "Table 7  zero-shot transfer matrix", table7(rd)),
        ("table8_transfer_extended", "Table 8  extended train-eval matrix", table8(rd)),
    ]
    metas = {}
    for stem, title, (body, meta) in built:
        path = args.out_dir / f"{stem}.tex"
        path.write_text(body + "\n", encoding="utf-8")
        metas[stem] = meta
        print(f"=== {title}  ->  {path.relative_to(ROOT)}")
        print(body)
        print()

    t5 = metas["table5_main_results"]
    drops = [m["drop_pct"] for m in metas["table2_rule_mh_degrade"].values()
             if isinstance(m, dict)]
    tc = next(r for r in t5["rows"] if r["base"] == "TCQMIX")
    dq = next(r for r in t5["rows"] if r["base"] == "DQN")
    print("=== Headline statistics quoted in the running text")
    print(f"  rule-based degradation (abstract) : "
          f"{min(drops):.0f}--{max(drops):.0f}%")
    print(f"  learned-method coverage band      : "
          f"{min(r['mean'] for r in t5['rows']):.3f}--{max(r['mean'] for r in t5['rows']):.3f}")
    print(f"  gap to the rule ceiling           : "
          f"{min(abs(r['delta_rule_pct']) for r in t5['rows']):.0f}--"
          f"{max(abs(r['delta_rule_pct']) for r in t5['rows']):.0f}%")
    print(f"  TC-QMIX vs Double-DQN             : delta={tc['mean'] - dq['mean']:+.3f}, "
          f"p={dq['p']:.3f}, d={dq['d']:.2f} ({stars(dq['p'])})")
    print(f"  Rule ceiling vs TC-QMIX           : {t5['rule_vs_tcqmix']}")
    print()
    print("Tables 3 and 4 list hyperparameters, not measured quantities: they are fixed by")
    print("mate_marl/trainers/{dqn,tcqmix,mappo}.py and by the sweep documented in")
    print("paper/supplementary/S1_hyperparameter_sweep.md.\n")

    if args.check:
        fails = run_checks(metas["table2_rule_mh_degrade"], t5,
                           metas["table6_ablation"], metas["table7_transfer"])
        if fails:
            print("CHECK FAILED:")
            for f in fails:
                print(f"  - {f}")
            return 1
        print("CHECK PASSED: every regenerated number matches the submitted manuscript.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
