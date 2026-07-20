"""
Comprehensive comparison — n_train=5000  (Uni-Mol + GNN backbones + STL baselines).

Produces comparison_all_n5000.png with:
  Row 1 : Val MAE curves  mu / U0 / U  (Uni-Mol, 4 MTL methods + STL dotted line)
  Row 2 : Uni-Mol best Test MAE bar chart  +  Uni-Mol AIM-Matrix τ heatmap
  Row 3 : GNN     best Test MAE bar chart  +  GNN     AIM-Matrix τ heatmap
  Footer: Δm% summary table vs STL baseline

Run from src/:
    python plot_comparison.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

RESULTS_DIR     = Path("../results")
GNN_RESULTS_DIR = Path("../../GNN/results_Adam")
SAVE_PATH       = RESULTS_DIR / "comparison_all_n5000.png"

RUNS = {
    "LS":         "ls_n5000_seed42",
    "PCGrad":     "pcgrad_n5000_seed42",
    "AIM Scalar": "aim_scalar_n5000_seed42",
    "AIM Matrix": "aim_matrix_n5000_seed42",
}

STL_RUNS = {
    "mu": "stl_task0_mu_n5000_seed42",
    "U0": "stl_task1_U0_n5000_seed42",
    "U":  "stl_task2_U_n5000_seed42",
}

COLORS = {
    "LS":         "#4878CF",
    "PCGrad":     "#6ACC65",
    "AIM Scalar": "#D65F5F",
    "AIM Matrix": "#B47CC7",
    "STL":        "#888888",
}

TASKS      = ["mu", "U0", "U"]
TASK_UNITS = {"mu": "Debye", "U0": "eV", "U": "eV"}

# ── Loaders ───────────────────────────────────────────────────────────────────

def load_history(run: str, base: Path) -> list:
    with open(base / run / "history.json") as f:
        return json.load(f)


def best_epoch(history: list) -> dict:
    """Best checkpoint by lowest mean validation MAE."""
    return min(history, key=lambda e: e["val_mae_mean"])


def best_stl_test(run: str, task: str, base: Path) -> float:
    """STL best: pick epoch that minimises val MAE for the *trained* task only."""
    history = load_history(run, base)
    best = min(history, key=lambda e: e["val_per_task"][task])
    return best["test_per_task"][task]


def build_stl_test(base: Path) -> dict:
    return {task: best_stl_test(run, task, base) for task, run in STL_RUNS.items()}


# ── Load Uni-Mol data ─────────────────────────────────────────────────────────

histories = {lbl: load_history(run, RESULTS_DIR) for lbl, run in RUNS.items()}
bests     = {lbl: best_epoch(h) for lbl, h in histories.items()}
stl_test  = build_stl_test(RESULTS_DIR)
stl_hists = {task: load_history(run, RESULTS_DIR) for task, run in STL_RUNS.items()}

# ── Load GNN data ─────────────────────────────────────────────────────────────

gnn_histories = {lbl: load_history(run, GNN_RESULTS_DIR) for lbl, run in RUNS.items()}
gnn_bests     = {lbl: best_epoch(h) for lbl, h in gnn_histories.items()}
gnn_stl_test  = build_stl_test(GNN_RESULTS_DIR)
gnn_stl_hists = {task: load_history(run, GNN_RESULTS_DIR) for task, run in STL_RUNS.items()}

# ── Figure layout ─────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(20, 18))
fig.suptitle(
    "AIM vs Baselines — n_train=5000 · 20 epochs · Uni-Mol & GNN Backbones",
    fontsize=14, fontweight="bold", y=0.995,
)

gs = fig.add_gridspec(
    3, 6,
    hspace=0.52, wspace=0.42,
    left=0.06, right=0.97, top=0.96, bottom=0.11,
)

ax_mu   = fig.add_subplot(gs[0, 0:2])   # Row 1 — val MAE curves
ax_U0   = fig.add_subplot(gs[0, 2:4])
ax_U    = fig.add_subplot(gs[0, 4:6])

ax_um_bar   = fig.add_subplot(gs[1, 0:4])  # Row 2 — Uni-Mol
ax_um_heat  = fig.add_subplot(gs[1, 4:6])

ax_gnn_bar  = fig.add_subplot(gs[2, 0:4])  # Row 3 — GNN
ax_gnn_heat = fig.add_subplot(gs[2, 4:6])

task_axes = {"mu": ax_mu, "U0": ax_U0, "U": ax_U}

# ── Row 1: Uni-Mol val MAE curves ─────────────────────────────────────────────

for lbl, history in histories.items():
    epochs = [e["epoch"] for e in history]
    color  = COLORS[lbl]
    lw     = 2.2 if "AIM" in lbl else 1.5
    ls     = "-"  if "AIM" in lbl else "--"
    for task in TASKS:
        vals = [e["val_per_task"][task] for e in history]
        task_axes[task].plot(
            epochs, vals,
            label=lbl, color=color,
            linewidth=lw, linestyle=ls,
            marker="o", markersize=3, markevery=2,
        )

for task in TASKS:
    hist   = stl_hists[task]
    epochs = [e["epoch"] for e in hist]
    vals   = [e["val_per_task"][task] for e in hist]
    task_axes[task].plot(
        epochs, vals,
        label="STL", color=COLORS["STL"],
        linewidth=1.5, linestyle=":", marker="s", markersize=3, markevery=2,
    )

for task, ax in task_axes.items():
    ax.set_title(f"Uni-Mol Val MAE — {task} ({TASK_UNITS[task]})", fontsize=10)
    ax.set_xlabel("Epoch", fontsize=8)
    ax.set_ylabel("MAE", fontsize=8)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")
    for lbl, history in histories.items():
        best = best_epoch(history)
        ax.axvline(
            x=best["epoch"], color=COLORS[lbl],
            alpha=0.20, linewidth=1, linestyle=":",
        )

# ── Bar chart helper ──────────────────────────────────────────────────────────

def plot_bar_chart(ax, bests_dict, stl_vals, title_prefix):
    all_labels = list(bests_dict.keys()) + ["STL"]
    n_methods  = len(all_labels)
    bar_width  = 0.15
    x          = np.arange(len(TASKS))

    for i, lbl in enumerate(all_labels):
        if lbl == "STL":
            vals  = [stl_vals[t] for t in TASKS]
            color = COLORS["STL"]
        else:
            test  = bests_dict[lbl]["test_per_task"]
            vals  = [test[t] for t in TASKS]
            color = COLORS[lbl]

        offset = (i - n_methods / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset, vals, width=bar_width,
            label=lbl, color=color,
            edgecolor="white", linewidth=0.5, alpha=0.88,
        )
        max_v = max(vals)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_v * 0.01,
                f"{v:.2f}" if v < 2 else f"{v:.0f}",
                ha="center", va="bottom", fontsize=6, rotation=0,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n({TASK_UNITS[t]})" for t in TASKS], fontsize=9)
    ax.set_ylabel("Test MAE (at best val epoch)", fontsize=8)
    ax.set_title(f"{title_prefix} — Best Test MAE: All Methods vs All Tasks", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

# ── Row 2 & 3: Bar charts ─────────────────────────────────────────────────────

plot_bar_chart(ax_um_bar,  bests,     stl_test,     "Uni-Mol")
plot_bar_chart(ax_gnn_bar, gnn_bests, gnn_stl_test, "GNN (SchNet)")

# ── Tau heatmap helper ────────────────────────────────────────────────────────

def plot_tau_heatmap(ax, bests_dict, title_prefix):
    best_aim   = bests_dict["AIM Matrix"]
    tau_matrix = np.array(best_aim["tau"])
    if tau_matrix.ndim == 0 or tau_matrix.shape != (3, 3):
        tau_matrix = np.full((3, 3), float(tau_matrix.flat[0]))
    mask = np.eye(3, dtype=bool)
    sns.heatmap(
        tau_matrix, ax=ax,
        annot=True, fmt=".3f",
        cmap="RdYlGn_r", center=0,
        linewidths=0.5, mask=mask,
        xticklabels=TASKS, yticklabels=TASKS,
        cbar_kws={"label": "τ (learned threshold)", "shrink": 0.75},
        annot_kws={"size": 9},
    )
    ax.set_title(
        f"{title_prefix} AIM-Matrix τ\n(best epoch = {best_aim['epoch']})",
        fontsize=10,
    )
    ax.set_xlabel("Suppressing task j", fontsize=8)
    ax.set_ylabel("Modified task i",    fontsize=8)

plot_tau_heatmap(ax_um_heat,  bests,     "Uni-Mol")
plot_tau_heatmap(ax_gnn_heat, gnn_bests, "GNN")

# ── Footer: Δm% table ────────────────────────────────────────────────────────

def delta_m_pct(test_per_task: dict, stl_vals: dict) -> float:
    pcts = [
        (stl_vals[t] - test_per_task[t]) / abs(stl_vals[t]) * 100.0
        for t in TASKS
    ]
    return float(np.mean(pcts))

header = (
    f"  {'Method':<13}  {'Uni-Mol Δm%':>11}  {'GNN Δm%':>9}  |"
    f"  {'UM-mu':>7}  {'UM-U0':>7}  {'UM-U':>7}  |"
    f"  {'GNN-mu':>7}  {'GNN-U0':>7}  {'GNN-U':>7}"
)
sep = "  " + "-" * (len(header) - 2)

rows = [
    "Δm% (positive = better than STL, higher is better):",
    header, sep,
]
for lbl in RUNS:
    um_dm  = delta_m_pct(bests[lbl]["test_per_task"], stl_test)
    gnn_dm = delta_m_pct(gnn_bests[lbl]["test_per_task"], gnn_stl_test)
    um_t   = bests[lbl]["test_per_task"]
    gnn_t  = gnn_bests[lbl]["test_per_task"]
    rows.append(
        f"  {lbl:<13}  {um_dm:>+10.2f}%  {gnn_dm:>+8.2f}%  |"
        f"  {um_t['mu']:>7.3f}  {um_t['U0']:>7.1f}  {um_t['U']:>7.1f}  |"
        f"  {gnn_t['mu']:>7.3f}  {gnn_t['U0']:>7.1f}  {gnn_t['U']:>7.1f}"
    )
rows.append(sep)
rows.append(
    f"  {'STL Ref':<13}  {'0.00%':>11}  {'0.00%':>9}  |"
    f"  {stl_test['mu']:>7.3f}  {stl_test['U0']:>7.1f}  {stl_test['U']:>7.1f}  |"
    f"  {gnn_stl_test['mu']:>7.3f}  {gnn_stl_test['U0']:>7.1f}  {gnn_stl_test['U']:>7.1f}"
)

fig.text(
    0.03, 0.005,
    "\n".join(rows),
    fontsize=7.5, family="monospace",
    verticalalignment="bottom",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", alpha=0.92),
)

# ── Save ─────────────────────────────────────────────────────────────────────

fig.savefig(SAVE_PATH, dpi=150, bbox_inches="tight")
print(f"Saved: {SAVE_PATH}")
plt.close(fig)
