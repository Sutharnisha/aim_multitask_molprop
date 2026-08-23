"""
GNN comparison plots for the original QM9 3-task run (mu, U0, U).

This plots the legacy 3-task result directories (results_lr_5e4_Seed_*),
trained before the pilot switched to the 2-task (mu, eps_LUMO) setup used
by plot.py / data.py. Kept separate so the old mu/U0/U runs stay plottable.

Produces comparison_all_n5000.png with:
  Row 1 : validation MAE curves for mu, U0, U
  Row 2 : best test MAE bar chart + AIM-Matrix τ heatmap
  Footer: Δm% summary table vs the STL baseline for mu

Run from GNN/src/:
    python plot_3Task.py
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

RESULTS_DIR_CANDIDATES = [
    Path("../results_10K_S_42_lr_5e4")
]

num = 10000
seed = 42

RUNS = {
    "LS": f"ls_n{num}_seed{seed}",
    "PCGrad": f"pcgrad_n{num}_seed{seed}",
    "AIM Scalar": f"aim_scalar_n{num}_seed{seed}",
    "AIM Matrix": f"aim_matrix_n{num}_seed{seed}",
}

STL_RUN_CANDIDATES = [
    f"stl_task0_mu_n{num}_seed{seed}",
]

COLORS = {
    "LS": "#4878CF",
    "PCGrad": "#6ACC65",
    "AIM Scalar": "#D65F5F",
    "AIM Matrix": "#B47CC7",
    "STL": "#888888",
}

TASKS      = ["mu", "U0", "U"]
TASK_UNITS = {"mu": "Debye", "U0": "eV", "U": "eV"}


# ── Helpers ─────────────────────────────────────────────────────────────────

def resolve_results_dir() -> Path:
    for candidate in RESULTS_DIR_CANDIDATES:
        if candidate.exists() and any((candidate / run).exists() for run in RUNS.values()):
            return candidate
    return Path("../results")


RESULTS_DIR = resolve_results_dir()
SAVE_PATH = RESULTS_DIR / f"comparison_all_n{num}.png"


def load_history(run: str, base: Path, required: bool = True) -> list:
    history_path = base / run / "history.json"
    if not history_path.exists():
        if required:
            raise FileNotFoundError(f"Missing history file: {history_path}")
        return []
    with open(history_path) as f:
        return json.load(f)


def best_epoch(history: list) -> dict:
    """Best checkpoint by lowest mean validation MAE."""
    if not history:
        raise ValueError("Empty history list")
    return min(history, key=lambda e: e["val_mae_mean"])


def resolve_stl_run(base: Path) -> str | None:
    for candidate in STL_RUN_CANDIDATES:
        if (base / candidate / "history.json").exists():
            return candidate

    for candidate in sorted(base.iterdir()):
        if candidate.is_dir() and candidate.name.startswith("stl_task0_mu_"):
            return candidate.name
    return None


def best_stl_test(run: str, task: str, base: Path) -> float:
    """STL best: pick the epoch that minimises validation MAE for the trained task."""
    history = load_history(run, base)
    if not history:
        raise FileNotFoundError(f"No history available for STL run: {run}")
    best = min(history, key=lambda e: e["val_per_task"][task])
    return best["test_per_task"][task]


def build_stl_test(base: Path) -> dict:
    """Only mu has a real STL baseline -- U0/U were never trained via STL,
    so those two entries stay NaN and are skipped in bar chart / footer."""
    stl_run = resolve_stl_run(base)
    if not stl_run:
        return {"mu": np.nan, "U0": np.nan, "U": np.nan}
    return {
        "mu": best_stl_test(stl_run, "mu", base),
        "U0": np.nan,
        "U":  np.nan,
    }


# ── Load experiment data ────────────────────────────────────────────────────

histories = {lbl: load_history(run, RESULTS_DIR) for lbl, run in RUNS.items()}
bests = {lbl: best_epoch(h) for lbl, h in histories.items()}
stl_test = build_stl_test(RESULTS_DIR)
stl_run = resolve_stl_run(RESULTS_DIR)
stl_hists = {"mu": load_history(stl_run, RESULTS_DIR, required=False)} if stl_run else {"mu": []}

# ── Figure layout ─────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 10))
fig.suptitle(
    f"GNN n_train={num} Seed={seed} · QM9 3-task (mu, U0, U)",
    fontsize=14, fontweight="bold", y=0.995,
)

gs = fig.add_gridspec(
    2, 6,
    hspace=0.40, wspace=0.35,
    left=0.06, right=0.97, top=0.96, bottom=0.06,
)

ax_mu = fig.add_subplot(gs[0, 0:2])
ax_U0 = fig.add_subplot(gs[0, 2:4])
ax_U  = fig.add_subplot(gs[0, 4:6])
ax_bar = fig.add_subplot(gs[1, 0:4])
ax_heat = fig.add_subplot(gs[1, 4:6])

task_axes = {"mu": ax_mu, "U0": ax_U0, "U": ax_U}

# ── Row 1: validation MAE curves ────────────────────────────────────────────

for lbl, history in histories.items():
    epochs = [e["epoch"] for e in history]
    color = COLORS[lbl]
    lw = 2.2 if "AIM" in lbl else 1.5
    ls = "-" if "AIM" in lbl else "--"
    for task in TASKS:
        vals = [e["val_per_task"][task] for e in history]
        task_axes[task].plot(
            epochs,
            vals,
            label=lbl,
            color=color,
            linewidth=lw,
            linestyle=ls,
            marker="o",
            markersize=3,
            markevery=2,
        )

hist = stl_hists["mu"]
if hist:
    epochs = [e["epoch"] for e in hist]
    vals = [e["val_per_task"]["mu"] for e in hist]
    ax_mu.plot(
        epochs,
        vals,
        label="STL",
        color=COLORS["STL"],
        linewidth=1.5,
        linestyle=":",
        marker="s",
        markersize=3,
        markevery=2,
    )

for task, ax in task_axes.items():
    ax.set_title(f"GNN Val MAE — {task} ({TASK_UNITS[task]})", fontsize=10)
    ax.set_xlabel("Epoch", fontsize=8)
    ax.set_ylabel("MAE", fontsize=8)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")
    for lbl, history in histories.items():
        best = best_epoch(history)
        ax.axvline(
            x=best["epoch"],
            color=COLORS[lbl],
            alpha=0.20,
            linewidth=1,
            linestyle=":",
        )

# ── Bar chart helper ─────────────────────────────────────────────────────────

def plot_bar_chart(ax, bests_dict, stl_vals, title_prefix):
    all_labels = list(bests_dict.keys()) + ["STL"]
    n_methods = len(all_labels)
    bar_width = 0.15
    x = np.arange(len(TASKS))

    for i, lbl in enumerate(all_labels):
        if lbl == "STL":
            vals = [stl_vals.get(t, np.nan) for t in TASKS]
            color = COLORS["STL"]
        else:
            test = bests_dict[lbl]["test_per_task"]
            vals = [test[t] for t in TASKS]
            color = COLORS[lbl]

        offset = (i - n_methods / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            vals,
            width=bar_width,
            label=lbl,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            alpha=0.88,
        )
        finite_vals = [v for v in vals if not np.isnan(v)]
        max_v = max(finite_vals) if finite_vals else 1.0
        for bar, v in zip(bars, vals):
            if np.isnan(v):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_v * 0.01,
                f"{v:.2f}" if v < 2 else f"{v:.0f}",
                ha="center",
                va="bottom",
                fontsize=6,
                rotation=0,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n({TASK_UNITS[t]})" for t in TASKS], fontsize=9)
    ax.set_ylabel("Test MAE (at best val epoch)", fontsize=8)
    ax.set_title(f"{title_prefix} — Best Test MAE: All Methods vs All Tasks", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)


# ── Bar chart and tau heatmap ───────────────────────────────────────────────

plot_bar_chart(ax_bar, bests, stl_test, f"GNN n_train={num} Seed={seed}")


def plot_tau_heatmap(ax, bests_dict):
    n = len(TASKS)
    best_aim = bests_dict["AIM Matrix"]
    tau_matrix = np.array(best_aim["tau"], dtype=float)
    if tau_matrix.ndim == 0 or tau_matrix.shape != (n, n):
        tau_matrix = np.full((n, n), float(tau_matrix.flat[0]))
    mask = np.eye(n, dtype=bool)
    sns.heatmap(
        tau_matrix,
        ax=ax,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn_r",
        center=0,
        linewidths=0.5,
        mask=mask,
        xticklabels=TASKS,
        yticklabels=TASKS,
        cbar_kws={"label": "τ (learned threshold)", "shrink": 0.75},
        annot_kws={"size": 9},
    )
    ax.set_title(f"GNN AIM-Matrix τ\n(best epoch = {best_aim['epoch']})", fontsize=10)
    ax.set_xlabel("Suppressing task j", fontsize=8)
    ax.set_ylabel("Modified task i", fontsize=8)


plot_tau_heatmap(ax_heat, bests)

# ── Save ─────────────────────────────────────────────────────────────────────

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(SAVE_PATH, dpi=150, bbox_inches="tight")
print(f"Saved: {SAVE_PATH}")
plt.close(fig)
