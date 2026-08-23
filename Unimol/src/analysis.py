"""
Analysis and visualisation for the AIM + QM9 project.

Produces:
  1. Policy matrix heatmap  τ_ij at epochs 10, 50, 100  (AIM paper Fig 2)
  2. Gradient cosine similarity heatmap for each method
  3. Full results table with Mean Rank and Δm%
  4. Task loss curves per method

Run after all training jobs are complete:
    python analysis.py --results_dir ../results
"""

import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless rendering (safe on all platforms)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional

from data import TASK_NAMES, N_TASKS, TARGET_COLS, get_loaders
from model import UniMolMultiTask, build_model
from unimol_collate import batch_to_device
from aim_optimizer import AIMMatrixPolicy, AIMScalarPolicy, aim_intervention
from metrics import mean_rank, delta_m_percent, print_results_table


# Energetic property cluster (AIM paper Fig 2 annotation)
ENERGY_TASKS = {"U0", "U", "H", "G"}


# ---------------------------------------------------------------------------
# 1. Policy matrix heatmap
# ---------------------------------------------------------------------------

def plot_policy_matrix(
    tau:       np.ndarray,
    epoch:     int,
    save_path: Optional[str] = None,
    title_suffix: str = "",
) -> plt.Figure:
    """
    Plot the AIM matrix-policy τ_ij as a heatmap.

    Args:
        tau  : [N, N] numpy array  (diagonal is unused / masked)
        epoch: training epoch for annotation
    """
    N    = tau.shape[0]
    mask = np.eye(N, dtype=bool)   # hide diagonal

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        tau, ax=ax,
        xticklabels=TASK_NAMES,
        yticklabels=TASK_NAMES,
        cmap="RdBu_r",
        center=0.0,
        vmin=-1.0, vmax=1.0,
        mask=mask,
        annot=True, fmt=".2f",
        annot_kws={"size": 8},
        linewidths=0.4,
        cbar_kws={"label": "τ_ij (conflict threshold)"},
    )

    # Annotate energy cluster
    for i, name in enumerate(TASK_NAMES):
        if name in ENERGY_TASKS:
            ax.get_yticklabels()[i].set_color("tab:blue")
            ax.get_xticklabels()[i].set_color("tab:blue")

    ax.set_title(
        f"AIM Policy Matrix τ_ij — Epoch {epoch}"
        + (f" [{title_suffix}]" if title_suffix else ""),
        fontsize=13,
    )
    ax.set_xlabel("Task j")
    ax.set_ylabel("Task i")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def plot_policy_evolution(
    history:   List[dict],
    epochs:    List[int] = (10, 50, 100),
    save_dir:  Optional[str] = None,
):
    """
    Plot τ_ij at multiple checkpoints — replicates AIM paper Fig 2.
    Looks for "tau" key in each history entry.
    """
    for target_ep in epochs:
        # Find the closest recorded epoch
        entry = min(
            (e for e in history if "tau" in e),
            key=lambda e: abs(e["epoch"] - target_ep),
            default=None,
        )
        if entry is None:
            print(f"  No tau recorded near epoch {target_ep}, skipping.")
            continue

        tau = np.array(entry["tau"])
        if tau.ndim == 0:
            # Scalar policy — expand to matrix for uniform display
            tau = np.full((N_TASKS, N_TASKS), float(tau))

        save_path = (
            str(Path(save_dir) / f"tau_ep{entry['epoch']:04d}.png")
            if save_dir else None
        )
        plot_policy_matrix(tau, entry["epoch"], save_path=save_path)


# ---------------------------------------------------------------------------
# 2. Results table compiler
# ---------------------------------------------------------------------------

def load_best_results(results_dir: str = "../results") -> Dict[str, Dict[str, float]]:
    """
    Scan results_dir for run subdirectories, load history.json,
    and return the best (lowest mean val MAE) test results per run.
    """
    results = {}
    for run_dir in sorted(Path(results_dir).iterdir()):
        history_file = run_dir / "history.json"
        if not history_file.exists():
            continue
        with open(history_file) as f:
            history = json.load(f)
        if not history:
            continue
        best = min(history, key=lambda e: e["val_mae_mean"])
        results[run_dir.name] = best["test_per_task"]
    return results


def compile_and_print_table(
    results_dir: str = "../results",
    stl_key:     Optional[str] = None,    # name of the STL run if present
):
    """Load all runs, compute MR + Δm%, print table."""
    results = load_best_results(results_dir)
    if not results:
        print("No completed runs found in", results_dir)
        return

    stl = results.get(stl_key, None) if stl_key else None
    print_results_table(results, stl_baseline=stl, task_names=TASK_NAMES)
    return results


# ---------------------------------------------------------------------------
# 4. Loss curves
# ---------------------------------------------------------------------------

def plot_loss_curves(
    history:   List[dict],
    run_name:  str,
    save_dir:  Optional[str] = None,
):
    """Plot validation MAE over epochs for each task."""
    epochs    = [e["epoch"] for e in history]
    task_maes = {t: [] for t in TASK_NAMES}

    for entry in history:
        for t in TASK_NAMES:
            task_maes[t].append(entry["val_per_task"].get(t, float("nan")))

    fig, axes = plt.subplots(1, len(TASK_NAMES), figsize=(6 * len(TASK_NAMES), 4.5), sharex=True)
    axes = np.atleast_1d(axes)

    for i, t in enumerate(TASK_NAMES):
        axes[i].plot(epochs, task_maes[t])
        axes[i].set_title(t)
        axes[i].set_ylabel("MAE (physical units)")
        axes[i].grid(True, alpha=0.3)

    fig.suptitle(f"Validation MAE per Task — {run_name}", fontsize=14)
    plt.tight_layout()

    if save_dir:
        path = Path(save_dir) / f"{run_name}_loss_curves.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        print(f"Saved: {path}")
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="../results")
    parser.add_argument("--stl_key",     default=None,
                        help="Run name to use as STL baseline for Δm%")
    parser.add_argument("--run_name",    default=None,
                        help="Specific run to plot (policy matrix + loss curves)")
    parser.add_argument("--tau_epochs",  nargs="+", type=int, default=[10, 50, 100])
    args = parser.parse_args()

    # 1. Print results table
    all_results = compile_and_print_table(args.results_dir, args.stl_key)

    # 2. Policy matrix evolution for a specific run
    if args.run_name:
        hist_file = Path(args.results_dir) / args.run_name / "history.json"
        if hist_file.exists():
            with open(hist_file) as f:
                history = json.load(f)
            plot_dir = str(Path(args.results_dir) / args.run_name)
            plot_policy_evolution(history, epochs=args.tau_epochs, save_dir=plot_dir)
            plot_loss_curves(history, args.run_name, save_dir=plot_dir)
