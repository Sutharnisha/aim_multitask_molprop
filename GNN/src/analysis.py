"""
Analysis and visualisation for AIM + MPNN (Gilmer 2017) on QM9.

Produces:
  1. Policy matrix heatmap  τ_ij at epochs 10, 50, 100  (AIM paper Fig 2)
  2. Full results table with Mean Rank and Δm%
  3. Validation MAE loss curves per method

Run after all training jobs are complete:
    python analysis.py --results_dir ../results_Adam
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional

from data import TASK_NAMES, N_TASKS
from metrics import mean_rank, delta_m_percent, print_results_table


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
        tau  : [N, N] numpy array  (diagonal masked out)
        epoch: training epoch for annotation
    """
    N    = tau.shape[0]
    mask = np.eye(N, dtype=bool)

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

    for i, name in enumerate(TASK_NAMES):
        if name in ENERGY_TASKS:
            ax.get_yticklabels()[i].set_color("tab:blue")
            ax.get_xticklabels()[i].set_color("tab:blue")

    ax.set_title(
        f"AIM Policy Matrix τ_ij — Epoch {epoch} [MPNN (Gilmer 2017)]"
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
    history:  List[dict],
    epochs:   List[int] = (10, 50, 100),
    save_dir: Optional[str] = None,
):
    """Plot τ_ij at multiple checkpoints — replicates AIM paper Fig 2."""
    for target_ep in epochs:
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


def resolve_stl_baseline(
    results: Dict[str, Dict[str, float]],
    stl_key: Optional[str] = None,
) -> Optional[Dict[str, float]]:
    """Pick an STL baseline run explicitly or infer it from run names."""
    if stl_key:
        return results.get(stl_key, None)

    stl_candidates = sorted(
        name for name in results.keys() if name.lower().startswith("stl")
    )
    if not stl_candidates:
        return None

    if len(stl_candidates) > 1:
        print(
            "Multiple STL runs found; using "
            f"'{stl_candidates[0]}'. Pass --stl_key to choose a different baseline."
        )
    else:
        print(f"Using STL baseline '{stl_candidates[0]}' for Δm%.")

    return results[stl_candidates[0]]


def compile_and_print_table(
    results_dir: str = "../results",
    stl_key:     Optional[str] = None,
):
    """Load all GNN runs, compute MR + Δm%, print table."""
    results = load_best_results(results_dir)
    if not results:
        print("No completed runs found in", results_dir)
        return

    stl = resolve_stl_baseline(results, stl_key)
    if stl_key and stl is None:
        print(f"Warning: STL baseline '{stl_key}' not found in {results_dir}; Δm% will be N/A.")
    print_results_table(results, stl_baseline=stl, task_names=TASK_NAMES)
    return results


# ---------------------------------------------------------------------------
# 3. Loss curves
# ---------------------------------------------------------------------------

def plot_loss_curves(
    history:  List[dict],
    run_name: str,
    save_dir: Optional[str] = None,
):
    """Plot validation MAE over epochs for all 3 tasks."""
    epochs    = [e["epoch"] for e in history]
    task_maes = {t: [] for t in TASK_NAMES}

    for entry in history:
        for t in TASK_NAMES:
            task_maes[t].append(entry["val_per_task"].get(t, float("nan")))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)

    for i, t in enumerate(TASK_NAMES):
        axes[i].plot(epochs, task_maes[t])
        axes[i].set_title(t)
        axes[i].set_ylabel("MAE (physical units)")
        axes[i].set_xlabel("Epoch")
        axes[i].grid(True, alpha=0.3)

    fig.suptitle(
        f"Validation MAE per Task — {run_name} [MPNN (Gilmer 2017)]", fontsize=13
    )
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
    parser.add_argument("--results_dir", default="../results_Adam")
    parser.add_argument("--stl_key",     default=None,
                        help="Run name to use as STL baseline for Δm%")
    parser.add_argument("--run_name",    default=None,
                        help="Specific run to plot (policy matrix + loss curves)")
    parser.add_argument("--tau_epochs",  nargs="+", type=int, default=[10, 50, 100])
    args = parser.parse_args()

    # Print results table
    all_results = compile_and_print_table(args.results_dir, args.stl_key)

    # Policy matrix + loss curves for a specific run
    if args.run_name:
        hist_file = Path(args.results_dir) / args.run_name / "history.json"
        if hist_file.exists():
            with open(hist_file) as f:
                history = json.load(f)
            plot_dir = str(Path(args.results_dir) / args.run_name)
            plot_policy_evolution(history, epochs=args.tau_epochs, save_dir=plot_dir)
            plot_loss_curves(history, args.run_name, save_dir=plot_dir)
