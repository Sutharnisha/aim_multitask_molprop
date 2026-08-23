"""
mae_method.py — Validation MAE vs. epoch, per method, seed=42 only.

Plots mu / eps_LUMO validation MAE curves across training, one line per method
(LS, PCGrad, AIM-Scalar, AIM-Matrix + STL reference on mu), for both the
Uni-Mol and GNN backbones. Single seed (42) -- no averaging, no error bars.

Produces: ../results/mae_vs_epoch_seed42.png

Run from src/:
    python mae_method.py
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

SEED = 42

BACKBONE_DIRS = {
    "Uni-Mol": Path("../results_lr5e4_S_42"),
    "GNN":     Path("../../GNN/results_lr_5e4_Seed_42"),
}

RUNS = {
    "LS":         f"ls_n5000_seed{SEED}",
    "PCGrad":     f"pcgrad_n5000_seed{SEED}",
    "AIM Scalar": f"aim_scalar_n5000_seed{SEED}",
    "AIM Matrix": f"aim_matrix_n5000_seed{SEED}",
}
# Only stl_task0 (mu) has ever been trained; eps_LUMO has no valid STL baseline.
STL_RUN = f"stl_task0_mu_n5000_seed{SEED}"

COLORS = {
    "LS":         "#4878CF",
    "PCGrad":     "#6ACC65",
    "AIM Scalar": "#D65F5F",
    "AIM Matrix": "#B47CC7",
    "STL":        "#888888",
}

TASKS      = ["mu", "eps_LUMO"]
TASK_UNITS = {"mu": "Debye", "eps_LUMO": "eV"}

SAVE_PATH = Path("../results/mae_vs_epoch_seed42.png")

# ── Loaders ───────────────────────────────────────────────────────────────────

def load_history(base: Path, run: str) -> list:
    with open(base / run / "history.json") as f:
        return json.load(f)

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(len(BACKBONE_DIRS), len(TASKS), figsize=(15, 4.2 * len(BACKBONE_DIRS)))
fig.suptitle(
    f"Validation MAE vs. Epoch — seed={SEED} only, n_train=5000, lr_model=5e-4",
    fontsize=13, fontweight="bold", y=1.01,
)

for row, (backbone, base) in enumerate(BACKBONE_DIRS.items()):
    histories = {lbl: load_history(base, run) for lbl, run in RUNS.items()}
    stl_hist  = load_history(base, STL_RUN)

    for col, task in enumerate(TASKS):
        ax = axes[row, col] if len(BACKBONE_DIRS) > 1 else axes[col]

        for lbl, history in histories.items():
            epochs = [e["epoch"] for e in history]
            vals   = [e["val_per_task"][task] for e in history]
            ax.plot(epochs, vals, label=lbl, color=COLORS[lbl], linewidth=1.5, alpha=0.9)

        # STL reference line: only meaningful on mu (the only trained STL task)
        if task == "mu":
            epochs = [e["epoch"] for e in stl_hist]
            vals   = [e["val_per_task"]["mu"] for e in stl_hist]
            ax.plot(epochs, vals, label="STL", color=COLORS["STL"],
                     linewidth=1.5, linestyle=":", alpha=0.9)

        ax.set_title(f"{backbone} — {task} ({TASK_UNITS[task]})", fontsize=10)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("Validation MAE", fontsize=8)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")

plt.tight_layout(pad=2.0)
fig.savefig(SAVE_PATH, dpi=150, bbox_inches="tight")
print(f"Saved: {SAVE_PATH}")
plt.close(fig)
