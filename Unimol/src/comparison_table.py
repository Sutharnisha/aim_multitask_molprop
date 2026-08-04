"""
comparison_table.py — Mean Rank + Delta-m% vs STL, aggregated over N=3 seeds
Backbones: Uni-Mol and GNN, both at lr_model=5e-4 (matched learning rate)

Outputs:
  ../results/comparison_table.png  — styled image table (one section per backbone)
  ../results/comparison_table.csv  — raw numbers (per-seed and aggregate)

Run from src/:
    python comparison_table.py

Notes on what changed from the single-seed version of this script:
  - Uses the real 3-seed folders (seeds 42, 123, 7), not a single seed=42 run.
  - Both backbones use lr_model=5e-4 (matched), not GNN's separate 5e-5 default.
  - STL's checkpoint is selected by its own trained task's (mu) validation MAE,
    not the mean across all 3 heads (two of which are untrained for STL) --
    the old selection picked epoch 1 instead of the true best epoch on GNN.
  - Delta_m% is reported on mu ONLY: STL was only ever trained for
    stl_task_idx=0 (mu); U0/U have no valid STL baseline in this project yet.
  - MR is computed over the four shared-backbone methods (LS, PCGrad,
    AIM-Scalar, AIM-Matrix) across mu/U0/U -- STL is excluded from MR since
    it only has a real value for one of the three tasks.
  - mean_rank() now uses average-of-tied-ranks (metrics.py fix), matching the
    paper's stated convention instead of the old sequential-integer ranking.
"""

import json
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from metrics import mean_rank, delta_m_percent

# ── Config ────────────────────────────────────────────────────────────────────

RESULTS_DIR = Path("../results")
SAVE_PNG    = RESULTS_DIR / "comparison_table.png"
SAVE_CSV    = RESULTS_DIR / "comparison_table.csv"

SEEDS   = [42, 123, 7]
TASKS   = ["mu", "U0", "U"]
METHODS = ["LS", "PCGrad", "AIM Scalar", "AIM Matrix"]
METHOD_KEY = {"LS": "ls", "PCGrad": "pcgrad", "AIM Scalar": "aim_scalar", "AIM Matrix": "aim_matrix"}

# Per-backbone: how to build the folder path for a given seed's result set.
BACKBONES = {
    "Uni-Mol": {
        seed: Path(f"../results_lr5e4_S_{seed}") if seed == 42 else Path(f"../results_lr_5e4_S_{seed}")
        for seed in SEEDS
    },
    "GNN": {
        seed: Path(f"../../GNN/results_lr5e4_S_{seed}")
        for seed in SEEDS
    },
}

# ── Data loading ──────────────────────────────────────────────────────────────

def load_history(path):
    with open(path / "history.json") as f:
        return json.load(f)

def best_shared_method_test(base, method_key, seed):
    h = load_history(base / f"{method_key}_n5000_seed{seed}")
    best = min(h, key=lambda e: e["val_mae_mean"])
    return best["test_per_task"]

def best_stl_test_mu(base, seed):
    """STL checkpoint selected by its OWN trained task's (mu) val MAE, not the
    3-task mean (which is dominated by noise from STL's two untrained heads)."""
    h = load_history(base / f"stl_task0_mu_n5000_seed{seed}")
    best = min(h, key=lambda e: e["val_per_task"]["mu"])
    return best["test_per_task"]["mu"]

# ── Per-backbone, per-seed computation ─────────────────────────────────────────

def compute_backbone(seed_paths):
    """Returns per-seed dm%(mu) and MR for all four methods, plus the 3-seed
    mean +/- std for each."""
    per_seed_dm = {m: [] for m in METHODS}
    per_seed_mr = {m: [] for m in METHODS}

    for seed, base in seed_paths.items():
        stl_mu = best_stl_test_mu(base, seed)

        results = {
            m: best_shared_method_test(base, METHOD_KEY[m], seed)
            for m in METHODS
        }

        # Delta_m%(mu) via metrics.py, single-task view
        dm = delta_m_percent(
            {m: {"mu": results[m]["mu"]} for m in METHODS},
            {"mu": stl_mu},
        )
        for m in METHODS:
            per_seed_dm[m].append(dm[m])

        # MR across mu/U0/U via metrics.py's (now tie-averaging) mean_rank
        mr = mean_rank({m: results[m] for m in METHODS})
        for m in METHODS:
            per_seed_mr[m].append(mr[m])

    summary = {}
    for m in METHODS:
        dm_arr, mr_arr = np.array(per_seed_dm[m]), np.array(per_seed_mr[m])
        summary[m] = {
            "dm_mean": float(dm_arr.mean()), "dm_std": float(dm_arr.std(ddof=1)),
            "mr_mean": float(mr_arr.mean()), "mr_std": float(mr_arr.std(ddof=1)),
            "dm_per_seed": per_seed_dm[m], "mr_per_seed": per_seed_mr[m],
        }
    return summary

backbone_summaries = {name: compute_backbone(paths) for name, paths in BACKBONES.items()}

# ── Terminal print + CSV export ────────────────────────────────────────────────

csv_rows = [["Backbone", "Method", "MR_mean", "MR_std", "Dm_pct_mu_mean", "Dm_pct_mu_std",
             "MR_seed42", "MR_seed123", "MR_seed7",
             "Dm_seed42", "Dm_seed123", "Dm_seed7"]]

for backbone, summary in backbone_summaries.items():
    print(f"\n{'='*72}\n  {backbone}  (lr_model=5e-4, n_train=5000, N=3 seeds: 42, 123, 7)\n{'='*72}")
    print(f"{'Method':<14}{'MR (mean+-std)':<20}{'Dm%(mu) (mean+-std)':<24}")
    for m in sorted(METHODS, key=lambda m: summary[m]["mr_mean"]):
        s = summary[m]
        print(f"{m:<14}{s['mr_mean']:.2f} +- {s['mr_std']:.2f}{'':<8}"
              f"{s['dm_mean']:+.2f} +- {s['dm_std']:.2f}")
        csv_rows.append([
            backbone, m, f"{s['mr_mean']:.3f}", f"{s['mr_std']:.3f}",
            f"{s['dm_mean']:.3f}", f"{s['dm_std']:.3f}",
            f"{s['mr_per_seed'][0]:.3f}", f"{s['mr_per_seed'][1]:.3f}", f"{s['mr_per_seed'][2]:.3f}",
            f"{s['dm_per_seed'][0]:.3f}", f"{s['dm_per_seed'][1]:.3f}", f"{s['dm_per_seed'][2]:.3f}",
        ])

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
with open(SAVE_CSV, "w", newline="") as f:
    csv.writer(f).writerows(csv_rows)

# ── PNG table ───────────────────────────────────────────────────────────────

HEADER_BG, HEADER_FG = "#2c3e50", "white"
ODD_BG, EVEN_BG = "#ffffff", "#f4f6f7"

def dm_bgcolor(val):
    if val >  15: return "#1a7a3a"
    if val >   8: return "#27ae60"
    if val >   3: return "#82e0aa"
    if val >   0: return "#d5f5e3"
    if val ==  0: return "#f2f3f4"
    if val >  -5: return "#fad7a0"
    if val > -15: return "#f1948a"
    return "#c0392b"

def dm_fgcolor(val):
    return "white" if abs(val) > 15 else "black"

def make_table(ax, backbone, summary, title):
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=6)

    ordered = sorted(METHODS, key=lambda m: summary[m]["mr_mean"])
    col_labels = ["Method", "MR (mean ± std, N=3)", r"$\Delta m\%_{\mu}$ (mean ± std, N=3)"]
    cell_text = [
        [m, f"{summary[m]['mr_mean']:.2f} ± {summary[m]['mr_std']:.2f}",
         f"{summary[m]['dm_mean']:+.2f} ± {summary[m]['dm_std']:.2f}"]
        for m in ordered
    ]

    tbl = ax.table(cellText=cell_text, colLabels=col_labels, colWidths=[0.22, 0.3, 0.36],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 2.4)

    for j in range(3):
        cell = tbl[0, j]
        cell.set_facecolor(HEADER_BG)
        cell.set_text_props(color=HEADER_FG, fontweight="bold", fontsize=9)

    for i, m in enumerate(ordered):
        base_bg = ODD_BG if i % 2 == 0 else EVEN_BG
        tbl[i + 1, 0].set_facecolor(base_bg)
        tbl[i + 1, 1].set_facecolor(base_bg)
        dm_val = summary[m]["dm_mean"]
        tbl[i + 1, 2].set_facecolor(dm_bgcolor(dm_val))
        tbl[i + 1, 2].set_text_props(color=dm_fgcolor(dm_val))
        if i == 0:
            for j in range(3):
                tbl[i + 1, j].set_text_props(fontweight="bold")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5))
fig.patch.set_facecolor("#fdfefe")
fig.suptitle(
    "Mean Rank and Δm%(μ) — N=3 seeds (42, 123, 7), lr_model=5e-4, n_train=5000",
    fontsize=12.5, fontweight="bold", y=1.02,
)
make_table(ax1, "Uni-Mol", backbone_summaries["Uni-Mol"], "Backbone: Uni-Mol")
make_table(ax2, "GNN",     backbone_summaries["GNN"],     "Backbone: GNN")

fig.text(
    0.5, -0.02,
    r"$\Delta m\%_{\mu}$ computed against STL (task-specific checkpoint selection), positive = better than STL. "
    "MR computed over the four shared-backbone methods across μ/U0/U (STL excluded, no valid U0/U baseline).",
    ha="center", fontsize=7.5, style="italic", color="#555555",
)

plt.tight_layout(pad=2.2)
fig.savefig(SAVE_PNG, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nSaved PNG : {SAVE_PNG}")
print(f"Saved CSV : {SAVE_CSV}")
plt.close()
