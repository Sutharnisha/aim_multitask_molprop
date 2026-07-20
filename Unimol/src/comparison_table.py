"""
comparison_table.py — Mean Rank + Delta-m% vs STL / LS / PCGrad baselines
Backbones: Uni-Mol and GNN (SchNet)

Outputs:
  ../results/comparison_table.png  — styled image table (one section per backbone)
  ../results/comparison_table.csv  — raw numbers for further analysis

Run from src/:
    python comparison_table.py
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

RESULTS_DIR     = Path("../results")
GNN_RESULTS_DIR = Path("../../GNN/results_Adam")
SAVE_PNG = RESULTS_DIR / "comparison_table.png"
SAVE_CSV = RESULTS_DIR / "comparison_table.csv"

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

TASKS     = ["mu", "U0", "U"]
METHODS   = list(RUNS.keys())
ROW_ORDER = ["STL"] + METHODS   # display order in table

# ── Data loading ──────────────────────────────────────────────────────────────

def load_history(run, base):
    with open(base / run / "history.json") as f:
        return json.load(f)

def best_epoch(h):
    return min(h, key=lambda e: e["val_mae_mean"])

def stl_best_test(run, task, base):
    h    = load_history(run, base)
    best = min(h, key=lambda e: e["val_per_task"][task])
    return best["test_per_task"][task]

def build_data(base):
    """Return (results_dict, stl_dict) for the given results directory."""
    results = {
        lbl: best_epoch(load_history(run, base))["test_per_task"]
        for lbl, run in RUNS.items()
    }
    stl = {task: stl_best_test(run, task, base) for task, run in STL_RUNS.items()}
    return results, stl

um_results,  um_stl  = build_data(RESULTS_DIR)
gnn_results, gnn_stl = build_data(GNN_RESULTS_DIR)

# ── Metric computation ────────────────────────────────────────────────────────

def compute_metrics(results, stl):
    """
    Include STL as a 5th participant when ranking (MR reflects all 5 methods).
    Returns (all_results, mr, dm_stl, dm_ls, dm_pcgrad).
    """
    all_r     = {"STL": stl, **results}
    mr        = mean_rank(all_r)
    dm_stl    = delta_m_percent(all_r, stl)
    dm_ls     = delta_m_percent(all_r, results["LS"])
    dm_pcgrad = delta_m_percent(all_r, results["PCGrad"])
    return all_r, mr, dm_stl, dm_ls, dm_pcgrad

um_all,  um_mr,  um_dm_stl,  um_dm_ls,  um_dm_pcgrad  = compute_metrics(um_results,  um_stl)
gnn_all, gnn_mr, gnn_dm_stl, gnn_dm_ls, gnn_dm_pcgrad = compute_metrics(gnn_results, gnn_stl)

# ── Row builder ───────────────────────────────────────────────────────────────
# Columns: method | MR | dm%(STL) | dm%(LS) | dm%(PCGrad) | mu | U0 | U

def build_rows(all_r, mr, dm_stl, dm_ls, dm_pcgrad):
    rows = []
    for m in ROW_ORDER:
        test = all_r[m]
        rows.append([
            m,
            mr[m],
            dm_stl[m],
            dm_ls[m],
            dm_pcgrad[m],
            test["mu"],
            test["U0"],
            test["U"],
        ])
    return rows

um_rows  = build_rows(um_all,  um_mr,  um_dm_stl,  um_dm_ls,  um_dm_pcgrad)
gnn_rows = build_rows(gnn_all, gnn_mr, gnn_dm_stl, gnn_dm_ls, gnn_dm_pcgrad)

# ── Terminal print ────────────────────────────────────────────────────────────

def fmt(val, col):
    if isinstance(val, str): return val
    if col == 1:           return f"{val:.2f}"
    if 2 <= col <= 4:      return f"{val:+.2f}%"
    if col == 5:           return f"{val:.4f}"
    return f"{val:.2f}"

COL_W = [13, 6, 12, 12, 14, 10, 10, 10]
COL_LABELS = ["Method", "MR",
              "Dm%(STL)", "Dm%(LS)", "Dm%(PCGrad)",
              "mu(D)", "U0(eV)", "U(eV)"]

def print_table(title, rows):
    sep = "-" * (sum(COL_W) + 2 * len(COL_W))
    hdr = "  ".join(f"{h:<{w}}" for h, w in zip(COL_LABELS, COL_W))
    print(f"\n{'='*len(hdr)}")
    print(f"  {title}")
    print(f"{'='*len(hdr)}")
    print(hdr)
    print(sep)
    for row in rows:
        cells = [fmt(row[i], i) for i in range(len(row))]
        print("  ".join(f"{c:<{w}}" for c, w in zip(cells, COL_W)))
    print()

print_table("Uni-Mol Backbone  (n_train=5000, seed=42)", um_rows)
print_table("GNN / SchNet Backbone  (n_train=5000, seed=42)", gnn_rows)

# ── CSV export ────────────────────────────────────────────────────────────────

csv_header = ["Backbone", "Method", "MR",
              "Dm_pct_STL", "Dm_pct_LS", "Dm_pct_PCGrad",
              "test_mu", "test_U0", "test_U"]

with open(SAVE_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(csv_header)
    for row in um_rows:
        w.writerow(["UniMol", *[fmt(row[i], i) for i in range(len(row))]])
    for row in gnn_rows:
        w.writerow(["GNN", *[fmt(row[i], i) for i in range(len(row))]])

# ── PNG table helpers ─────────────────────────────────────────────────────────

HEADER_BG   = "#2c3e50"
HEADER_FG   = "white"
STL_ROW_BG  = "#fef9e7"   # soft yellow  — reference row
ODD_BG      = "#ffffff"
EVEN_BG     = "#f4f6f7"

COL_LABELS_PNG = [
    "Method", "MR",
    r"$\Delta m\%$" + "\n(vs STL)",
    r"$\Delta m\%$" + "\n(vs LS)",
    r"$\Delta m\%$" + "\n(vs PCGrad)",
    "mu\n(Debye)", "U0\n(eV)", "U\n(eV)",
]

def dm_bgcolor(val):
    """Green for positive (better than baseline), red for negative (worse)."""
    if np.isnan(val):  return "#ffffff"
    if val >  15:      return "#1a7a3a"
    if val >   8:      return "#27ae60"
    if val >   3:      return "#82e0aa"
    if val >   0:      return "#d5f5e3"
    if val == 0.0:     return "#f2f3f4"
    if val >  -5:      return "#fad7a0"
    if val > -15:      return "#f1948a"
    return "#c0392b"

def dm_fgcolor(val):
    if np.isnan(val): return "black"
    if val > 15 or val < -15: return "white"
    return "black"

def make_table(ax, rows, title):
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=6)

    n_cols = len(COL_LABELS_PNG)
    n_rows = len(rows)

    # Column widths (relative, summing to 1)
    col_widths = [0.18, 0.06, 0.13, 0.12, 0.14, 0.12, 0.12, 0.12]

    def fmt_png(val, col):
        if isinstance(val, str): return val
        if col == 1:          return f"{val:.2f}"
        if 2 <= col <= 4:     return f"{val:+.2f}%"
        if col == 5:          return f"{val:.4f}"
        return f"{val:.1f}"

    cell_text = [[fmt_png(row[i], i) for i in range(n_cols)] for row in rows]

    tbl = ax.table(
        cellText=cell_text,
        colLabels=COL_LABELS_PNG,
        colWidths=col_widths,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2.2)

    # ── Header row styling ────────────────────────────────────────────────
    for j in range(n_cols):
        cell = tbl[0, j]
        cell.set_facecolor(HEADER_BG)
        cell.set_text_props(color=HEADER_FG, fontweight="bold", fontsize=8.5)
        cell.set_edgecolor("#4a4a4a")

    # ── Data row styling ──────────────────────────────────────────────────
    for i, row in enumerate(rows):
        method = row[0]
        is_stl = (method == "STL")

        for j in range(n_cols):
            cell = tbl[i + 1, j]
            cell.set_edgecolor("#d5d8dc")

            # Base row color
            base_bg = STL_ROW_BG if is_stl else (ODD_BG if i % 2 == 0 else EVEN_BG)

            if 2 <= j <= 4:
                val = row[j]
                cell.set_facecolor(dm_bgcolor(val))
                cell.set_text_props(color=dm_fgcolor(val), fontsize=9)
            else:
                cell.set_facecolor(base_bg)
                cell.set_text_props(fontsize=9)

            # Italic method name for STL reference row
            if j == 0 and is_stl:
                cell.set_text_props(fontstyle="italic", color="#7d6608", fontsize=9)

    # ── Bold best value in each numeric column ────────────────────────────
    for j in range(1, n_cols):
        vals = [row[j] for row in rows]
        # Dm% columns (2-4): higher = better. MR and MAE columns: lower = better.
        best_i = int(np.argmax(vals)) if 2 <= j <= 4 else int(np.argmin(vals))
        tbl[best_i + 1, j].set_text_props(fontweight="bold")

# ── Build PNG figure ──────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
fig.patch.set_facecolor("#fdfefe")
fig.suptitle(
    "Mean Rank and Δm% Comparison Table  |  n_train=5000, seed=42",
    fontsize=13, fontweight="bold", y=1.01,
)

make_table(ax1, um_rows,  "Backbone: Uni-Mol   (MR ranked over 5 methods: STL, LS, PCGrad, AIM-Scalar, AIM-Matrix)")
make_table(ax2, gnn_rows, "Backbone: GNN / SchNet   (MR ranked over 5 methods: STL, LS, PCGrad, AIM-Scalar, AIM-Matrix)")

# ── Color legend ──────────────────────────────────────────────────────────────

legend_text = (
    "Dm% color key:  "
    "dark-green > 8%  |  green 3% to 8%  |  light-green 0% to 3%  "
    "(positive = better than baseline)  |  "
    "light-red -5% to 0%  |  red < -5%  (negative = worse than baseline)"
)
fig.text(
    0.5, -0.01, legend_text,
    ha="center", fontsize=7.5, style="italic", color="#555555",
)

plt.tight_layout(pad=2.5)
fig.savefig(SAVE_PNG, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved PNG : {SAVE_PNG}")
print(f"Saved CSV : {SAVE_CSV}")
plt.close()
