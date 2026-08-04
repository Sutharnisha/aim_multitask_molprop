# AIM: Multi-Task Molecular Property Prediction

Two implementations of **AIM** (Adaptive Intervention for Multi-task Learning) for molecular property prediction on QM9, sharing the same algorithm, training protocol, and evaluation code, but with different encoder backbones:

- **[`GNN/`](GNN/)** — an MPNN (Gilmer et al., 2017) trained from scratch.
- **[`Unimol/`](Unimol/)** — a pretrained Uni-Mol SE(3)-Transformer, fine-tuned.


> **Paper:** Minot & Schneider — *AIM: Adaptive Intervention for Deep Multi-task Learning of Molecular Properties*, NeurIPS AI4Science 2025. [arXiv:2509.25955](https://arxiv.org/abs/2509.25955)
> **Paper:** Minot & Schneider — *AIM: Adaptive Intervention for Deep Multi-task Learning of Molecular Properties*, NeurIPS AI4Science 2025. [arXiv:2509.25955](https://arxiv.org/abs/2509.25955)

This repository is an independent extension by Nisha Suthar, not affiliated with the original authors.

The paper itself only ever benchmarks one GNN backbone; testing AIM with a second, unrelated architecture (Uni-Mol) is this repo's own extension of the paper's suggested future work...

The paper itself only ever benchmarks one GNN backbone; testing AIM with a second, unrelated architecture (Uni-Mol) is this repo's own extension of the paper's suggested future work — *"evaluating AIM's generalizability across diverse model architectures beyond the GNN used here."* To make that a fair test, every hyperparameter that isn't intrinsically tied to a specific encoder (epochs, batch size, learning rates, policy temperature, loss weights, task-head width, etc.) is kept **identical** between the two pipelines — see [Training](#training) below. Only the encoder itself differs.

---

## What AIM does

Training one encoder on several molecular property targets at once runs into gradient conflicts: tasks pull the shared parameters in different directions. Static heuristics like PCGrad resolve this with a fixed geometric rule. AIM instead **learns** a per-task-pair policy, a threshold τ per task pair — that decides how much of each task's conflicting gradient component to project out, trained jointly with the main model using a guidance loss plus two differentiable regularizers (magnitude preservation, progress-on-hard-tasks). Both backbones here compare AIM (scalar and matrix policy variants) against linear scalarization (LS), PCGrad, and single-task learning (STL) baselines on a QM9 task subset.

> **Note:** `GNN/` and `Unimol/` currently use *different* task subsets. `Unimol/` still uses the original 3-task pilot: dipole moment (`mu`), internal energy at 0 K (`U0`), internal energy at 298 K (`U`) — but `U0`/`U` are near-duplicate targets (r=1.00 in the raw QM9 labels, differing only by a small thermal correction), so there's almost no real gradient conflict for AIM/PCGrad to resolve on that pair, which let plain LS win in practice. `GNN/` was switched to a 2-task pilot instead: `mu` + `eps_LUMO` (r=-0.39), the strongest conflict among physically distinct QM9 properties, to actually exercise gradient-conflict resolution before scaling to the paper's full 11-task QM9 setup.

Both pipelines assume a normal single-GPU machine — there is no reduced-memory or CPU-offload mode; per-task gradients are computed from one shared forward pass via `torch.autograd.grad` and combined directly on-device.

---

## Status

Training pipelines for both backbones are implemented and validated end-to-end 
(data loading → training → checkpointing → analysis). Full method × seed 
benchmark runs are in progress; results and comparison plots will be added here 
once complete.
---

## Repository layout

```
├── GNN/
│   ├── src/                 # training pipeline (see below)
│   └── results_Adam/        # trained runs: <method>_n<N>_seed<S>/{best_model.pt, history.json}
├── Unimol/
│   ├── src/                 # training pipeline (see below)
│   └── results/             # trained runs: <method>_n<N>_seed<S>/{best_model.pt, history.json}
└── data/qm9/raw/             # shared QM9 data (gdb9.sdf, gdb9.sdf.csv, uncharacterized.txt)
```

`GNN/src/` and `Unimol/src/` mirror each other file-for-file wherever the logic isn't backbone-specific:

| File | Role | Backbone-specific? |
|---|---|---|
| `train.py` | Training loop + CLI entry point | Yes — different `build_model` call, otherwise identical structure |
| `model.py` | Encoder + multi-task heads | Yes — MPNN vs. Uni-Mol |
| `data.py` | QM9 loading, splits, normalization | Path defaults only |
| `gnn_collate.py` / `unimol_collate.py` | Batch collation (padded tensors) | Yes — different input formats |
| `aim_optimizer.py` | AIM policy + gradient intervention (Eq. 1–6) | **No — byte-identical in both** |
| `baselines.py` | LS / PCGrad gradient combination | **No — byte-identical in both** |
| `metrics.py` | Mean Rank, Δm% | **No — byte-identical in both** |
| `analysis.py` | Results table + policy-matrix/loss plots for that backbone alone | No (same pattern, GNN's is the maintained one — see note below) |


Uni-Mol's `src/` additionally has `download_data.py` (QM9 + pretrained-weight setup/smoke test), `run_experiments.py` (batch launcher for the full method × subset × seed matrix), `environment.yml`, and the cross-backbone comparison tools `comparison_table.py` / `plot_comparison.py` (these read both `Unimol/results/` and `GNN/results_Adam/` to compare methods across both backbones side by side).

> **Note:** `Unimol/src/analysis.py` is a stale leftover from before this repo was split into `GNN/`+`Unimol/` — it's superseded by `comparison_table.py`/`plot_comparison.py`, has unused imports, and its `plot_loss_curves`  assumes the project 3-task setup. It isn't part of the intended pipeline; flagging it here rather than silently documenting it as a real feature.

---

## Installation

**GNN** (from `GNN/src/`):
```bash
conda create -n aim-gnn python=3.9
conda activate aim-gnn
conda install pytorch=2.1 torchvision=0.16 -c pytorch
conda install -c conda-forge rdkit=2023.9
pip install tqdm>=4.65 matplotlib>=3.7 seaborn>=0.12 pandas>=1.5 numpy>=1.24
```

**Uni-Mol** (from `Unimol/src/`):
```bash
conda env create -f environment.yml
conda activate aim_unimol
```
or manually:
```bash
conda create -n aim_unimol python=3.10
conda activate aim_unimol
conda install -c conda-forge rdkit
pip install torch unimol-tools tqdm matplotlib seaborn pandas numpy
```
Pretrained Uni-Mol weights (~700 MB) download automatically via `unimol_tools` on first run, cached in `~/.unimol/`.

---

## Data

This repo does **not** ship a `data/` folder — you need to fetch QM9 yourself and place the raw files at:

```
data/qm9/raw/gdb9.sdf
data/qm9/raw/gdb9.sdf.csv
data/qm9/raw/uncharacterized.txt   # optional exclusion list
```

(`--data_root` defaults to `../../data/qm9` for GNN and `../data/qm9` for Uni-Mol — both resolve to this same top-level `data/qm9/` when run from their respective `src/` directories.)

**Easiest way — let PyTorch Geometric fetch it for you:**
```bash
pip install torch_geometric
python -c "from torch_geometric.datasets import QM9; QM9(root='data/qm9')"
```
This downloads and extracts `gdb9.sdf`, `gdb9.sdf.csv`, and `uncharacterized.txt` into `data/qm9/raw/` automatically — no manual unpacking needed. `torch_geometric` itself isn't otherwise a dependency of this project; it's only used here as a convenient downloader.

**Manual alternative:** grab the same three files from [quantum-machine.org/datasets](http://quantum-machine.org/datasets/) or the [Figshare QM9 collection](https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904) and place them under `data/qm9/raw/` yourself.

> **Note:** despite its name, `Unimol/src/download_data.py` does not fetch QM9 — it only *verifies* the raw files already exist at `../data/qm9/raw/` (and triggers the separate pretrained Uni-Mol weight download). Run it after the data is in place, from `Unimol/src/`, as an end-to-end smoke test.

---

## Training

Run from `GNN/src/` or `Unimol/src/` respectively — the commands are identical except for the working directory:

```bash
python train.py --method aim_matrix --n_train 5000 --n_epochs 300   # AIM, matrix policy
python train.py --method aim_scalar --n_train 5000 --n_epochs 300   # AIM, scalar policy
python train.py --method ls         --n_train 5000 --n_epochs 300   # linear scalarization
python train.py --method pcgrad     --n_train 5000 --n_epochs 300   # PCGrad
python train.py --method stl --stl_task_idx 0 --n_train 5000 --n_epochs 300   # STL: trains ONE task only
                                                                                # GNN task indices:    0=mu, 1=eps_LUMO
                                                                                # Unimol task indices: 0=mu, 1=U0, 2=U
```

**Key arguments — identical defaults on both backbones:**

| Argument | Default | Description |
|---|---|---|
| `--method` | `ls` | `ls` / `pcgrad` / `aim_scalar` / `aim_matrix` / `stl` |
| `--n_train` | `5000` | Training set size |
| `--n_epochs` | `300` | Number of epochs |
| `--batch_size` | `32` | Batch size |
| `--head_hidden` | `64` | Task-head hidden width |
| `--trainable_layers` | `2` | `2` = freeze all layers, except last 2 trainable  |
| `--lr_model` | `5e-5` | Encoder + head learning rate |
| `--lr_policy` | `1e-3` | AIM policy learning rate |
| `--lambda_g` | `1.0` | Policy loss — guidance weight |
| `--lambda_m` | `0.01` | Policy loss — magnitude weight |
| `--lambda_p` | `0.08` | Policy loss — progress weight |
| `--k` | `10.0` | Policy sigmoid temperature |

**The only things that legitimately differ between backbones:**

| Argument | GNN default | Uni-Mol default | Why it differs |
|---|---|---|---|
| `--data_root` | `../../data/qm9` | `../data/qm9` | Directory depth from each `src/` |
| `--save_dir` | `../results_Adam` | `../results` | Separate result trees per backbone |
| `--freeze_backbone` | *(not applicable)* | off | Only Uni-Mol has pretrained weights to optionally freeze |

Checkpoints (`best_model.pt`) and full per-epoch history (`history.json`) are saved to `<save_dir>/<run_name>/`.

> **STL caveat:** `--method stl` only trains the single head named by `--stl_task_idx` — the other two heads in that run's `history.json` were never trained and their MAE columns are meaningless noise. To get all three properties' STL baselines, run it three times with `--stl_task_idx 0`, `1`, `2`. Any of the other four methods (`ls`/`pcgrad`/`aim_scalar`/`aim_matrix`) train all three tasks together in a single run.

To launch the full method × subset-size × seed matrix in one go (Uni-Mol only), see `Unimol/src/run_experiments.py`.

---

## Analysis & plotting

**Per-backbone** (run from that backbone's `src/`):
```bash
python analysis.py --results_dir ../results_Adam   # GNN: results table + policy heatmap + loss curves
python plot_stl_result.py                            # either backbone: plots every stl_task*_... run found
```

**Cross-backbone** (Uni-Mol's `src/` only, reads both `Unimol/results/` and `../../GNN/results_Adam/`):
```bash
python comparison_table.py           # Mean Rank + Δm% table, Uni-Mol and GNN side by side (PNG + CSV)
python plot_comparison.py            # combined val-MAE curves, bar charts, τ heatmaps
python plot_aim_matrix_unimol.py     # AIM-Matrix diagnostics for the Uni-Mol run alone
```

Both `analysis.py` and `comparison_table.py`/`plot_comparison.py` use the same Δm% convention (positive = better than STL) and the same Mean Rank definition, from the shared `metrics.py`.

---

## Learning resources

- **[`Unimol/src/explanation_train.md`](Unimol/src/explanation_train.md)** — a beginner-oriented, section-by-section walkthrough of `train.py`: reading order, what each function does, the `torch.autograd.grad`/`retain_graph` mechanics behind per-task gradient collection, a worked example trace, and a glossary. Written for the Uni-Mol `train.py`, but the same structure applies to `GNN/src/train.py`.

---

## Results structure

```
GNN/results_Adam/<method>_n<N>_seed<S>/
├── best_model.pt      # checkpoint at the best validation-MAE epoch
└── history.json        # per-epoch: val/test MAE per task, train loss, AIM policy losses + τ (if applicable)

Unimol/results/<method>_n<N>_seed<S>/
├── best_model.pt
└── history.json
```

`<method>` is one of `ls`, `pcgrad`, `aim_scalar`, `aim_matrix`, or `stl_task<i>_<name>`. `<N>` is `--n_train`, `<S>` is `--seed`.

---


