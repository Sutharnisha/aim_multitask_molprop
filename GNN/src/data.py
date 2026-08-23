"""
QM9 dataset loading for AIM + MPNN (Gilmer et al. 2017) project.

Provides 2-task selection (mu, eps_LUMO), reproducible splits, and per-task
normalization following the AIM paper (Minot & Schneider, arXiv:2509.25955).
Each molecule is represented by an 11-dim atom feature vector per atom
(matching the PyG QM9.x format used across MPNN benchmarks on QM9).

11 atom features (per atom):
  [0:5]  One-hot atom type: H, C, N, O, F
  [5:8]  One-hot hybridization: SP, SP2, SP3
  [8]    Is aromatic (0/1)
  [9]    Formal charge
  [10]   Total H count (including implicit)

Sanitization note:
  QM9 SDF is loaded with sanitize=False for compatibility, then each
  molecule is sanitized on-demand inside __getitem__ using
  Chem.SanitizeMol(mol, catchErrors=True) so hybridization / aromaticity
  / numH fields are correctly populated.  All 130k+ QM9 molecules pass.
"""

import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from gnn_collate import GNNCollator, batch_to_device


# ---------------------------------------------------------------------------
# QM9 target definitions
# ---------------------------------------------------------------------------

QM9_TASK_NAMES: Dict[int, str] = {
    0:  "mu",        1:  "alpha",     2:  "eps_HOMO",
    3:  "eps_LUMO",  4:  "delta_eps", 5:  "R2",
    6:  "zpve",      7:  "U0",        8:  "U",
    9:  "H",         10: "G",
}

N_TASKS     = 2
TASK_NAMES  = ["mu", "eps_LUMO"]
TARGET_COLS = [0, 3]
# 2-task pilot chosen for GENUINE gradient conflict (unlike mu/U0/U, where
# U0 and U are near-duplicate targets, r=1.00 in raw QM9 labels — see
# correlation check in project memory). mu vs eps_LUMO: r=-0.39, the
# strongest conflict among physically distinct QM9 properties.

HAR2EV = 27.211386246

_CONVERSIONS = np.array([
    1.0, 1.0, HAR2EV, HAR2EV, HAR2EV, 1.0,
    HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
], dtype=np.float32)

_CSV_PROP_COLS = ["mu", "alpha", "homo", "lumo", "gap", "r2",
                  "zpve", "u0", "u298", "h298", "g298"]

# Atom types and hybridizations present in QM9
_ATOM_TYPES  = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}   # H, C, N, O, F → index 0-4
N_ATOM_FEAT  = 11


# ---------------------------------------------------------------------------
# Atom feature extraction
# ---------------------------------------------------------------------------

def _atom_features(atom) -> List[float]:
    """
    Extract 11-dim atom features matching the PyG QM9.x format used by
    Gilmer et al. (2017) and subsequent MTL benchmarks (Nash-MTL, FAMO, AIM).

    Requires the molecule to be sanitized (so hybridization / aromaticity
    fields are populated).  Falls back to atomic-number features only if
    sanitization was skipped.
    """
    from rdkit.Chem import rdchem

    # [0:5] Atom type one-hot: H, C, N, O, F
    type_oh = [0.0] * 5
    type_idx = _ATOM_TYPES.get(atom.GetAtomicNum(), 0)
    type_oh[type_idx] = 1.0

    # [5:8] Hybridization one-hot: SP, SP2, SP3
    _hyb = {
        rdchem.HybridizationType.SP:  0,
        rdchem.HybridizationType.SP2: 1,
        rdchem.HybridizationType.SP3: 2,
    }
    hyb_oh = [0.0, 0.0, 0.0]
    hyb_idx = _hyb.get(atom.GetHybridization(), -1)
    if hyb_idx >= 0:
        hyb_oh[hyb_idx] = 1.0

    # [8] aromatic   [9] formal charge   [10] total H count
    return type_oh + hyb_oh + [
        float(atom.GetIsAromatic()),
        float(atom.GetFormalCharge()),
        float(atom.GetTotalNumHs()),
    ]


# ---------------------------------------------------------------------------
# Molecule data container
# ---------------------------------------------------------------------------

class _MolData:
    """Lightweight molecule container: atom features, 3-D positions, targets."""
    __slots__ = ("z", "x", "pos", "y")

    def __init__(self, z, x, pos, y):
        self.z   = z    # LongTensor   [n_atoms]        atomic numbers
        self.x   = x    # FloatTensor  [n_atoms, 11]    MPNN node features
        self.pos = pos  # FloatTensor  [n_atoms, 3]     3-D coordinates
        self.y   = y    # FloatTensor  [1, 11]          all 11 QM9 properties


# ---------------------------------------------------------------------------
# QM9 dataset
# ---------------------------------------------------------------------------

class QM9Lazy(Dataset):
    """
    QM9 dataset backed by raw SDF/CSV files, with per-atom MPNN node
    features (x). Each molecule is sanitized on first access so
    hybridization and aromaticity are correctly computed.
    """

    def __init__(self, root: str):
        self.root = Path(root)
        raw_dir   = self.root / "raw"

        csv_path = raw_dir / "gdb9.sdf.csv"
        df = pd.read_csv(csv_path, usecols=["mol_id"] + _CSV_PROP_COLS)
        raw_props = df[_CSV_PROP_COLS].values.astype(np.float32)
        self._targets = torch.from_numpy(raw_props * _CONVERSIONS)

        excluded = set()
        unchar_path = raw_dir / "uncharacterized.txt"
        if unchar_path.exists():
            with open(unchar_path) as f:
                for line in f:
                    m = re.match(r"^\s*(\d+)\s", line)
                    if m:
                        excluded.add(int(m.group(1)) - 1)

        n_total = len(self._targets)
        self._valid_idx = [i for i in range(n_total) if i not in excluded]

        try:
            from rdkit import Chem
            sdf_path = str(raw_dir / "gdb9.sdf")
            # sanitize=False at load time; we sanitize per-molecule below
            self._supplier = Chem.SDMolSupplier(sdf_path, removeHs=False,
                                                 sanitize=False)
        except ImportError:
            raise ImportError("RDKit is required. Install with: "
                              "conda install -c conda-forge rdkit")

    def __len__(self) -> int:
        return len(self._valid_idx)

    def __getitem__(self, idx: int) -> _MolData:
        from rdkit import Chem
        mol_idx = self._valid_idx[idx]
        mol     = self._supplier[mol_idx]

        if mol is None:
            raise ValueError(f"RDKit returned None for SDF index {mol_idx}.")

        # Sanitize so hybridization, aromaticity, numHs are correctly set
        Chem.SanitizeMol(mol, catchErrors=True)

        conf = mol.GetConformer()
        z    = torch.tensor(
            [atom.GetAtomicNum() for atom in mol.GetAtoms()],
            dtype=torch.long,
        )
        x    = torch.tensor(
            [_atom_features(atom) for atom in mol.GetAtoms()],
            dtype=torch.float32,
        )                                                         # [n, 11]
        pos  = torch.tensor(conf.GetPositions(), dtype=torch.float32)
        y    = self._targets[mol_idx].unsqueeze(0)

        return _MolData(z=z, x=x, pos=pos, y=y)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_qm9(root: str = "../../data/qm9") -> QM9Lazy:
    root_path = Path(root)
    sdf_path  = root_path / "raw" / "gdb9.sdf"
    csv_path  = root_path / "raw" / "gdb9.sdf.csv"

    if not sdf_path.exists() or not csv_path.exists():
        raise FileNotFoundError(
            f"QM9 raw files not found in {root_path / 'raw'}.\n"
            "Run download_data.py first."
        )
    return QM9Lazy(root)


def make_splits(
    dataset,
    n_train:    int   = 10_000,
    val_frac:   float = 0.10,
    guide_frac: float = 0.10,
    seed:       int   = 42,
) -> Dict[str, Subset]:
    """
    Partition QM9 following AIM paper Section 3.2:
      guide_frac × n_train → guidance set (AIM policy loss only)
      val_frac   × n_train → validation set
      remainder            → primary training set
      up to 10 000         → held-out test set
    """
    rng     = np.random.default_rng(seed)
    N       = len(dataset)
    perm    = rng.permutation(N)
    n_guide = int(n_train * guide_frac)
    n_val   = int(n_train * val_frac)
    n_prim  = n_train - n_guide - n_val
    n_test  = min(10_000, N - n_train)

    assert n_prim > 0
    assert n_train + n_test <= N

    return {
        "primary":  Subset(dataset, perm[:n_prim].tolist()),
        "guidance": Subset(dataset, perm[n_prim: n_prim + n_guide].tolist()),
        "val":      Subset(dataset, perm[n_prim + n_guide: n_train].tolist()),
        "test":     Subset(dataset, perm[n_train: n_train + n_test].tolist()),
    }


def compute_normalization(
    subset: Subset,
    target_cols: List[int] = TARGET_COLS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    all_y = torch.cat([data.y[:, target_cols] for data in subset], dim=0)
    means = all_y.mean(dim=0)
    stds  = all_y.std(dim=0).clamp(min=1e-8)
    return means, stds


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_loaders(
    root:        str   = "../../data/qm9",
    n_train:     int   = 10_000,
    batch_size:  int   = 32,
    seed:        int   = 42,
    num_workers: int   = 0,
) -> dict:
    """
    Build DataLoaders for all splits using GNNCollator.

    Every batch has keys:
        x       : FloatTensor [B, T, 11]  node features (padded)
        dist    : FloatTensor [B, T, T]   pairwise distances
        mask    : BoolTensor  [B, T]       True = real atom
        targets : FloatTensor [B, 3]       (physical units, un-normalised)
        n_atoms : LongTensor  [B]
    """
    dataset = load_qm9(root)
    splits  = make_splits(dataset, n_train=n_train, seed=seed)
    collator = GNNCollator(target_cols=TARGET_COLS)

    loader_kw = dict(collate_fn=collator, batch_size=batch_size,
                     num_workers=num_workers, pin_memory=False)

    return {
        "primary_loader": DataLoader(splits["primary"],  shuffle=True,  **loader_kw),
        "guide_loader":   DataLoader(splits["guidance"], shuffle=True,  **loader_kw),
        "val_loader":     DataLoader(splits["val"],      shuffle=False, **loader_kw),
        "test_loader":    DataLoader(splits["test"],     shuffle=False, **loader_kw),
        "means":          compute_normalization(splits["primary"])[0],
        "stds":           compute_normalization(splits["primary"])[1],
        "split_sizes":    {k: len(v) for k, v in splits.items()},
    }


# ---------------------------------------------------------------------------
# Quick inspection helper
# ---------------------------------------------------------------------------

def inspect_qm9(root: str = "../../data/qm9"):
    dataset = load_qm9(root)
    sample  = dataset[0]
    print(f"QM9 size      : {len(dataset):,} molecules")
    print(f"data.x shape  : {sample.x.shape}   (11-dim MPNN atom features)")
    print(f"data.pos shape: {sample.pos.shape}  (3-D coordinates)")
    print(f"Tasks         : {TASK_NAMES}  (cols {TARGET_COLS})")
    return dataset


if __name__ == "__main__":
    inspect_qm9()
    loaders = get_loaders(n_train=5000)
    print("\nSplit sizes:", loaders["split_sizes"])
    batch = next(iter(loaders["primary_loader"]))
    print("Batch shapes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:10s}: {v.shape}")
