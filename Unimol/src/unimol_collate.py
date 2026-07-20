"""
UniMolCollator — converts a batch of QM9 PyG Data objects into the
padded tensor format expected by the Uni-Mol SE(3)-Transformer encoder.

Uni-Mol input tensors (all batch-first):
  src_tokens   : LongTensor  [B, T]        atom-type token ids (padded with PAD_IDX)
  src_coord    : FloatTensor [B, T, 3]     3-D coordinates     (padded with 0)
  src_distance : FloatTensor [B, T, T]     pairwise distances  (padded with 0)
  src_edge_type: LongTensor  [B, T, T]     atom-pair type ids  (padded with 0)

where T = max atoms in the batch + 1  (the +1 is for the [CLS] token prepended).

Atom type vocabulary:
  Uni-Mol uses a fixed dictionary of element symbols.
  Standard special tokens: [CLS]=0, [PAD]=1, [SEP]=2, [UNK]=3
  Element tokens start at 4.
  We try to load the dictionary from unimol_tools; if unavailable we fall
  back to a hard-coded QM9 subset (H, C, N, O, F — covers all QM9 molecules).

Edge type encoding:
  For each atom pair (i, j): edge_type = token_i * vocab_size + token_j
  This is the standard Uni-Mol edge-type encoding used in the original paper.
"""

import torch
import numpy as np
from typing import List, Optional


# ---------------------------------------------------------------------------
# Atom vocabulary
# ---------------------------------------------------------------------------

# Hard-coded fallback for QM9 elements (covers 100% of QM9 molecules)
# Indices match the Uni-Mol molecule dictionary (atom_list order from paper)
_QM9_ATOMIC_NUM_TO_SYMBOL = {
    1:  "H",
    5:  "B",
    6:  "C",
    7:  "N",
    8:  "O",
    9:  "F",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    33: "As",
    34: "Se",
    35: "Br",
    52: "Te",
    53: "I",
}

# Standard Uni-Mol special token indices
CLS_IDX = 0
PAD_IDX = 1
SEP_IDX = 2
UNK_IDX = 3

# Default atom list (from Uni-Mol paper / unimol_tools dictionary)
_DEFAULT_ATOM_LIST = [
    "[CLS]", "[PAD]", "[SEP]", "[UNK]",   # 0-3 special
    "H",  "He",
    "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
    "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar",
    "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe",
    "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se",
    "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo",
    "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I",  "Xe",
]
_SYMBOL_TO_IDX = {sym: i for i, sym in enumerate(_DEFAULT_ATOM_LIST)}


def _load_unimol_dictionary():
    """Try to load Uni-Mol's official dictionary from unimol_tools."""
    try:
        from unimol_tools.data.dictionary import Dictionary
        d = Dictionary.load("molecule")
        return {sym: d.index(sym) for sym in _DEFAULT_ATOM_LIST if d.index(sym) != d.unk()}
    except Exception:
        pass
    try:
        from unicore.data import Dictionary
        d = Dictionary.load("molecule")
        return {sym: d.index(sym) for sym in _DEFAULT_ATOM_LIST}
    except Exception:
        pass
    return None   # use hard-coded fallback


_DICT_CACHE: Optional[dict] = None


def get_symbol_to_idx() -> dict:
    global _DICT_CACHE
    if _DICT_CACHE is None:
        loaded = _load_unimol_dictionary()
        _DICT_CACHE = loaded if loaded else _SYMBOL_TO_IDX
    return _DICT_CACHE


def atomic_num_to_token(z: int) -> int:
    """Map an atomic number to a Uni-Mol vocabulary index."""
    symbol = _QM9_ATOMIC_NUM_TO_SYMBOL.get(z, None)
    if symbol is None:
        return UNK_IDX
    return get_symbol_to_idx().get(symbol, UNK_IDX)


def vocab_size() -> int:
    return len(get_symbol_to_idx())


# ---------------------------------------------------------------------------
# Core collation logic
# ---------------------------------------------------------------------------

def _pad_1d(tensors: List[torch.Tensor], pad_val: int = 0) -> torch.Tensor:
    """Pad a list of 1-D tensors to the same length."""
    max_len = max(t.shape[0] for t in tensors)
    out = torch.full((len(tensors), max_len), pad_val, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        out[i, :t.shape[0]] = t
    return out


def _pad_2d(tensors: List[torch.Tensor], pad_val: float = 0.0) -> torch.Tensor:
    """Pad a list of 2-D tensors [L, D] to [B, max_L, D]."""
    max_len = max(t.shape[0] for t in tensors)
    D = tensors[0].shape[1]
    out = torch.full((len(tensors), max_len, D), pad_val, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        out[i, :t.shape[0], :] = t
    return out


def _pad_3d(tensors: List[torch.Tensor], pad_val: float = 0.0) -> torch.Tensor:
    """Pad a list of 2-D square tensors [L, L] to [B, max_L, max_L]."""
    max_len = max(t.shape[0] for t in tensors)
    out = torch.full((len(tensors), max_len, max_len), pad_val, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        L = t.shape[0]
        out[i, :L, :L] = t
    return out


class UniMolCollator:
    """
    Callable collator for torch.utils.data.DataLoader.

    Converts a list of _MolData objects (from data.py's QM9Lazy dataset,
    each exposing .z, .pos, .y) into a dict of padded tensors ready for
    the Uni-Mol encoder.

    Usage:
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=32,
                            collate_fn=UniMolCollator(target_cols=list(range(11))))
    """

    def __init__(self, target_cols: List[int] = list(range(11))):
        self.target_cols = target_cols
        self._sym2idx = None   # lazy-loaded

    def _get_sym2idx(self) -> dict:
        if self._sym2idx is None:
            self._sym2idx = get_symbol_to_idx()
        return self._sym2idx

    def __call__(self, data_list) -> dict:
        """
        Args:
            data_list: list of _MolData objects

        Returns:
            dict with keys:
              src_tokens, src_coord, src_distance, src_edge_type  — encoder inputs
              targets    : FloatTensor [B, n_tasks]
              n_atoms    : LongTensor  [B]           number of real atoms (excl. CLS)
        """
        sym2idx  = self._get_sym2idx()
        voc_size = len(sym2idx)

        tokens_list    = []
        coord_list     = []
        distance_list  = []
        edge_type_list = []
        targets_list   = []
        n_atoms_list   = []

        for data in data_list:
            z   = data.z.numpy()            # [n_atoms]  atomic numbers
            pos = data.pos.numpy()          # [n_atoms, 3]
            n   = len(z)

            # ── Token ids ────────────────────────────────────────────────
            atom_tokens = np.array(
                [sym2idx.get(
                    _QM9_ATOMIC_NUM_TO_SYMBOL.get(int(zi), "[UNK]"),
                    UNK_IDX
                ) for zi in z],
                dtype=np.int64,
            )
            # Prepend [CLS] token
            tokens = np.concatenate([[CLS_IDX], atom_tokens])    # [n+1]

            # ── Coordinates (CLS gets zero coord) ─────────────────────────
            cls_coord = np.zeros((1, 3), dtype=np.float32)
            coords    = np.concatenate([cls_coord, pos], axis=0) # [n+1, 3]

            # ── Pairwise distances ────────────────────────────────────────
            diff = coords[:, None, :] - coords[None, :, :]       # [n+1, n+1, 3]
            dist = np.sqrt((diff ** 2).sum(axis=-1)).astype(np.float32)  # [n+1, n+1]

            # ── Edge type = token_i * vocab_size + token_j ────────────────
            edge_type = (
                tokens[:, None] * voc_size + tokens[None, :]
            ).astype(np.int64)                                    # [n+1, n+1]

            # ── Targets ───────────────────────────────────────────────────
            y = data.y[0, self.target_cols].numpy().astype(np.float32)  # [n_tasks]

            tokens_list.append(torch.tensor(tokens,    dtype=torch.long))
            coord_list.append(torch.tensor(coords,     dtype=torch.float32))
            distance_list.append(torch.tensor(dist,    dtype=torch.float32))
            edge_type_list.append(torch.tensor(edge_type, dtype=torch.long))
            targets_list.append(torch.tensor(y,        dtype=torch.float32))
            n_atoms_list.append(n)

        # ── Pad all sequences to max length in batch ──────────────────────
        src_tokens    = _pad_1d(tokens_list,   pad_val=PAD_IDX)     # [B, T]
        src_coord     = _pad_2d(coord_list,    pad_val=0.0)         # [B, T, 3]
        src_distance  = _pad_3d(distance_list, pad_val=0.0)         # [B, T, T]
        src_edge_type = _pad_3d(edge_type_list,pad_val=0)           # [B, T, T]
        targets       = torch.stack(targets_list, dim=0)            # [B, n_tasks]
        n_atoms       = torch.tensor(n_atoms_list, dtype=torch.long)# [B]

        return {
            "src_tokens":    src_tokens,
            "src_coord":     src_coord,
            "src_distance":  src_distance,
            "src_edge_type": src_edge_type,
            "targets":       targets,
            "n_atoms":       n_atoms,
        }


# ---------------------------------------------------------------------------
# Helper to move a collated batch to a device
# ---------------------------------------------------------------------------

def batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move all tensors in the collated batch dict to the given device."""
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
