"""
Training loop — AIM + MPNN (Gilmer 2017) on QM9  (multi-task molecular property prediction).

Implements the AIM algorithm (Minot & Schneider, NeurIPS AI4Science 2025; arXiv:2509.25955)
with an MPNN backbone trained from scratch on QM9.

Algorithm (AIM paper Section 3.2 + Appendix A.1):
────────────────────────────────────────────────────────────────────
For each epoch:
  For each primary batch:

  [MODEL UPDATE]
    1. Forward: MPNN encoder → [B, 128] → 3 heads → per-task losses L_i(θ)
    2. Per-task gradients: g_i = ∇_{θ_encoder} L_i   [stop-grad for policy]
    3. AIM intervention:  g_intervened = AIM(g_1…g_N ; τ)      [Eq 1-3]
    4. Override encoder grads with g_intervened; run backward for head grads
    5. optimizer_model.step()

  [POLICY UPDATE  — on guidance batch]
    6. Compute g_i_guide on guidance batch
    7. Re-run AIM intervention (differentiable w.r.t. τ)
    8. L_policy = λ_g·L_guide + λ_m·L_magnitude + λ_p·L_progress    [Eq 4-6]
    9. optimizer_policy.step()

Per-task gradients (steps 2 and 6) come from one shared encoder forward pass:
each task's loss is backed out against the shared params with
torch.autograd.grad(retain_graph=True), so the encoder only runs once per batch.

Training protocol notes:
    - Main optimizer uses ReduceLROnPlateau on validation MAE.
    - AIM policy optimizer uses CosineAnnealingWarmRestarts with paper defaults.
    - Dataset splits are unchanged.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import itertools
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional
import argparse

from data import get_loaders, TASK_NAMES, N_TASKS, TARGET_COLS
from model import MPNNMultiTask, build_model
from gnn_collate import batch_to_device
from aim_optimizer import (
    AIMScalarPolicy, AIMMatrixPolicy, AIMPolicyLoss,
    flatten_grads, set_grad_from_flat,
)
from baselines import combine_gradients, BASELINE_METHODS as _BASELINE_METHODS
from metrics import print_results_table

BASELINE_METHODS = list(_BASELINE_METHODS) + ["stl"]


# ---------------------------------------------------------------------------
# Per-task gradient collection (one shared forward, N autograd.grad calls)
# ---------------------------------------------------------------------------

def _per_task_grads(
    model:  MPNNMultiTask,
    batch:  dict,
    means:  torch.Tensor,
    stds:   torch.Tensor,
    device: torch.device,
) -> tuple:
    """
    Compute per-task flat encoder gradients from a single shared forward pass.

    Each task's loss is differentiated against the shared params directly
    (torch.autograd.grad), so the encoder runs once per batch instead of once
    per task.

    Returns:
        flat_grads    : List[Tensor[d]]  one flat grad per task (GPU, fp32)
        task_loss_vals: List[float]      per-task loss values (logging)
    """
    shared_params = model.shared_params()
    batch = batch_to_device(batch, device)
    targets_norm  = (batch["targets"] - means) / stds   # [B, 3]

    h = model.shared_forward(batch)                     # [B, 2*hidden_dim]

    flat_grads     = []
    task_loss_vals = []

    for i in range(N_TASKS):
        loss_i = F.l1_loss(model.task_forward(h, i), targets_norm[:, i])
        grads_i = torch.autograd.grad(
            loss_i, shared_params, retain_graph=(i < N_TASKS - 1)
        )
        task_loss_vals.append(loss_i.item())
        flat_grads.append(flatten_grads(grads_i, shared_params))

    return flat_grads, task_loss_vals


def _fill_head_grads_and_override_encoder(
    model:      MPNNMultiTask,
    batch:      dict,
    means:      torch.Tensor,
    stds:       torch.Tensor,
    device:     torch.device,
    g_encoder:  torch.Tensor,
    optimizer:  torch.optim.Optimizer,
) -> None:
    """
    One forward+backward on mean loss to fill head gradients,
    then override the encoder gradient with g_encoder.
    """
    shared_params = model.shared_params()
    batch = batch_to_device(batch, device)
    targets_norm = (batch["targets"] - means) / stds

    optimizer.zero_grad()
    h     = model.shared_forward(batch)
    preds = torch.stack([model.task_forward(h, i) for i in range(N_TASKS)], dim=1)
    loss  = F.l1_loss(preds, targets_norm)
    loss.backward()

    set_grad_from_flat(g_encoder, shared_params)
    optimizer.step()


# ---------------------------------------------------------------------------
# STL step (single-task learning)
# ---------------------------------------------------------------------------

def _step_stl(
    model:     MPNNMultiTask,
    optimizer: torch.optim.Optimizer,
    batch:     dict,
    means:     torch.Tensor,
    stds:      torch.Tensor,
    device:    torch.device,
    task_idx:  int,
) -> dict:
    """Single-task learning: train only on task_idx, encoder gets only that task's gradient."""
    batch = batch_to_device(batch, device)
    targets_norm = (batch["targets"] - means) / stds

    optimizer.zero_grad()
    h    = model.shared_forward(batch)
    loss = F.l1_loss(model.task_forward(h, task_idx), targets_norm[:, task_idx])
    loss.backward()
    optimizer.step()

    task_losses = [0.0] * N_TASKS
    task_losses[task_idx] = loss.item()
    return {"task_losses": task_losses}


# ---------------------------------------------------------------------------
# LS step
# ---------------------------------------------------------------------------

def _step_ls(
    model:     MPNNMultiTask,
    optimizer: torch.optim.Optimizer,
    batch:     dict,
    means:     torch.Tensor,
    stds:      torch.Tensor,
    device:    torch.device,
) -> dict:
    batch = batch_to_device(batch, device)
    targets_norm = (batch["targets"] - means) / stds

    optimizer.zero_grad()
    h     = model.shared_forward(batch)
    losses = [
        F.l1_loss(model.task_forward(h, i), targets_norm[:, i])
        for i in range(N_TASKS)
    ]
    torch.stack(losses).mean().backward()
    optimizer.step()

    return {"task_losses": [l.item() for l in losses]}


# ---------------------------------------------------------------------------
# PCGrad step
# ---------------------------------------------------------------------------

def _step_pcgrad(
    model:     MPNNMultiTask,
    optimizer: torch.optim.Optimizer,
    batch:     dict,
    means:     torch.Tensor,
    stds:      torch.Tensor,
    device:    torch.device,
) -> dict:
    flat_grads, task_loss_vals = _per_task_grads(
        model, batch, means, stds, device
    )
    g_combined = combine_gradients("pcgrad", flat_grads)
    _fill_head_grads_and_override_encoder(
        model, batch, means, stds, device, g_combined, optimizer
    )
    return {"task_losses": task_loss_vals}


# ---------------------------------------------------------------------------
# AIM step
# ---------------------------------------------------------------------------

def _step_aim(
    model:            MPNNMultiTask,
    policy:           torch.nn.Module,
    policy_loss_fn:   AIMPolicyLoss,
    optimizer_model:  torch.optim.Optimizer,
    optimizer_policy: torch.optim.Optimizer,
    primary_batch:    dict,
    guide_batch:      dict,
    means:            torch.Tensor,
    stds:             torch.Tensor,
    device:           torch.device,
) -> dict:
    # ── 1-2: Per-task encoder gradients on primary batch ─────────────────
    flat_grads, task_loss_vals = _per_task_grads(
        model, primary_batch, means, stds, device
    )

    # ── 3: AIM intervention ───────────────────────────────────────────────
    g_intervened, _ = policy(flat_grads)

    # ── 4-5: Model update ─────────────────────────────────────────────────
    _fill_head_grads_and_override_encoder(
        model, primary_batch, means, stds, device, g_intervened, optimizer_model
    )

    # ── 6: Guidance gradients ─────────────────────────────────────────────
    guide_flat_grads, guide_loss_vals = _per_task_grads(
        model, guide_batch, means, stds, device
    )

    # ── 7: AIM on guide grads (differentiable w.r.t. τ) ──────────────────
    g_int_guide, _ = policy(guide_flat_grads)

    # ── 8-9: Policy update ────────────────────────────────────────────────
    guide_loss_tensors = [
        torch.tensor(v, dtype=torch.float32, device=device) for v in guide_loss_vals
    ]
    optimizer_policy.zero_grad()
    L_policy, policy_info = policy_loss_fn(
        task_losses=guide_loss_tensors,
        gradients=guide_flat_grads,
        g_intervened=g_int_guide,
        guide_losses=guide_loss_tensors,
    )
    L_policy.backward()
    optimizer_policy.step()

    return {"task_losses": task_loss_vals, **policy_info}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model:  MPNNMultiTask,
    loader,
    means:  torch.Tensor,
    stds:   torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    """Compute MAE in physical (un-normalised) units for each task."""
    model.eval()
    accum_abs_err = [0.0] * N_TASKS
    n_total = 0

    for batch in loader:
        batch   = batch_to_device(batch, device)
        targets = batch["targets"]
        B       = targets.shape[0]

        h = model.shared_forward(batch)
        preds_norm = torch.stack(
            [model.task_forward(h, i) for i in range(N_TASKS)], dim=1
        )
        preds = preds_norm * stds + means

        for i in range(N_TASKS):
            accum_abs_err[i] += F.l1_loss(
                preds[:, i], targets[:, i], reduction="sum"
            ).item()
        n_total += B

    model.train()
    return {TASK_NAMES[i]: accum_abs_err[i] / n_total for i in range(N_TASKS)}


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    method:           str   = "aim_scalar",
    n_train:          int   = 5000,
    n_epochs:         int   = 5,
    seed:             int   = 42,
    head_hidden:      int   = 64,
    trainable_layers: int   = -1,     # -1 = all layers (no pretrained weights)
    lr_model:         float = 1e-4,
    lr_policy:        float = 1e-3,
    batch_size:       int   = 32,
    lambda_g:         float = 1.0,
    lambda_m:         float = 0.01,
    lambda_p:         float = 0.08,
    k:                float = 10.0,
    stl_task_idx:     int   = 0,
    data_root:        str   = "../../data/qm9",
    save_dir:         str   = "../results_Adam",
    log_every:        int   = 1,
    device_str:       Optional[str] = None,
) -> List[dict]:

    # ── Device ────────────────────────────────────────────────────────────
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*65}")
    print(f"  AIM + MPNN (Gilmer 2017) | method={method}  n_train={n_train:,}  "
          f"seed={seed}  device={device}")
    print(f"{'='*65}")

    # ── Data ──────────────────────────────────────────────────────────────
    loaders = get_loaders(
        root=data_root, n_train=n_train,
        batch_size=batch_size, seed=seed,
    )
    means = loaders["means"].to(device)
    stds  = loaders["stds"].to(device)
    print("Split sizes:", loaders["split_sizes"])

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(
        device=device,
        head_hidden=head_hidden,
        trainable_layers=trainable_layers,
    )
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_model  = torch.optim.Adam(trainable_params, lr=lr_model)
    scheduler_model  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_model,
        mode="min",
        factor=0.5,
        patience=50,
    )

    # ── Policy (AIM only) ─────────────────────────────────────────────────
    policy           = None
    optimizer_policy = None
    scheduler_policy = None
    policy_loss_fn   = None

    if method == "aim_scalar":
        policy           = AIMScalarPolicy(n_tasks=N_TASKS, k=k).to(device)
        optimizer_policy = torch.optim.Adam(policy.parameters(), lr=lr_policy)
        scheduler_policy = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer_policy,
            T_0=20,
            T_mult=1,
            eta_min=1e-6,
        )
        policy_loss_fn   = AIMPolicyLoss(lambda_g, lambda_m, lambda_p)

    elif method == "aim_matrix":
        policy           = AIMMatrixPolicy(n_tasks=N_TASKS, k=k).to(device)
        optimizer_policy = torch.optim.Adam(policy.parameters(), lr=lr_policy)
        scheduler_policy = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer_policy,
            T_0=20,
            T_mult=1,
            eta_min=1e-6,
        )
        policy_loss_fn   = AIMPolicyLoss(lambda_g, lambda_m, lambda_p)

    elif method not in BASELINE_METHODS:
        raise ValueError(f"Unknown method: {method!r}. "
                         f"Choose from: ls, pcgrad, aim_scalar, aim_matrix, stl")

    # ── Save path ─────────────────────────────────────────────────────────
    if method == "stl":
        run_name = f"stl_task{stl_task_idx}_{TASK_NAMES[stl_task_idx]}_n{n_train}_seed{seed}"
    else:
        run_name = f"{method}_n{n_train}_seed{seed}"
    save_path = Path(save_dir) / run_name
    save_path.mkdir(parents=True, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────
    history   = []
    best_val  = float("inf")
    guide_cyc = itertools.cycle(loaders["guide_loader"])

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_steps: List[dict] = []

        for primary_batch in tqdm(
            loaders["primary_loader"],
            desc=f"Ep {epoch:3d}/{n_epochs} [{method}]",
            leave=False,
        ):
            if method == "stl":
                step_info = _step_stl(
                    model, optimizer_model,
                    primary_batch, means, stds, device,
                    stl_task_idx,
                )
            elif method == "ls":
                step_info = _step_ls(
                    model, optimizer_model,
                    primary_batch, means, stds, device,
                )
            elif method == "pcgrad":
                step_info = _step_pcgrad(
                    model, optimizer_model,
                    primary_batch, means, stds, device,
                )
            else:
                guide_batch = next(guide_cyc)
                step_info = _step_aim(
                    model, policy, policy_loss_fn,
                    optimizer_model, optimizer_policy,
                    primary_batch, guide_batch,
                    means, stds, device,
                )
            epoch_steps.append(step_info)

        # ── Validation ────────────────────────────────────────────────────
        val_results  = evaluate(model, loaders["val_loader"],  means, stds, device)
        test_results = evaluate(model, loaders["test_loader"], means, stds, device)
        val_mae_mean = float(np.mean(list(val_results.values())))
        scheduler_model.step(val_mae_mean)
        if scheduler_policy is not None:
            scheduler_policy.step(epoch)

        avg_task_losses = np.mean(
            [s["task_losses"] for s in epoch_steps], axis=0
        ).tolist()

        log: dict = {
            "epoch":           epoch,
            "val_mae_mean":    val_mae_mean,
            "val_per_task":    val_results,
            "test_per_task":   test_results,
            "train_task_loss": avg_task_losses,
        }

        if policy is not None:
            tau_np = policy.tau.detach().cpu().numpy()
            log["tau"] = tau_np.tolist()
            if "L_policy" in epoch_steps[0]:
                for key in ("L_policy", "L_magnitude", "L_progress", "L_guide"):
                    vals = [s.get(key, 0.0) for s in epoch_steps]
                    log[key] = float(np.mean(vals))

        history.append(log)

        # ── Checkpoint ────────────────────────────────────────────────────
        if val_mae_mean < best_val:
            best_val = val_mae_mean
            ckpt = {
                "model":        model.state_dict(),
                "epoch":        epoch,
                "val_results":  val_results,
                "test_results": test_results,
            }
            if policy is not None:
                ckpt["policy"] = policy.state_dict()
            torch.save(ckpt, save_path / "best_model.pt")

        # ── Logging ───────────────────────────────────────────────────────
        if epoch % log_every == 0 or epoch == 1:
            task_str = "  ".join(
                f"{k}={v:.4f}" for k, v in val_results.items()
            )
            policy_str = ""
            if policy is not None and "L_policy" in log:
                policy_str = (
                    f"  |  L_pol={log['L_policy']:.4f} "
                    f"L_mag={log['L_magnitude']:.4f} "
                    f"L_prg={log['L_progress']:.4f}"
                )
            print(f"Ep {epoch:3d} | val_MAE={val_mae_mean:.4f} | "
                  f"{task_str}{policy_str}")

    with open(save_path / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val MAE : {best_val:.4f}")
    print(f"Results saved: {save_path}")
    return history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":


    p = argparse.ArgumentParser(description="Train AIM or baseline with MPNN (Gilmer 2017) on QM9")

    p.add_argument("--method",           default="aim_matrix",
                   choices=["ls", "pcgrad", "aim_scalar", "aim_matrix", "stl"])
    p.add_argument("--stl_task_idx",     type=int,   default=0,
                   help="Task index for STL (0=mu, 1=U0, 2=U)")
    p.add_argument("--n_train",          type=int,   default=5000)
    p.add_argument("--n_epochs",         type=int,   default=250)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--head_hidden",      type=int,   default=64)
    p.add_argument("--trainable_layers", type=int,   default= 2,
                   help="-1 = all layers trainable (default for GNN from scratch)")
    p.add_argument("--lr_model",         type=float, default=5e-4)
    p.add_argument("--lr_policy",        type=float, default=4e-4)
    p.add_argument("--batch_size",       type=int,   default=32)
    p.add_argument("--lambda_g",         type=float, default=1.0)
    p.add_argument("--lambda_m",         type=float, default=0.01)
    p.add_argument("--lambda_p",         type=float, default=0.08)
    p.add_argument("--k",                type=float, default=10.0)
    p.add_argument("--data_root",        default="../../data/qm9")
    p.add_argument("--save_dir",         default="../results_lr5e4")
    p.add_argument("--log_every",        type=int,   default=1)
    p.add_argument("--device",           default=None, dest="device_str")
    args = p.parse_args()

    train(**vars(args))
