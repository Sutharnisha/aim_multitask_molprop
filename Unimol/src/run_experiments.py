"""
Full experiment runner — AIM + Uni-Mol on QM9.

Runs all 36 training jobs:
  Methods : ls, pcgrad, aim_scalar, aim_matrix
  Subsets : 10k, 50k, 100k
  Seeds   : 42, 123, 7

Launch:
    python run_experiments.py                          # all jobs sequentially
    python run_experiments.py --n_train 10000          # only 10k jobs
    python run_experiments.py --methods aim_scalar ls  # specific methods
    python run_experiments.py --dry_run                # preview schedule only

GPU:
    Uni-Mol requires ~12 GB VRAM.
    Use a cloud GPU: Colab A100, Kaggle T4, or Lambda Labs.
    Pass --device cuda (default) or --device cpu for CPU testing.
"""

import subprocess
import sys
import argparse
from pathlib import Path
from itertools import product


DEFAULT_METHODS  = ["ls", "pcgrad", "aim_scalar", "aim_matrix"]
DEFAULT_N_TRAINS = [10_000, 50_000, 100_000]
DEFAULT_SEEDS    = [42, 123, 7]

# AIM hyperparameters from paper Table 3
AIM_HP = dict(lambda_g=1.0, lambda_m=0.01, lambda_p=0.08, k=10.0)


def build_cmd(
    method:      str,
    n_train:     int,
    seed:        int,
    n_epochs:    int,
    batch_size:  int,
    head_hidden: int,
    lr_model:    float,
    lr_policy:   float,
    data_root:   str,
    save_dir:    str,
    device:      str,
) -> list:
    cmd = [
        sys.executable, "train.py",
        "--method",      method,
        "--n_train",     str(n_train),
        "--n_epochs",    str(n_epochs),
        "--seed",        str(seed),
        "--head_hidden", str(head_hidden),
        "--batch_size",  str(batch_size),
        "--lr_model",    str(lr_model),
        "--lr_policy",   str(lr_policy),
        "--data_root",   data_root,
        "--save_dir",    save_dir,
    ]
    if device:
        cmd += ["--device", device]
    if method in ("aim_scalar", "aim_matrix"):
        cmd += [
            "--lambda_g", str(AIM_HP["lambda_g"]),
            "--lambda_m", str(AIM_HP["lambda_m"]),
            "--lambda_p", str(AIM_HP["lambda_p"]),
            "--k",        str(AIM_HP["k"]),
        ]
    return cmd


def run_all(
    methods:      list  = None,
    n_trains:     list  = None,
    seeds:        list  = None,
    n_epochs:     int   = 300,
    batch_size:   int   = 32,
    head_hidden:  int   = 64,
    lr_model:     float = 1e-4,
    lr_policy:    float = 1e-3,
    data_root:    str   = "./data/qm9",
    save_dir:     str   = "./results",
    device:       str   = "cuda",
    dry_run:      bool  = False,
    skip_existing:bool  = True,
):
    methods  = methods  or DEFAULT_METHODS
    n_trains = n_trains or DEFAULT_N_TRAINS
    seeds    = seeds    or DEFAULT_SEEDS

    jobs = list(product(methods, n_trains, seeds))
    print(f"\nScheduled {len(jobs)} jobs (Uni-Mol backbone):\n")

    src_dir = Path(__file__).parent

    for i, (method, n_train, seed) in enumerate(jobs, 1):
        run_name  = f"{method}_n{n_train}_seed{seed}"
        done_flag = Path(save_dir) / run_name / "history.json"

        if skip_existing and done_flag.exists():
            print(f"  [{i:2d}/{len(jobs)}] SKIP (exists) : {run_name}")
            continue

        cmd = build_cmd(
            method=method, n_train=n_train, seed=seed,
            n_epochs=n_epochs, batch_size=batch_size,
            head_hidden=head_hidden,
            lr_model=lr_model, lr_policy=lr_policy,
            data_root=data_root, save_dir=save_dir, device=device,
        )

        print(f"  [{i:2d}/{len(jobs)}] {'DRY' if dry_run else 'RUN'} : {run_name}")
        print(f"         {' '.join(cmd[2:])}")

        if not dry_run:
            result = subprocess.run(cmd, cwd=str(src_dir))
            if result.returncode != 0:
                print(f"  !! FAILED: {run_name} (exit code {result.returncode})")

    if not dry_run:
        print("\nAll jobs done. Running analysis...")
        subprocess.run(
            [sys.executable, "analysis.py", "--results_dir", save_dir],
            cwd=str(src_dir),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch all AIM + Uni-Mol QM9 experiments"
    )
    parser.add_argument("--methods",       nargs="+", default=None,
                        choices=DEFAULT_METHODS)
    parser.add_argument("--n_train",       nargs="+", type=int,   default=None)
    parser.add_argument("--seeds",         nargs="+", type=int,   default=None)
    parser.add_argument("--n_epochs",      type=int,   default=300)
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--head_hidden",   type=int,   default=64)
    parser.add_argument("--lr_model",      type=float, default=1e-4)
    parser.add_argument("--lr_policy",     type=float, default=1e-3)
    parser.add_argument("--data_root",     default="./data/qm9")
    parser.add_argument("--save_dir",      default="./results")
    parser.add_argument("--device",        default="cuda")
    parser.add_argument("--dry_run",       action="store_true")
    parser.add_argument("--no_skip",       action="store_false", dest="skip_existing")
    args = parser.parse_args()

    run_all(
        methods=args.methods,
        n_trains=args.n_train,
        seeds=args.seeds,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        head_hidden=args.head_hidden,
        lr_model=args.lr_model,
        lr_policy=args.lr_policy,
        data_root=args.data_root,
        save_dir=args.save_dir,
        device=args.device,
        dry_run=args.dry_run,
        skip_existing=args.skip_existing,
    )
