"""
Evaluation metrics from the AIM paper.

Mean Rank (MR):
    For each task, rank all methods by MAE (rank 1 = best).
    MR for a method = average rank across all tasks.
    Lower MR is better.

Delta_m% (Δm%):
    Mean percentage improvement of method over the STL baseline.
    Δm% = (1/T) * sum_i  [ sign_i * (stl_i - method_i) / stl_i * 100 ]
    where sign_i = +1 when lower is better (MAE regression), -1 otherwise.
    Higher Δm% is better (more positive = bigger improvement).
"""

import numpy as np
from typing import Dict, List, Optional


TaskMAE = Dict[str, float]          # {task_name: mae_value}
Results = Dict[str, TaskMAE]        # {method_name: TaskMAE}


def _average_ranks(values: List[float]) -> List[float]:
    """
    Rank values ascending (rank 1 = smallest/best), giving tied values the
    average of the rank positions they jointly occupy — e.g. two values tied
    for 2nd/3rd both get rank 2.5. Matches the paper's stated tie-breaking
    convention (equivalent to scipy.stats.rankdata(..., method="average")).
    """
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0   # 1-indexed average of positions i+1..j+1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def mean_rank(results: Results) -> Dict[str, float]:
    """
    Compute Mean Rank for every method.

    Args:
        results : {method: {task: mae}}

    Returns:
        {method: mean_rank}  — lower is better
    """
    methods = list(results.keys())
    tasks   = list(next(iter(results.values())).keys())

    rank_lists: Dict[str, List[float]] = {m: [] for m in methods}

    for task in tasks:
        values = [results[m].get(task, float("inf")) for m in methods]
        ranks  = _average_ranks(values)
        for method, rank in zip(methods, ranks):
            rank_lists[method].append(rank)

    return {m: float(np.mean(rank_lists[m])) for m in methods}


def delta_m_percent(
    results:        Results,
    stl_baseline:   TaskMAE,
    higher_is_better: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute Δm% for every method relative to single-task learning (STL).

    Δm%_method = (1/T) * sum_i  (stl_i - method_i) / |stl_i| * 100

    Interpretation: positive Δm% means the method is BETTER than STL.
    This implementation uses the more intuitive convention where higher Δm% = better.

    Args:
        results          : {method: {task: mae}}
        stl_baseline     : {task: mae}  from single-task training
        higher_is_better : list of task names where higher metric is better
                           (none for MAE-only benchmarks)

    Returns:
        {method: delta_m_percent}
    """
    if higher_is_better is None:
        higher_is_better = []

    methods = list(results.keys())
    tasks   = list(stl_baseline.keys())

    delta = {}
    for method in methods:
        pct_improvements = []
        for task in tasks:
            stl_val    = stl_baseline[task]
            method_val = results[method].get(task, float("nan"))
            if abs(stl_val) < 1e-12:
                continue
            if task in higher_is_better:
                pct = (method_val - stl_val) / abs(stl_val) * 100.0
            else:
                # For MAE: improvement = stl > method
                pct = (stl_val - method_val) / abs(stl_val) * 100.0
            pct_improvements.append(pct)

        delta[method] = float(np.mean(pct_improvements)) if pct_improvements else float("nan")

    return delta


def print_results_table(
    results:      Results,
    stl_baseline: Optional[TaskMAE] = None,
    task_names:   Optional[List[str]] = None,
):
    """Pretty-print the full results table."""
    methods = list(results.keys())
    if task_names is None:
        task_names = list(next(iter(results.values())).keys())

    mr = mean_rank(results)
    dm = delta_m_percent(results, stl_baseline) if stl_baseline else {}
    has_delta = stl_baseline is not None

    col_w = 9
    header_parts = [f"{'Method':<22}", f"{'MR':>5}", f"{'Δm%':>7}"]
    for t in task_names:
        header_parts.append(f"{t:>{col_w}}")
    print("  ".join(header_parts))
    print("-" * (22 + 5 + 7 + (col_w + 2) * len(task_names) + 10))

    for method in sorted(methods, key=lambda m: mr.get(m, 999)):
        delta_text = f"{dm.get(method, 0):>7.2f}%" if has_delta else f"{'N/A':>7}"
        row = [f"{method:<22}", f"{mr.get(method, 0):>5.1f}", delta_text]
        for t in task_names:
            val = results[method].get(t, float("nan"))
            row.append(f"{val:>{col_w}.4f}")
        print("  ".join(row))
