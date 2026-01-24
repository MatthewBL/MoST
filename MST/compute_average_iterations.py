from pathlib import Path
from collections import defaultdict
import argparse
import csv
import sys


def compute_average_iterations(data_root: Path, include_missing_as_zero: bool = False):
    """
    Compute the average number of iteration directories per sub-experiment across all experiments.

    Definitions:
    - "Experiment": a top-level folder under data_root (e.g., "deepseek 7B - a40").
    - "Sub-experiment": a folder under an experiment (e.g., "16_16").
    - "Iteration": any directory under a sub-experiment (e.g., "2026-01-15_15-53-19").

    Behavior:
    - For each sub-experiment name, we gather iteration counts from each experiment that contains it.
    - The average is computed across experiments that have the sub-experiment. If
      include_missing_as_zero=True, experiments that lack the sub-experiment contribute 0.

    Returns:
    - averages: dict[str, float] mapping sub-experiment name to average iteration count.
    - meta: dict[str, dict] with summary stats per sub-experiment (total_iterations, experiments_considered).
    """

    if not data_root.exists() or not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist or is not a directory: {data_root}")

    experiments = [p for p in data_root.iterdir() if p.is_dir()]

    # Map experiment -> {subexp_name: iteration_count}
    exp_subexp_counts: dict[str, dict[str, int]] = {}
    all_subexp_names: set[str] = set()

    for exp in experiments:
        subexp_dirs = [p for p in exp.iterdir() if p.is_dir()]
        sub_counts: dict[str, int] = {}
        for sub in subexp_dirs:
            # Count iteration directories under the sub-experiment
            try:
                iter_count = sum(1 for q in sub.iterdir() if q.is_dir())
            except PermissionError:
                # Skip unreadable directories but keep going
                iter_count = 0
            sub_counts[sub.name] = iter_count
            all_subexp_names.add(sub.name)
        exp_subexp_counts[exp.name] = sub_counts

    # Aggregate counts per sub-experiment across experiments
    subexp_counts_by_name: defaultdict[str, list[int]] = defaultdict(list)
    for name in sorted(all_subexp_names):
        for exp in experiments:
            counts_for_exp = exp_subexp_counts.get(exp.name, {})
            if name in counts_for_exp:
                subexp_counts_by_name[name].append(counts_for_exp[name])
            elif include_missing_as_zero:
                subexp_counts_by_name[name].append(0)
            else:
                # Exclude experiments where the sub-experiment doesn't exist
                pass

    # Compute averages and summary metadata
    averages: dict[str, float] = {}
    meta: dict[str, dict] = {}
    for name, counts in subexp_counts_by_name.items():
        if counts:
            total = sum(counts)
            considered = len(counts)
            averages[name] = total / considered if considered > 0 else 0.0
            meta[name] = {
                "total_iterations": total,
                "experiments_considered": considered,
                "include_missing_as_zero": include_missing_as_zero,
            }

    return averages, meta


def compute_global_average_iterations(data_root: Path, include_missing_as_zero: bool = False) -> float:
    """
    Compute a single global average of iteration counts across ALL (experiment, sub-experiment) pairs.

    - "Experiment": a top-level folder under data_root (e.g., "deepseek 7B - a40").
    - "Sub-experiment": a folder under an experiment (e.g., "16_16").
    - "Iteration": any directory under a sub-experiment (e.g., "2026-01-15_15-53-19").

    Definitions of averaging:
    - If include_missing_as_zero=False (default): average across all PRESENT (experiment, sub-experiment) pairs.
      That is, we only consider pairs where the sub-experiment exists for a given experiment.
    - If include_missing_as_zero=True: average across the UNION of sub-experiment names across experiments,
      counting 0 for experiments that do not have a given sub-experiment.

    Returns:
    - A single float: the global average number of iterations per (experiment, sub-experiment) pair.
    """

    if not data_root.exists() or not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist or is not a directory: {data_root}")

    experiments = [p for p in data_root.iterdir() if p.is_dir()]
    if not experiments:
        return 0.0

    # Collect counts per experiment and sub-experiment
    exp_sub_counts: dict[str, dict[str, int]] = {}
    all_subexp_names: set[str] = set()

    for exp in experiments:
        subexp_dirs = [p for p in exp.iterdir() if p.is_dir()]
        sub_counts: dict[str, int] = {}
        for sub in subexp_dirs:
            try:
                iter_count = sum(1 for q in sub.iterdir() if q.is_dir())
            except PermissionError:
                iter_count = 0
            sub_counts[sub.name] = iter_count
            all_subexp_names.add(sub.name)
        exp_sub_counts[exp.name] = sub_counts

    total_iterations = 0
    total_pairs = 0

    if include_missing_as_zero:
        # Consider the full grid: experiments x union(sub-experiment names)
        for exp in experiments:
            sub_counts = exp_sub_counts.get(exp.name, {})
            for name in all_subexp_names:
                total_iterations += sub_counts.get(name, 0)
                total_pairs += 1
    else:
        # Consider only PRESENT pairs
        for exp in experiments:
            sub_counts = exp_sub_counts.get(exp.name, {})
            for _, count in sub_counts.items():
                total_iterations += count
                total_pairs += 1

    if total_pairs == 0:
        return 0.0
    return total_iterations / total_pairs


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute the average number of iterations per sub-experiment across all experiments "
            "under a given data root."
        )
    )
    default_data_root = Path(__file__).resolve().parent / "data"
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(default_data_root),
        help=f"Path to the data root (default: {default_data_root})",
    )
    parser.add_argument(
        "--include-missing-as-zero",
        action="store_true",
        help=(
            "If set, experiments without a given sub-experiment will contribute 0 to that sub-experiment's average."
        ),
    )
    parser.add_argument(
        "--global-average",
        action="store_true",
        help=(
            "If set, outputs a single global average across all (experiment, sub-experiment) pairs."
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional CSV output path to write the results.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists() or not data_root.is_dir():
        print(f"ERROR: Data root not found or not a directory: {data_root}", file=sys.stderr)
        sys.exit(1)

    if args.global_average:
        value = compute_global_average_iterations(data_root, include_missing_as_zero=args.include_missing_as_zero)
        print("metric,value")
        print(f"global_average_iterations,{value:.6f}")
        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["metric", "value"])
                writer.writerow(["global_average_iterations", f"{value:.6f}"])
            print(f"Wrote CSV: {out_path}")
    else:
        averages, meta = compute_average_iterations(data_root, include_missing_as_zero=args.include_missing_as_zero)

        # Print to console
        print("sub_experiment,average_iterations,experiments_considered,total_iterations")
        for name in sorted(averages.keys()):
            m = meta[name]
            print(f"{name},{averages[name]:.3f},{m['experiments_considered']},{m['total_iterations']}")

        # Optionally write CSV
        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "sub_experiment",
                    "average_iterations",
                    "experiments_considered",
                    "total_iterations",
                ])
                for name in sorted(averages.keys()):
                    m = meta[name]
                    writer.writerow([name, f"{averages[name]:.6f}", m["experiments_considered"], m["total_iterations"]])
            print(f"Wrote CSV: {out_path}")


if __name__ == "__main__":
    main()
