import os
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Optional
import argparse

import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Configuration
# -----------------------------
RUN_DIR_PATTERN = re.compile(r"^requests_deepseek_\d+$")
TOKEN_DIR_PATTERN = re.compile(r"^(\d+)_(\d+)$")
TIMESTAMP_DIR_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})$")
RESULTS_FILE = "results.csv"
FIRST_HALF_FILE = "first_half.csv"
SECOND_HALF_FILE = "second_half.csv"

# -----------------------------
# Helpers
# -----------------------------

def parse_token_dir(name: str) -> Optional[Tuple[int, int]]:
    m = TOKEN_DIR_PATTERN.match(name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def parse_timestamp_dir(name: str) -> Optional[datetime]:
    m = TIMESTAMP_DIR_PATTERN.match(name)
    if not m:
        return None
    try:
        return datetime.strptime(f"{m.group(1)}_{m.group(2)}", "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        return None


def read_result_csv(csv_path: Path) -> Optional[Tuple[bool, float]]:
    """
    Read a results.csv and return (evaluation_bool, req_min_float) for that attempt.
    Returns None if file unreadable or missing required columns.
    """
    if not csv_path.exists():
        return None
    try:
        with csv_path.open(newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # Expect a single row
            for row in reader:
                eval_val = row.get("EVALUATION")
                if eval_val is None:
                    return None
                evaluation = str(eval_val).strip().lower() == "true"
                req_min_val = row.get("REQ_MIN")
                if req_min_val is None:
                    return None
                req_min = float(req_min_val)
                return evaluation, req_min
        return None
    except Exception:
        return None


def collect_mst_for_run(run_root: Path) -> Dict[Tuple[int, int], float]:
    """
    For a given run root (e.g., requests_deepseek_1), compute MST for each token pair:
    - For each token directory (e.g., 128_256), find the latest timestamp subdir
      with EVALUATION == True in results.csv; use its REQ_MIN as MST.
    Returns a mapping {(input_tokens, output_tokens): mst_value}.
    """
    mst: Dict[Tuple[int, int], float] = {}

    for token_dir in sorted(p for p in run_root.iterdir() if p.is_dir()):
        parsed = parse_token_dir(token_dir.name)
        if not parsed:
            continue
        in_tok, out_tok = parsed

        # Gather timestamped subdirectories with valid timestamps
        timestamp_dirs: List[Tuple[datetime, Path]] = []
        for ts_dir in token_dir.iterdir():
            if not ts_dir.is_dir():
                continue
            ts = parse_timestamp_dir(ts_dir.name)
            if ts is not None:
                timestamp_dirs.append((ts, ts_dir))

        if not timestamp_dirs:
            continue

        # Sort by timestamp ascending, then pick the last True
        timestamp_dirs.sort(key=lambda x: x[0])
        last_true_reqmin: Optional[float] = None
        for _, ts_path in timestamp_dirs:
            result = read_result_csv(ts_path / RESULTS_FILE)
            if not result:
                continue
            evaluation, req_min = result
            if evaluation:
                last_true_reqmin = req_min

        if last_true_reqmin is not None:
            mst[(in_tok, out_tok)] = last_true_reqmin
        # If no True found, leave as missing (not added)

    return mst


def collect_mst_iteration_dirs_for_run(run_root: Path) -> Dict[Tuple[int, int], Path]:
    """
    For a given run root, identify the iteration directory (timestamp folder)
    corresponding to the MST (latest EVALUATION==True) for each token pair.
    Returns mapping {(input_tokens, output_tokens): ts_dir_path}.
    """
    chosen: Dict[Tuple[int, int], Path] = {}

    for token_dir in sorted(p for p in run_root.iterdir() if p.is_dir()):
        parsed = parse_token_dir(token_dir.name)
        if not parsed:
            continue
        in_tok, out_tok = parsed

        timestamp_dirs: List[Tuple[datetime, Path]] = []
        for ts_dir in token_dir.iterdir():
            if not ts_dir.is_dir():
                continue
            ts = parse_timestamp_dir(ts_dir.name)
            if ts is not None:
                timestamp_dirs.append((ts, ts_dir))
        if not timestamp_dirs:
            continue
        timestamp_dirs.sort(key=lambda x: x[0])

        last_true_path: Optional[Path] = None
        for _, ts_path in timestamp_dirs:
            result = read_result_csv(ts_path / RESULTS_FILE)
            if not result:
                continue
            evaluation, _ = result
            if evaluation:
                last_true_path = ts_path

        if last_true_path is not None:
            chosen[(in_tok, out_tok)] = last_true_path

    return chosen


def build_matrix(mst_map: Dict[Tuple[int, int], float]) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Build a matrix (rows=output tokens, cols=input tokens) filled with MST values.
    Returns (matrix, x_inputs_sorted, y_outputs_sorted)
    """
    if not mst_map:
        return np.zeros((0, 0)), [], []

    inputs = sorted({k[0] for k in mst_map.keys()})
    outputs = sorted({k[1] for k in mst_map.keys()})
    mat = np.full((len(outputs), len(inputs)), np.nan, dtype=float)

    out_index = {o: i for i, o in enumerate(outputs)}
    in_index = {i: j for j, i in enumerate(inputs)}

    for (i_tok, o_tok), val in mst_map.items():
        mat[out_index[o_tok], in_index[i_tok]] = val

    return mat, inputs, outputs


def plot_heatmap(matrix: np.ndarray, inputs: List[int], outputs: List[int], title: str, out_path: Path, cbar_label: str = "MST (req/min)") -> None:
    if matrix.size == 0:
        print(f"Skipping plot for {title}: no data")
        return

    # Reverse rows so that smaller output tokens are at the bottom
    matrix_plot = matrix[::-1, :]
    outputs_plot = outputs[::-1]

    plt.figure(figsize=(max(6, len(inputs) * 0.8), max(5, len(outputs) * 0.6)))
    sns.heatmap(
        matrix_plot,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        xticklabels=inputs,
        yticklabels=outputs_plot,
        linewidths=0.5,
        linecolor='white',
        cbar_kws={"label": cbar_label}
    )
    plt.xlabel("Input tokens")
    plt.ylabel("Output tokens")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def compute_outlier_flags(
    per_run_mst: Dict[str, Dict[Tuple[int, int], float]],
    method: str = "hybrid",
    z_thresh: float = 3.5,
    iqr_k: float = 1.5,
    ratio_upper: float = 2.5,
    ratio_lower: float = 0.4,
    log_space: bool = False,
    min_runs: int = 3,
    report_path: Optional[Path] = None,
) -> set:
    """
    Detect outliers across runs for each (input, output) token pair using configurable robust statistics.
    - method: 'mad', 'iqr', 'ratio', or 'hybrid' (any flagged in any rule)
    - log_space: apply detection on log10 scale (recommended for multiplicative outliers)
    - min_runs: require at least N values to evaluate outliers for a pair
    Returns a set of (run_name, pair) flagged as outliers. Optionally writes a CSV report.
    """
    per_pair: Dict[Tuple[int, int], List[Tuple[str, float]]] = {}
    for run_name, mapping in per_run_mst.items():
        for pair, val in mapping.items():
            per_pair.setdefault(pair, []).append((run_name, float(val)))

    flagged: set = set()
    rows_for_report: List[List[object]] = []

    for pair, items in per_pair.items():
        if len(items) < min_runs:
            continue
        run_names = [r for r, _ in items]
        vals = np.array([v for _, v in items], dtype=float)

        # Working values for robust stats
        work_vals = np.log10(np.clip(vals, a_min=1e-12, a_max=None)) if log_space else vals
        med = float(np.median(work_vals))
        mad = float(np.median(np.abs(work_vals - med)))
        q1, q3 = np.percentile(work_vals, [25, 75])
        iqr = float(q3 - q1)
        lower_iqr = q1 - iqr_k * iqr
        upper_iqr = q3 + iqr_k * iqr

        base_med_raw = float(np.median(vals)) if len(vals) else 0.0
        ratio = vals / base_med_raw if base_med_raw != 0 else np.full_like(vals, np.inf)

        mask_mad = np.zeros_like(vals, dtype=bool)
        mask_iqr = np.zeros_like(vals, dtype=bool)
        mask_ratio = np.zeros_like(vals, dtype=bool)

        if method in ("mad", "hybrid") and mad > 0:
            mod_z = 0.6745 * (work_vals - med) / mad
            mask_mad = np.abs(mod_z) > z_thresh
        if method in ("iqr", "hybrid") and iqr > 0:
            mask_iqr = (work_vals < lower_iqr) | (work_vals > upper_iqr)
        if method in ("ratio", "hybrid"):
            mask_ratio = (ratio > ratio_upper) | (ratio < ratio_lower)

        if method == "mad":
            mask = mask_mad
        elif method == "iqr":
            mask = mask_iqr
        elif method == "ratio":
            mask = mask_ratio
        else:
            mask = mask_mad | mask_iqr | mask_ratio

        for rn, v, wv, r, is_out in zip(run_names, vals, work_vals, ratio, mask):
            if bool(is_out):
                flagged.add((rn, pair))
            if report_path is not None:
                rows_for_report.append([
                    rn, pair[0], pair[1], v, wv,
                    med, mad, z_thresh,
                    q1, q3, iqr, iqr_k,
                    r, ratio_lower, ratio_upper,
                    method, bool(is_out)
                ])

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "run", "input_tokens", "output_tokens", "value_raw", "value_work",
                "median_work", "mad", "z_thresh",
                "q1_work", "q3_work", "iqr", "iqr_k",
                "ratio", "ratio_lower", "ratio_upper",
                "method", "flagged"
            ])
            w.writerows(rows_for_report)

    return flagged


def parse_folder_timestamp(name: str) -> Optional[datetime]:
    return parse_timestamp_dir(name)


def load_half_csv(csv_path: Path) -> List[Tuple[datetime, float]]:
    rows: List[Tuple[datetime, float]] = []
    if not csv_path.exists():
        return rows
    try:
        with csv_path.open(newline='', encoding='utf-8') as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                ts_s = row.get('received_timestamp')
                rt_s = row.get('complete_response_time')
                if ts_s is None or rt_s is None:
                    continue
                try:
                    ts = datetime.strptime(ts_s, "%Y-%m-%d %H:%M:%S")
                    rt = float(rt_s)
                    rows.append((ts, rt))
                except Exception:
                    continue
    except Exception:
        return rows
    return rows


def compute_spike_minute_for_iteration(ts_dir: Path, spike_factor: float, spike_mad_k: Optional[float], spike_min_obs: int, spike_fallback: str) -> Optional[int]:
    """
    Return the minute (0..29) within the iteration when response times started to increase,
    computed relative to iteration start (folder timestamp - 5 minutes). We report the absolute
    minute in the iteration (e.g., 17 for 2 minutes into second half), or None if unavailable.
    """
    folder_dt = parse_folder_timestamp(ts_dir.name)
    if folder_dt is None:
        return None
    iteration_start = folder_dt - timedelta(minutes=5)
    second_half_start = iteration_start + timedelta(minutes=15)

    # Load data
    first = load_half_csv(ts_dir / FIRST_HALF_FILE)
    second = load_half_csv(ts_dir / SECOND_HALF_FILE)
    if not first or not second:
        return None

    first_vals = np.array([rt for _, rt in first], dtype=float)
    if first_vals.size == 0:
        return None

    # Baseline on first half
    base_med = float(np.median(first_vals))
    base_mad = float(np.median(np.abs(first_vals - base_med))) if first_vals.size > 0 else 0.0

    # Bucket second-half into minute bins 0..14
    minute_bins: Dict[int, List[float]] = {i: [] for i in range(15)}
    for ts, rt in second:
        delta = ts - second_half_start
        minute = int(delta.total_seconds() // 60)
        if 0 <= minute < 15:
            minute_bins[minute].append(rt)

    # Compute per-minute medians
    minute_meds: Dict[int, float] = {}
    for m, vals in minute_bins.items():
        if len(vals) >= spike_min_obs:
            minute_meds[m] = float(np.median(np.array(vals, dtype=float)))

    # Detection rule
    def is_spike(val: float) -> bool:
        cond_factor = (val >= base_med * spike_factor)
        if spike_mad_k is not None and base_mad > 0:
            cond_mad = (val >= base_med + spike_mad_k * base_mad)
            return cond_factor or cond_mad
        return cond_factor

    # Find first minute that exceeds thresholds
    for m in range(15):
        if m in minute_meds and is_spike(minute_meds[m]):
            return 15 + m

    if spike_fallback == 'maxdev' and minute_meds:
        # Choose the minute with max ratio to baseline
        best_m = max(minute_meds.keys(), key=lambda k: (minute_meds[k] / base_med) if base_med > 0 else -np.inf)
        return 15 + int(best_m)

    return None


def compute_avg_response_time_for_iteration(ts_dir: Path) -> Optional[float]:
    """
    Compute average complete_response_time across first_half.csv and second_half.csv
    for the given iteration directory.
    """
    first = load_half_csv(ts_dir / FIRST_HALF_FILE)
    second = load_half_csv(ts_dir / SECOND_HALF_FILE)
    vals = []
    vals.extend([rt for _, rt in first])
    vals.extend([rt for _, rt in second])
    if not vals:
        return None
    return float(np.mean(np.array(vals, dtype=float)))


def compute_response_count_for_iteration(ts_dir: Path) -> Optional[int]:
    """
    Count the number of responses in first_half.csv and second_half.csv
    for the given iteration directory. Returns None if neither file has
    valid rows.
    """
    first = load_half_csv(ts_dir / FIRST_HALF_FILE)
    second = load_half_csv(ts_dir / SECOND_HALF_FILE)
    total = len(first) + len(second)
    if total == 0:
        return None
    return int(total)


def compute_response_rate_per_min_for_iteration(ts_dir: Path) -> Optional[float]:
    """
    Compute average responses per minute for the iteration using actual
    timestamp coverage: responses_per_minute = total_responses / active_minutes,
    where active_minutes is the number of unique minute buckets that contain
    at least one response across first and second halves.
    Returns None if there are zero responses or zero active minutes.
    """
    first = load_half_csv(ts_dir / FIRST_HALF_FILE)
    second = load_half_csv(ts_dir / SECOND_HALF_FILE)
    all_rows = []
    all_rows.extend(first)
    all_rows.extend(second)
    if not all_rows:
        return None
    # Unique minute buckets with data
    active_minutes = set()
    for ts, _ in all_rows:
        minute_bucket = ts.replace(second=0, microsecond=0)
        active_minutes.add(minute_bucket)
    if not active_minutes:
        return None
    total = len(all_rows)
    return float(total) / float(len(active_minutes))


def collect_spike_minutes_for_run(run_root: Path, spike_factor: float, spike_mad_k: Optional[float], spike_min_obs: int, spike_fallback: str) -> Dict[Tuple[int, int], int]:
    """
    For each token pair, pick the evaluation==False iteration with lowest REQ_MIN, then compute
    the spike minute within that iteration. Return mapping to minute index (integer).
    """
    out: Dict[Tuple[int, int], int] = {}
    for token_dir in sorted(p for p in run_root.iterdir() if p.is_dir()):
        parsed = parse_token_dir(token_dir.name)
        if not parsed:
            continue
        in_tok, out_tok = parsed

        # Scan timestamped subdirs to find failing iterations
        candidates: List[Tuple[float, Path]] = []
        for ts_dir in token_dir.iterdir():
            if not ts_dir.is_dir():
                continue
            if parse_timestamp_dir(ts_dir.name) is None:
                continue
            rr = read_result_csv(ts_dir / RESULTS_FILE)
            if not rr:
                continue
            evaluation, req_min = rr
            if evaluation is False:
                candidates.append((float(req_min), ts_dir))
        if not candidates:
            continue

        # Choose lowest req/min failing iteration
        candidates.sort(key=lambda x: x[0])
        _, chosen_dir = candidates[0]

        minute = compute_spike_minute_for_iteration(
            chosen_dir,
            spike_factor=spike_factor,
            spike_mad_k=spike_mad_k,
            spike_min_obs=spike_min_obs,
            spike_fallback=spike_fallback,
        )
        if minute is not None:
            out[(in_tok, out_tok)] = int(minute)

    return out

def parse_args():
    p = argparse.ArgumentParser(description="Compute MST heatmaps with optional outlier cleaning.")
    p.add_argument("--method", choices=["mad", "iqr", "ratio", "hybrid"], default="hybrid")
    p.add_argument("--z-thresh", type=float, default=3.5)
    p.add_argument("--iqr-k", type=float, default=1.5)
    p.add_argument("--ratio-upper", type=float, default=2.5)
    p.add_argument("--ratio-lower", type=float, default=0.4)
    p.add_argument("--log-space", action="store_true")
    p.add_argument("--min-runs", type=int, default=3)
    p.add_argument("--winsorize", action="store_true", help="Cap extremes in clean average instead of dropping")
    p.add_argument("--winsor-quantiles", type=str, default="0.05,0.95")
    # Base/root directory of the project (defaults to script folder)
    p.add_argument("--base-dir", type=str, default=None)
    # Explicit data and plots directories; if not provided, default to <base>/data and <base>/plots
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--plots-dir", type=str, default=None)
    # Spike detection controls
    p.add_argument("--spike-factor", type=float, default=1.5, help="Factor vs first-half median to detect spike")
    p.add_argument("--spike-mad-k", type=float, default=None, help="Optional MAD multiplier over first-half median on work scale")
    p.add_argument("--spike-min-obs", type=int, default=1, help="Minimum observations within a minute to consider")
    p.add_argument("--spike-fallback", choices=["none", "maxdev"], default="maxdev")
    # Manual override options
    p.add_argument("--ignore-pairs", type=str, default="", help="Comma-separated list like '2048:2048,128:512' to exclude from clean plots")
    p.add_argument("--ignore-file", type=str, default=None, help="CSV with columns input_tokens,output_tokens to exclude from clean plots")
    p.add_argument("--drop-above", type=float, default=None, help="Exclude any MST strictly above this value in clean plots/averages")
    return p.parse_args()


def main(
    base_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
    plots_dir: Optional[str] = None,
    method: str = "hybrid",
    z_thresh: float = 3.5,
    iqr_k: float = 1.5,
    ratio_upper: float = 2.5,
    ratio_lower: float = 0.4,
    log_space: bool = False,
    min_runs: int = 3,
    winsorize: bool = False,
    winsor_quantiles: Tuple[float, float] = (0.05, 0.95),
    spike_factor: float = 1.5,
    spike_mad_k: Optional[float] = None,
    spike_min_obs: int = 3,
    spike_fallback: str = "maxdev",
):
    # Resolve base, data and plots directories
    root = Path(base_dir) if base_dir else Path(__file__).resolve().parent
    data_base = Path(data_dir) if data_dir else (root / "data")
    plots_base = Path(plots_dir) if plots_dir else (root / "plots")

    if not data_base.exists() or not data_base.is_dir():
        print(f"Data directory not found: {data_base}")
        return
    plots_base.mkdir(parents=True, exist_ok=True)

    # Find run roots matching pattern requests_deepseek_* inside data dir
    run_roots: List[Path] = [
        p for p in data_base.iterdir()
        if p.is_dir() and RUN_DIR_PATTERN.match(p.name)
    ]

    if not run_roots:
        print("No run directories matching 'requests_deepseek_*' found.")
        return

    print(f"Found runs in {data_base}: {[p.name for p in run_roots]}")

    # Collect MSTs per run
    per_run_mst: Dict[str, Dict[Tuple[int, int], float]] = {}
    for run_root in sorted(run_roots, key=lambda p: p.name):
        print(f"Processing {run_root.name} ...")
        mst_map = collect_mst_for_run(run_root)
        per_run_mst[run_root.name] = mst_map

        # Save per-run CSV (to plots)
        if mst_map:
            df_out = plots_base / f"mst_{run_root.name}.csv"
            with df_out.open('w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["input_tokens", "output_tokens", "mst_req_min"])
                for (inp, outp), val in sorted(mst_map.items()):
                    writer.writerow([inp, outp, val])

        # Plot per-run heatmap
        matrix, inputs, outputs = build_matrix(mst_map)
        plot_heatmap(
            matrix,
            inputs,
            outputs,
            title=f"MST Heatmap - {run_root.name}",
            out_path=plots_base / f"mst_heatmap_{run_root.name}.png",
        )

    # Build average across runs
    # Collect all token pairs observed across runs
    all_pairs = set()
    for m in per_run_mst.values():
        all_pairs.update(m.keys())

    if not all_pairs:
        print("No MST data found to compute average.")
        return

    # For each pair, average across runs (ignore missing)
    avg_map: Dict[Tuple[int, int], float] = {}
    for pair in sorted(all_pairs):
        vals = [m[pair] for m in per_run_mst.values() if pair in m]
        if vals:
            avg_map[pair] = float(np.mean(vals))

    # Save average CSV
    avg_out = plots_base / "mst_average.csv"
    with avg_out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["input_tokens", "output_tokens", "avg_mst_req_min"])
        for (inp, outp), val in sorted(avg_map.items()):
            writer.writerow([inp, outp, val])

    # Plot average heatmap
    matrix, inputs, outputs = build_matrix(avg_map)
    plot_heatmap(
        matrix,
        inputs,
        outputs,
        title="MST Heatmap - Average Across Runs",
        out_path=plots_base / "mst_heatmap_average.png",
    )

    # Detect outliers across runs and plot cleaned heatmaps
    flagged = compute_outlier_flags(
        per_run_mst,
        method=method,
        z_thresh=z_thresh,
        iqr_k=iqr_k,
        ratio_upper=ratio_upper,
        ratio_lower=ratio_lower,
        log_space=log_space,
        min_runs=min_runs,
        report_path=plots_base / "mst_outliers_report.csv",
    )

    # Manual overrides
    # These are passed via global args parsed in __main__; to keep main() pure and testable,
    # read from environment variables set by wrapper below. This avoids threading args everywhere.
    ignore_pairs_env = os.environ.get("MST_IGNORE_PAIRS", "")
    drop_above_env = os.environ.get("MST_DROP_ABOVE", "")
    ignore_file_env = os.environ.get("MST_IGNORE_FILE", "")

    def parse_pairs(s: str) -> set:
        out = set()
        for part in s.split(','):
            part = part.strip()
            if not part:
                continue
            if ':' in part:
                a, b = part.split(':', 1)
            elif '_' in part:
                a, b = part.split('_', 1)
            else:
                continue
            try:
                out.add((int(a), int(b)))
            except ValueError:
                continue
        return out

    manual_pairs = parse_pairs(ignore_pairs_env)
    if ignore_file_env:
        try:
            pth = Path(ignore_file_env)
            if pth.exists():
                with pth.open(newline='', encoding='utf-8') as f:
                    rdr = csv.DictReader(f)
                    for row in rdr:
                        try:
                            manual_pairs.add((int(row['input_tokens']), int(row['output_tokens'])))
                        except Exception:
                            pass
        except Exception:
            pass

    drop_above_val: Optional[float] = None
    try:
        drop_above_val = float(drop_above_env) if drop_above_env else None
    except Exception:
        drop_above_val = None

    # Augment flagged set with manual criteria
    if manual_pairs or (drop_above_val is not None):
        for run_name, mapping in per_run_mst.items():
            for pair, v in mapping.items():
                if pair in manual_pairs:
                    flagged.add((run_name, pair))
                if drop_above_val is not None and v > drop_above_val:
                    flagged.add((run_name, pair))

    # Per-run cleaned heatmaps (mask flagged values)
    for run_root in sorted(run_roots, key=lambda p: p.name):
        run_name = run_root.name
        orig_map = per_run_mst.get(run_name, {})
        clean_map = {pair: v for pair, v in orig_map.items() if (run_name, pair) not in flagged}
        matrix_c, inputs_c, outputs_c = build_matrix(clean_map)
        plot_heatmap(
            matrix_c,
            inputs_c,
            outputs_c,
            title=f"MST Heatmap - {run_name} (clean)",
            out_path=plots_base / f"mst_heatmap_{run_name}_clean.png",
        )

    # Cleaned average across runs (ignore missing; drop flagged or winsorize)
    avg_map_clean: Dict[Tuple[int, int], float] = {}
    for pair in sorted(all_pairs):
        vals: List[float] = []
        for run_name, m in per_run_mst.items():
            if pair not in m:
                continue
            v = m[pair]
            if (run_name, pair) in flagged:
                if winsorize:
                    vals.append(v)  # keep, will cap later
                else:
                    continue  # drop outlier for clean average
            else:
                vals.append(v)

        if not vals:
            continue

        if winsorize and len(vals) >= 3:
            lo_q, hi_q = winsor_quantiles
            lo, hi = np.quantile(vals, [lo_q, hi_q])
            vals = [min(max(x, lo), hi) for x in vals]

        avg_map_clean[pair] = float(np.mean(vals))

    matrix_ac, inputs_ac, outputs_ac = build_matrix(avg_map_clean)
    plot_heatmap(
        matrix_ac,
        inputs_ac,
        outputs_ac,
        title="MST Heatmap - Average Across Runs (clean)",
        out_path=plots_base / "mst_heatmap_average_clean.png",
    )

    # Min and Max heatmaps across runs (raw)
    min_map: Dict[Tuple[int, int], float] = {}
    max_map: Dict[Tuple[int, int], float] = {}
    for pair in sorted(all_pairs):
        vals = [m[pair] for m in per_run_mst.values() if pair in m]
        if vals:
            min_map[pair] = float(np.min(vals))
            max_map[pair] = float(np.max(vals))

    matrix_min, inputs_min, outputs_min = build_matrix(min_map)
    plot_heatmap(
        matrix_min,
        inputs_min,
        outputs_min,
        title="MST Heatmap - Min Across Runs",
        out_path=plots_base / "mst_heatmap_min.png",
    )

    matrix_max, inputs_max, outputs_max = build_matrix(max_map)
    plot_heatmap(
        matrix_max,
        inputs_max,
        outputs_max,
        title="MST Heatmap - Max Across Runs",
        out_path=plots_base / "mst_heatmap_max.png",
    )

    # Min and Max heatmaps across runs (clean: exclude flagged only)
    min_map_clean: Dict[Tuple[int, int], float] = {}
    max_map_clean: Dict[Tuple[int, int], float] = {}
    for pair in sorted(all_pairs):
        vals_clean = [m[pair] for run_name, m in per_run_mst.items() if pair in m and (run_name, pair) not in flagged]
        if vals_clean:
            min_map_clean[pair] = float(np.min(vals_clean))
            max_map_clean[pair] = float(np.max(vals_clean))

    matrix_min_c, inputs_min_c, outputs_min_c = build_matrix(min_map_clean)
    plot_heatmap(
        matrix_min_c,
        inputs_min_c,
        outputs_min_c,
        title="MST Heatmap - Min Across Runs (clean)",
        out_path=plots_base / "mst_heatmap_min_clean.png",
    )

    matrix_max_c, inputs_max_c, outputs_max_c = build_matrix(max_map_clean)
    plot_heatmap(
        matrix_max_c,
        inputs_max_c,
        outputs_max_c,
        title="MST Heatmap - Max Across Runs (clean)",
        out_path=plots_base / "mst_heatmap_max_clean.png",
    )

    print("Done. Generated raw and cleaned heatmaps, plus min/max across runs.")

    # -----------------------------------------------
    # Spike minute heatmaps (using failing iterations)
    # -----------------------------------------------
    print("Computing spike minute heatmaps (failing iterations)...")
    per_run_spike: Dict[str, Dict[Tuple[int, int], int]] = {}
    for run_root in sorted(run_roots, key=lambda p: p.name):
        run_name = run_root.name
        spikes_map = collect_spike_minutes_for_run(
            run_root,
            spike_factor=spike_factor,
            spike_mad_k=spike_mad_k,
            spike_min_obs=spike_min_obs,
            spike_fallback=spike_fallback,
        )
        per_run_spike[run_name] = spikes_map

        # Save per-run spikes CSV
        if spikes_map:
            out_csv = plots_base / f"spike_minutes_{run_name}.csv"
            with out_csv.open('w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(["input_tokens", "output_tokens", "spike_minute"])
                for (inp, outp), minute in sorted(spikes_map.items()):
                    w.writerow([inp, outp, minute])

        # Plot per-run spikes heatmap
        smatrix, sinputs, soutputs = build_matrix({k: float(v) for k, v in spikes_map.items()})
        plot_heatmap(
            smatrix,
            sinputs,
            soutputs,
            title=f"Spike Minute Heatmap - {run_name}",
            out_path=plots_base / f"spike_heatmap_{run_name}.png",
        )

        # Plot per-run spikes heatmap (clean): exclude flagged (run, pair)
        spikes_map_clean = {pair: v for pair, v in spikes_map.items() if (run_name, pair) not in flagged}
        smatrix_c, sinputs_c, soutputs_c = build_matrix({k: float(v) for k, v in spikes_map_clean.items()})
        plot_heatmap(
            smatrix_c,
            sinputs_c,
            soutputs_c,
            title=f"Spike Minute Heatmap - {run_name} (clean)",
            out_path=plots_base / f"spike_heatmap_{run_name}_clean.png",
        )

    # Average spike minute across runs (where available)
    all_spike_pairs = set()
    for m in per_run_spike.values():
        all_spike_pairs.update(m.keys())

    if all_spike_pairs:
        avg_spike: Dict[Tuple[int, int], float] = {}
        for pair in sorted(all_spike_pairs):
            vals = [m[pair] for m in per_run_spike.values() if pair in m]
            if vals:
                avg_spike[pair] = float(np.mean(vals))
        smatrix, sinputs, soutputs = build_matrix(avg_spike)
        plot_heatmap(
            smatrix,
            sinputs,
            soutputs,
            title="Spike Minute Heatmap - Average Across Runs",
            out_path=plots_base / "spike_heatmap_average.png",
        )

        # Clean average spike minutes (exclude flagged)
        avg_spike_clean: Dict[Tuple[int, int], float] = {}
        for pair in sorted(all_spike_pairs):
            vals_c = [m[pair] for run_name, m in per_run_spike.items() if pair in m and (run_name, pair) not in flagged]
            if vals_c:
                avg_spike_clean[pair] = float(np.mean(vals_c))
        smatrix_c, sinputs_c, soutputs_c = build_matrix(avg_spike_clean)
        plot_heatmap(
            smatrix_c,
            sinputs_c,
            soutputs_c,
            title="Spike Minute Heatmap - Average Across Runs (clean)",
            out_path=plots_base / "spike_heatmap_average_clean.png",
        )

        # Min/Max spike minute across runs (raw)
        min_spike: Dict[Tuple[int, int], float] = {}
        max_spike: Dict[Tuple[int, int], float] = {}
        for pair in sorted(all_spike_pairs):
            vals = [m[pair] for m in per_run_spike.values() if pair in m]
            if vals:
                min_spike[pair] = float(np.min(vals))
                max_spike[pair] = float(np.max(vals))
        smatrix_min, sinputs_min, soutputs_min = build_matrix(min_spike)
        plot_heatmap(
            smatrix_min,
            sinputs_min,
            soutputs_min,
            title="Spike Minute Heatmap - Min Across Runs",
            out_path=plots_base / "spike_heatmap_min.png",
        )
        smatrix_max, sinputs_max, soutputs_max = build_matrix(max_spike)
        plot_heatmap(
            smatrix_max,
            sinputs_max,
            soutputs_max,
            title="Spike Minute Heatmap - Max Across Runs",
            out_path=plots_base / "spike_heatmap_max.png",
        )

        # Min/Max spike minute across runs (clean)
        min_spike_c: Dict[Tuple[int, int], float] = {}
        max_spike_c: Dict[Tuple[int, int], float] = {}
        for pair in sorted(all_spike_pairs):
            vals_c = [m[pair] for run_name, m in per_run_spike.items() if pair in m and (run_name, pair) not in flagged]
            if vals_c:
                min_spike_c[pair] = float(np.min(vals_c))
                max_spike_c[pair] = float(np.max(vals_c))
        smatrix_min_c, sinputs_min_c, soutputs_min_c = build_matrix(min_spike_c)
        plot_heatmap(
            smatrix_min_c,
            sinputs_min_c,
            soutputs_min_c,
            title="Spike Minute Heatmap - Min Across Runs (clean)",
            out_path=plots_base / "spike_heatmap_min_clean.png",
        )
        smatrix_max_c, sinputs_max_c, soutputs_max_c = build_matrix(max_spike_c)
        plot_heatmap(
            smatrix_max_c,
            sinputs_max_c,
            soutputs_max_c,
            title="Spike Minute Heatmap - Max Across Runs (clean)",
            out_path=plots_base / "spike_heatmap_max_clean.png",
        )
    else:
        print("No failing iterations found for spike minute heatmaps.")

    # -----------------------------------------------------
    # Avg response time heatmaps for MST iterations (True)
    # -----------------------------------------------------
    print("Computing average response time heatmaps for MST iterations...")
    per_run_mst_dirs: Dict[str, Dict[Tuple[int, int], Path]] = {}
    per_run_avg_rt: Dict[str, Dict[Tuple[int, int], float]] = {}
    for run_root in sorted(run_roots, key=lambda p: p.name):
        run_name = run_root.name
        mst_dirs = collect_mst_iteration_dirs_for_run(run_root)
        per_run_mst_dirs[run_name] = mst_dirs
        avg_map: Dict[Tuple[int, int], float] = {}
        for pair, ts_dir in mst_dirs.items():
            avg_rt = compute_avg_response_time_for_iteration(ts_dir)
            if avg_rt is not None:
                avg_map[pair] = avg_rt
        per_run_avg_rt[run_name] = avg_map

        # Save per-run avg RT CSV
        if avg_map:
            out_csv = plots_base / f"avg_rt_mst_{run_name}.csv"
            with out_csv.open('w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(["input_tokens", "output_tokens", "avg_response_time_ms"])
                for (inp, outp), val in sorted(avg_map.items()):
                    w.writerow([inp, outp, val])

        # Plot per-run avg RT heatmap (raw)
        rmat, rinputs, routputs = build_matrix(avg_map)
        plot_heatmap(
            rmat,
            rinputs,
            routputs,
            title=f"Avg Response Time (ms) - {run_name} MST",
            out_path=plots_base / f"avg_rt_mst_heatmap_{run_name}.png",
        )

        # Plot per-run avg RT heatmap (clean): exclude flagged
        avg_map_clean = {pair: v for pair, v in avg_map.items() if (run_name, pair) not in flagged}
        rmat_c, rinputs_c, routputs_c = build_matrix(avg_map_clean)
        plot_heatmap(
            rmat_c,
            rinputs_c,
            routputs_c,
            title=f"Avg Response Time (ms) - {run_name} MST (clean)",
            out_path=plots_base / f"avg_rt_mst_heatmap_{run_name}_clean.png",
        )

    # Average across runs (raw)
    all_pairs_rt = set()
    for m in per_run_avg_rt.values():
        all_pairs_rt.update(m.keys())

    if all_pairs_rt:
        avg_rt_over_runs: Dict[Tuple[int, int], float] = {}
        for pair in sorted(all_pairs_rt):
            vals = [m[pair] for m in per_run_avg_rt.values() if pair in m]
            if vals:
                avg_rt_over_runs[pair] = float(np.mean(vals))
        hmat, hinputs, houtputs = build_matrix(avg_rt_over_runs)
        plot_heatmap(
            hmat,
            hinputs,
            houtputs,
            title="Avg Response Time (ms) - Average Across Runs (MST)",
            out_path=plots_base / "avg_rt_mst_heatmap_average.png",
        )

        # Average across runs (clean)
        avg_rt_over_runs_clean: Dict[Tuple[int, int], float] = {}
        for pair in sorted(all_pairs_rt):
            vals_c = [m[pair] for run_name, m in per_run_avg_rt.items() if pair in m and (run_name, pair) not in flagged]
            if vals_c:
                avg_rt_over_runs_clean[pair] = float(np.mean(vals_c))
        hmat_c, hinputs_c, houtputs_c = build_matrix(avg_rt_over_runs_clean)
        plot_heatmap(
            hmat_c,
            hinputs_c,
            houtputs_c,
            title="Avg Response Time (ms) - Average Across Runs (MST, clean)",
            out_path=plots_base / "avg_rt_mst_heatmap_average_clean.png",
        )

        # Min/Max across runs (raw)
        min_rt_map: Dict[Tuple[int, int], float] = {}
        max_rt_map: Dict[Tuple[int, int], float] = {}
        for pair in sorted(all_pairs_rt):
            vals = [m[pair] for m in per_run_avg_rt.values() if pair in m]
            if vals:
                min_rt_map[pair] = float(np.min(vals))
                max_rt_map[pair] = float(np.max(vals))
        mmat, minputs, moutputs = build_matrix(min_rt_map)
        plot_heatmap(
            mmat,
            minputs,
            moutputs,
            title="Avg Response Time (ms) - Min Across Runs (MST)",
            out_path=plots_base / "avg_rt_mst_heatmap_min.png",
        )
        xmat, xinputs, xoutputs = build_matrix(max_rt_map)
        plot_heatmap(
            xmat,
            xinputs,
            xoutputs,
            title="Avg Response Time (ms) - Max Across Runs (MST)",
            out_path=plots_base / "avg_rt_mst_heatmap_max.png",
        )

        # Min/Max across runs (clean)
        min_rt_map_c: Dict[Tuple[int, int], float] = {}
        max_rt_map_c: Dict[Tuple[int, int], float] = {}
        for pair in sorted(all_pairs_rt):
            vals_c = [m[pair] for run_name, m in per_run_avg_rt.items() if pair in m and (run_name, pair) not in flagged]
            if vals_c:
                min_rt_map_c[pair] = float(np.min(vals_c))
                max_rt_map_c[pair] = float(np.max(vals_c))
        mmat_c, minputs_c, moutputs_c = build_matrix(min_rt_map_c)
        plot_heatmap(
            mmat_c,
            minputs_c,
            moutputs_c,
            title="Avg Response Time (ms) - Min Across Runs (MST, clean)",
            out_path=plots_base / "avg_rt_mst_heatmap_min_clean.png",
        )
        xmat_c, xinputs_c, xoutputs_c = build_matrix(max_rt_map_c)
        plot_heatmap(
            xmat_c,
            xinputs_c,
            xoutputs_c,
            title="Avg Response Time (ms) - Max Across Runs (MST, clean)",
            out_path=plots_base / "avg_rt_mst_heatmap_max_clean.png",
        )
    else:
        print("No MST iterations found with response time data.")

    # -----------------------------------------------------------
    # Response count heatmaps for MST iterations (row counts)
    # -----------------------------------------------------------
    print("Computing responses-per-minute heatmaps for MST iterations...")
    # Reuse per_run_mst_dirs if available; otherwise, rebuild
    try:
        per_run_mst_dirs
    except NameError:
        per_run_mst_dirs = {}
        for run_root in sorted(run_roots, key=lambda p: p.name):
            per_run_mst_dirs[run_root.name] = collect_mst_iteration_dirs_for_run(run_root)

    per_run_resp_count: Dict[str, Dict[Tuple[int, int], float]] = {}
    for run_root in sorted(run_roots, key=lambda p: p.name):
        run_name = run_root.name
        mst_dirs = per_run_mst_dirs.get(run_name, {})
        cnt_map: Dict[Tuple[int, int], float] = {}
        for pair, ts_dir in mst_dirs.items():
            rpm = compute_response_rate_per_min_for_iteration(ts_dir)
            if rpm is not None:
                cnt_map[pair] = rpm
        per_run_resp_count[run_name] = cnt_map

        # Save per-run response count CSV
        if cnt_map:
            out_csv = plots_base / f"resp_count_mst_{run_name}.csv"
            with out_csv.open('w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(["input_tokens", "output_tokens", "responses_per_minute"])
                for (inp, outp), val in sorted(cnt_map.items()):
                    w.writerow([inp, outp, val])

        # Plot per-run response count heatmap (raw)
        cmat, cinp, cout = build_matrix({k: float(v) for k, v in cnt_map.items()})
        plot_heatmap(
            cmat,
            cinp,
            cout,
            title=f"Responses per Minute - {run_name} MST",
            out_path=plots_base / f"resp_count_mst_heatmap_{run_name}.png",
            cbar_label="Responses/min",
        )

        # Plot per-run response count heatmap (clean): exclude flagged
        cnt_map_clean = {pair: v for pair, v in cnt_map.items() if (run_name, pair) not in flagged}
        cmat_c, cinp_c, cout_c = build_matrix({k: float(v) for k, v in cnt_map_clean.items()})
        plot_heatmap(
            cmat_c,
            cinp_c,
            cout_c,
            title=f"Responses per Minute - {run_name} MST (clean)",
            out_path=plots_base / f"resp_count_mst_heatmap_{run_name}_clean.png",
            cbar_label="Responses/min",
        )

    # Average response count across runs (raw)
    all_pairs_cnt = set()
    for m in per_run_resp_count.values():
        all_pairs_cnt.update(m.keys())

    if all_pairs_cnt:
        avg_cnt_over_runs: Dict[Tuple[int, int], float] = {}
        for pair in sorted(all_pairs_cnt):
            vals = [m[pair] for m in per_run_resp_count.values() if pair in m]
            if vals:
                avg_cnt_over_runs[pair] = float(np.mean(vals))

        # Save average CSV
        out_avg_csv = plots_base / "resp_count_mst_average.csv"
        with out_avg_csv.open('w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(["input_tokens", "output_tokens", "avg_responses_per_minute"])
            for (inp, outp), val in sorted(avg_cnt_over_runs.items()):
                w.writerow([inp, outp, val])

        hcmat, hcinp, hcout = build_matrix(avg_cnt_over_runs)
        plot_heatmap(
            hcmat,
            hcinp,
            hcout,
            title="Responses per Minute - Average Across Runs (MST)",
            out_path=plots_base / "resp_count_mst_heatmap_average.png",
            cbar_label="Responses/min",
        )

        # Average across runs (clean)
        avg_cnt_over_runs_clean: Dict[Tuple[int, int], float] = {}
        for pair in sorted(all_pairs_cnt):
            vals_c = [m[pair] for run_name, m in per_run_resp_count.items() if pair in m and (run_name, pair) not in flagged]
            if vals_c:
                avg_cnt_over_runs_clean[pair] = float(np.mean(vals_c))
        hcmat_c, hcinp_c, hcout_c = build_matrix(avg_cnt_over_runs_clean)
        plot_heatmap(
            hcmat_c,
            hcinp_c,
            hcout_c,
            title="Responses per Minute - Average Across Runs (MST, clean)",
            out_path=plots_base / "resp_count_mst_heatmap_average_clean.png",
            cbar_label="Responses/min",
        )
    else:
        print("No MST iterations found with response count data.")


if __name__ == "__main__":
    args = parse_args()
    winsor_q = tuple(map(float, args.winsor_quantiles.split(','))) if isinstance(args.winsor_quantiles, str) else args.winsor_quantiles
    # Pass manual override parameters via environment to keep main signature stable
    if args.ignore_pairs:
        os.environ["MST_IGNORE_PAIRS"] = args.ignore_pairs
    if args.ignore_file:
        os.environ["MST_IGNORE_FILE"] = args.ignore_file
    if args.drop_above is not None:
        os.environ["MST_DROP_ABOVE"] = str(args.drop_above)
    main(
        base_dir=args.base_dir,
        data_dir=args.data_dir,
        plots_dir=args.plots_dir,
        method=args.method,
        z_thresh=args.z_thresh,
        iqr_k=args.iqr_k,
        ratio_upper=args.ratio_upper,
        ratio_lower=args.ratio_lower,
        log_space=args.log_space,
        min_runs=args.min_runs,
        winsorize=args.winsorize,
        winsor_quantiles=winsor_q,  # type: ignore[assignment]
        spike_factor=args.spike_factor,
        spike_mad_k=args.spike_mad_k,
        spike_min_obs=args.spike_min_obs,
        spike_fallback=args.spike_fallback,
    )
