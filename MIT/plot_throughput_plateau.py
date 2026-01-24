import os
import re
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Pattern to capture the tokens folder (e.g., 16_16) from the log line
RESULTS_PATH_RE = re.compile(r"writing results to file: .*?/(\d+_\d+)/result_sweep_u\d+\.json", re.IGNORECASE)

# Header detection (we care about locating the 'throughput' column index)
THROUGHPUT_HEADER_RE = re.compile(r"\bthroughput\b", re.IGNORECASE)

# Row lines start with the u value (10, 50, 100, ...), then columns
ROW_LINE_RE = re.compile(r"^\s*(\d+)\s+")


def _format_k(value, _):
    """Format numbers so thousands show as k (e.g., 1000 -> 1k, 1500 -> 1.5k)."""
    abs_val = abs(value)
    if abs_val >= 1000:
        val = value / 1000.0
        # Drop trailing .0 for clean integers
        if abs(val - int(val)) < 1e-9:
            return f"{int(val)}k"
        return f"{val:.1f}k"
    # For non-thousands, prefer integer display when applicable
    if abs(value - int(value)) < 1e-9:
        return f"{int(value)}"
    return f"{value:.1f}"


def parse_slurm_file(filepath):
    """Parse a slurm-*.out file to extract throughput per \n
    tokens folder (e.g., 16_16, 32_16, ...) and per load (u).

    Returns: dict mapping tokens_dir -> dict mapping u -> throughput
    """
    tokens_dir = None
    header_cols = None
    throughput_idx = None
    results = defaultdict(dict)

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Capture current tokens_dir when we enter a new results block
                m_path = RESULTS_PATH_RE.search(line)
                if m_path:
                    tokens_dir = m_path.group(1)
                    # reset header context for new block
                    header_cols = None
                    throughput_idx = None
                    continue

                # Detect the header line containing 'throughput'
                if tokens_dir and THROUGHPUT_HEADER_RE.search(line):
                    # Normalize multiple spaces and backslashes
                    header_cols = [c for c in line.strip().replace('\\', '').split()]
                    # Identify the index of 'throughput' column
                    try:
                        throughput_idx = header_cols.index('throughput')
                    except ValueError:
                        # Sometimes capitalization or spacing differs
                        throughput_idx = next((i for i, c in enumerate(header_cols) if c.lower() == 'throughput'), None)
                    continue

                # Parse data rows while in a block with known header
                if tokens_dir and header_cols and throughput_idx is not None:
                    m_row = ROW_LINE_RE.match(line)
                    if m_row:
                        parts = [p for p in line.strip().replace('\\', '').split()]
                        # The first token is the u value (row index in the printed table)
                        try:
                            u_val = int(parts[0])
                        except Exception:
                            continue
                        # Align parts with header; some tables include row index as the first value
                        # So throughput is at throughput_idx + 1 in parts (since row index precedes columns)
                        col_offset = 1
                        idx = throughput_idx + col_offset
                        if idx < len(parts):
                            try:
                                throughput = float(parts[idx])
                                results[tokens_dir][u_val] = throughput
                            except Exception:
                                # Skip malformed rows
                                pass
                        continue

                # Reset when a latency header shows up (end of the throughput table block)
                if tokens_dir and ('latency_prefill_ms' in line or 'latency_nexttoken_ms' in line):
                    header_cols = None
                    throughput_idx = None
                    # Do not clear tokens_dir; subsequent blocks might relate to same or new tokens_dir
                    continue
    except FileNotFoundError:
        pass

    return results


def collect_all_results(base_dir):
    """Walk the base_dir and parse all slurm-*.out files.
    Returns: dict tokens_dir -> dict u -> throughput
    """
    aggregate = defaultdict(dict)

    # Find slurm logs in model/gpu folders directly under base_dir
    for root, dirs, files in os.walk(base_dir):
        for fname in files:
            if fname.startswith('slurm-') and fname.endswith('.out'):
                fpath = os.path.join(root, fname)
                parsed = parse_slurm_file(fpath)
                # Merge into aggregate; prefer larger u entries when duplicates conflict
                for tdir, series in parsed.items():
                    for u, thr in series.items():
                        # If conflict, prefer the latest parsed value
                        aggregate[tdir][u] = thr
    return aggregate


def group_by_input(aggregate):
    """Group aggregate series by input token count (XXX from 'XXX_YYY').

    Returns: dict[int, dict[str, dict[int, float]]]
             maps in_tokens -> tokens_dir -> {u -> throughput}
    """
    groups = defaultdict(dict)
    for tokens_dir, series in aggregate.items():
        try:
            x_str, y_str = tokens_dir.split('_')
            in_tokens = int(x_str)
        except Exception:
            # Skip unexpected folder names
            continue
        groups[in_tokens][tokens_dir] = series
    return groups


def plot_per_input(in_tokens, series_map, base_title, out_path):
    """Plot throughput vs load (u) for all YYY under a given input token count."""
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Sort by output tokens (YYY) for consistent coloring
    def out_key(tdir):
        try:
            _, y = tdir.split('_')
            return int(y)
        except Exception:
            return 0

    for tokens_dir in sorted(series_map.keys(), key=out_key):
        series = series_map[tokens_dir]
        if not series:
            continue
        # Only use up to 500 concurrent users for plots
        max_u = 500
        xs = sorted(u for u in series.keys() if u <= max_u)
        if not xs:
            continue
        ys = [series[u] for u in xs]
        # Label by output tokens only to reduce clutter
        try:
            _, y = tokens_dir.split('_')
            label = f"out={y}"
        except Exception:
            label = tokens_dir
        plt.plot(xs, ys, marker='o', linewidth=2, label=label)

    # Bigger text for labels and ticks (keep title default size)
    plt.xlabel('Concurrent requests', fontsize=16)
    plt.ylabel('Throughput (tokens/sec)', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Format thousands as k on the Y axis (throughput)
    ax.yaxis.set_major_formatter(FuncFormatter(_format_k))
    plt.title(f"{base_title} — input tokens: {in_tokens}", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(ncol=3, fontsize=14, title='Output tokens', title_fontsize=14)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot throughput vs load from MIT slurm logs.')
    parser.add_argument('--base-dir', default=os.path.join('MIT', 'data', 'deepseek 7B - a40'),
                        help='Base directory containing model/gpu results with XXX_YYY folders and slurm logs.')
    parser.add_argument('--out-dir', default=os.path.join('MIT', 'plots', 'deepseek 7B - a40'),
                        help='Directory to save per-input plots.')
    args = parser.parse_args()

    aggregate = collect_all_results(args.base_dir)

    if not aggregate:
        print('No throughput data found. Ensure slurm-*.out logs exist and contain summary tables.')
        return

    base_title = f"Throughput vs Load (u) — {os.path.basename(args.base_dir)}"
    groups = group_by_input(aggregate)

    # Save one figure per input token count
    for in_tokens, series_map in sorted(groups.items()):
        fname = f"throughput_vs_load_{os.path.basename(args.base_dir).replace(' ', '_')}_in_{in_tokens}.png"
        out_path = os.path.join(args.out_dir, fname)
        plot_per_input(in_tokens, series_map, base_title, out_path)


if __name__ == '__main__':
    main()
