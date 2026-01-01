import os
import re
import shutil
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

DATE_FMT = "%Y-%m-%d_%H-%M-%S"
COMBO_DIR_RE = re.compile(r"^(\d+)_(\d+)$")
DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")


def parse_timestamp(name: str) -> datetime:
    try:
        return datetime.strptime(name, DATE_FMT)
    except ValueError:
        return None


def find_combo_dirs(base_dir: str) -> List[str]:
    try:
        entries = os.listdir(base_dir)
    except FileNotFoundError:
        raise RuntimeError(f"Base directory not found: {base_dir}")

    combo_dirs = []
    for entry in entries:
        full_path = os.path.join(base_dir, entry)
        if os.path.isdir(full_path) and COMBO_DIR_RE.match(entry):
            combo_dirs.append(full_path)
    return sorted(combo_dirs)


def gather_dated_subdirs(base_dir: str) -> List[Dict]:
    """Collect all dated subdirectories under each combo directory.

    Returns list of dicts: { 'combo': '128_128', 'path': full_path, 'timestamp': datetime }
    """
    items = []
    for combo_dir in find_combo_dirs(base_dir):
        combo_name = os.path.basename(combo_dir)
        try:
            subentries = os.listdir(combo_dir)
        except FileNotFoundError:
            # Skip if dir disappears mid-run
            continue
        for sub in subentries:
            sub_path = os.path.join(combo_dir, sub)
            if not os.path.isdir(sub_path):
                continue
            if not DATE_DIR_RE.match(sub):
                continue
            ts = parse_timestamp(sub)
            if ts is None:
                continue
            items.append({
                "combo": combo_name,
                "path": sub_path,
                "timestamp": ts,
            })
    # Sort globally by time
    items.sort(key=lambda d: d["timestamp"])
    return items


def compute_run_starts(items: List[Dict], start_marker_combos: List[str] | Tuple[str, ...] = ("128_128",), min_week_gap_hours: int = 120) -> List[datetime]:
    """Determine weekly run start times from first occurrences of start markers.

    Rules:
    - Consider timestamps where `combo` is in `start_marker_combos`.
    - Choose the first timestamp as a start, then only accept a new start
      when the gap from the previous accepted start exceeds `min_week_gap_hours`.
    """
    start_ts_all = [entry["timestamp"] for entry in items if entry["combo"] in start_marker_combos]
    start_ts_all.sort()
    if not start_ts_all:
        return []

    accepted: List[datetime] = []
    prev: datetime | None = None
    for ts in start_ts_all:
        if prev is None:
            accepted.append(ts)
            prev = ts
            continue
        gap = ts - prev
        if gap.total_seconds() > min_week_gap_hours * 3600:
            accepted.append(ts)
            prev = ts
    return accepted


def partition_runs_by_starts(items: List[Dict], start_marker_combos: List[str] | Tuple[str, ...] = ("128_128",), min_week_gap_hours: int = 120) -> List[List[Dict]]:
    """Partition items into runs using weekly start markers.

    Each run is the half-open interval [start_i, start_{i+1}), using accepted start
    times computed with `min_week_gap_hours`. The last run is [start_last, +inf).
    """
    starts = compute_run_starts(items, start_marker_combos=start_marker_combos, min_week_gap_hours=min_week_gap_hours)
    if not starts:
        # Fallback: single run containing all items
        return [items[:]] if items else []

    runs: List[List[Dict]] = []
    for i, start_ts in enumerate(starts):
        end_ts = starts[i + 1] if (i + 1) < len(starts) else None
        window: List[Dict] = []
        for entry in items:
            ts = entry["timestamp"]
            if end_ts is None:
                if ts >= start_ts:
                    window.append(entry)
            else:
                if start_ts <= ts < end_ts:
                    window.append(entry)
        runs.append(window)
    return runs


def partition_runs(items: List[Dict], gap_hours: int, end_marker_combos: List[str] | Tuple[str, ...] = ("2048_2048", "2048_1024"), use_end_marker: bool = True, end_gap_hours: int = 24, start_marker_combos: List[str] | Tuple[str, ...] = ("128_128",), start_gap_hours: int = 12) -> List[List[Dict]]:
    """Partition globally sorted items into runs.

    A new run starts when:
    - The time gap from previous item exceeds `gap_hours`, OR
    - The previous item closed a run by being an end marker combo if enabled.

    If an end marker appears, we close the current run immediately after including that item.
    """
    runs: List[List[Dict]] = []
    current: List[Dict] = []
    prev_ts: datetime | None = None

    for idx, entry in enumerate(items):
        ts = entry["timestamp"]
        combo = entry["combo"]

        # Start new run if empty
        if not current:
            current.append(entry)
            prev_ts = ts
            # If first entry itself is end marker, close instantly
            if use_end_marker and combo in end_marker_combos:
                runs.append(current)
                current = []
                prev_ts = None
            continue

        # If a start marker appears after a sufficient gap, start a new run here
        if prev_ts is not None and combo in start_marker_combos and (ts - prev_ts).total_seconds() > start_gap_hours * 3600:
            runs.append(current)
            current = [entry]
            prev_ts = ts
            continue

        # Check gap threshold between previous and current
        assert prev_ts is not None
        gap = ts - prev_ts
        gap_exceeds = gap.total_seconds() > gap_hours * 3600

        if gap_exceeds:
            # Close current run and start a new one
            runs.append(current)
            current = [entry]
            prev_ts = ts
            # If the new first entry is end marker, consider closing based on lookahead
            if use_end_marker and combo in end_marker_combos:
                next_ts = items[idx + 1]["timestamp"] if (idx + 1) < len(items) else None
                if next_ts is None or (next_ts - ts).total_seconds() > end_gap_hours * 3600:
                    runs.append(current)
                    current = []
                    prev_ts = None
                    continue
            continue

        # Same run
        current.append(entry)
        prev_ts = ts

        # Close if end marker and followed by a sufficiently large gap (or end of list)
        if use_end_marker and combo in end_marker_combos:
            next_ts = items[idx + 1]["timestamp"] if (idx + 1) < len(items) else None
            if next_ts is None or (next_ts - ts).total_seconds() > end_gap_hours * 3600:
                runs.append(current)
                current = []
                prev_ts = None

    # Flush remaining
    if current:
        runs.append(current)

    return runs


def summarize_runs(runs: List[List[Dict]]) -> List[Tuple[datetime, datetime, int]]:
    summary = []
    for run in runs:
        if not run:
            summary.append((None, None, 0))
            continue
        start = run[0]["timestamp"]
        end = run[-1]["timestamp"]
        summary.append((start, end, len(run)))
    return summary


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def suggest_run_names(parent_dir: str, base_name: str, count: int, start_index: int = 1) -> List[str]:
    names = []
    idx = start_index
    while len(names) < count:
        name = f"{base_name}_{idx}"
        full = os.path.join(parent_dir, name)
        if not os.path.exists(full):
            names.append(full)
        idx += 1
    return names


def move_runs(runs: List[List[Dict]], base_dir: str, apply: bool, start_index: int, base_name_override: str | None = None) -> List[Tuple[str, List[Tuple[str, str]]]]:
    parent = os.path.dirname(os.path.abspath(base_dir))
    base_name = base_name_override or os.path.basename(os.path.abspath(base_dir))

    run_roots = suggest_run_names(parent, base_name, len(runs), start_index=start_index)

    move_plans: List[Tuple[str, List[Tuple[str, str]]]] = []

    for run_idx, run in enumerate(runs):
        run_root = run_roots[run_idx]
        plans: List[Tuple[str, str]] = []
        for entry in run:
            combo = entry["combo"]
            ts_name = entry["path"].split(os.sep)[-1]
            dest = os.path.join(run_root, combo, ts_name)
            plans.append((entry["path"], dest))
        move_plans.append((run_root, plans))

    # Execute moves if requested
    if apply:
        for run_root, plans in move_plans:
            ensure_dir(run_root)
            for src, dest in plans:
                ensure_dir(os.path.dirname(dest))
                if os.path.exists(dest):
                    raise RuntimeError(f"Destination already exists: {dest}")
                shutil.move(src, dest)

    return move_plans


def main():
    parser = argparse.ArgumentParser(description="Split experiment runs into numbered directories using weekly starts (e.g., first 128_128).")
    parser.add_argument("--base", default="requests_deepseek", help="Base directory containing combo subfolders (e.g., 'requests_deepseek').")
    parser.add_argument("--gap-hours", type=int, default=120, help="(Fallback mode) Gap threshold (hours) to start a new run. Defaults to 120 (5 days).")
    parser.add_argument("--use-end-markers", action="store_true", help="Use end-marker based partitioning instead of weekly starts.")
    parser.add_argument("--end-markers", default="2048_2048,2048_1024", help="(Fallback mode) Comma-separated list of end marker combos (default: 2048_2048,2048_1024).")
    parser.add_argument("--start-markers", default="128_128", help="Comma-separated list of start marker combos (default: 128_128).")
    parser.add_argument("--apply", action="store_true", help="Actually move directories. Without this, a dry-run is performed.")
    parser.add_argument("--start-index", type=int, default=1, help="Starting index for run directory naming (e.g., _1).")
    parser.add_argument("--name-prefix", default=None, help="Override base name for run directories (default uses base folder name).")
    parser.add_argument("--end-gap-hours", type=int, default=24, help="(Fallback mode) Gap threshold (hours) after an end marker to close a run. Defaults to 24.")
    parser.add_argument("--min-week-gap-hours", type=int, default=120, help="Minimum gap (hours) between accepted weekly starts (default: 120).")

    args = parser.parse_args()

    base_dir = os.path.abspath(args.base)
    use_end_marker = args.use_end_markers
    end_markers = tuple([s.strip() for s in args.end_markers.split(',') if s.strip()])
    start_markers = tuple([s.strip() for s in args.start_markers.split(',') if s.strip()])

    print(f"Scanning base: {base_dir}")
    items = gather_dated_subdirs(base_dir)

    if not items:
        print("No dated subdirectories found.")
        return

    if use_end_marker:
        runs = partition_runs(items, gap_hours=args.gap_hours, end_marker_combos=end_markers, use_end_marker=True, end_gap_hours=args.end_gap_hours, start_marker_combos=start_markers, start_gap_hours=12)
    else:
        runs = partition_runs_by_starts(items, start_marker_combos=start_markers, min_week_gap_hours=args.min_week_gap_hours)

    summary = summarize_runs(runs)
    print("\nDetected runs:")
    for i, (start, end, count) in enumerate(summary, start=1):
        duration = (end - start) if (start and end) else timedelta(0)
        print(f"- Run {i}: {start} -> {end} ({duration}), items: {count}")

    apply = args.apply
    print("\nMode:")
    print("- APPLY" if apply else "- DRY-RUN (no changes)")

    plans = move_runs(runs, base_dir, apply=apply, start_index=args.start_index, base_name_override=args.name_prefix)

    print("\nPlanned moves:")
    for run_idx, (run_root, moves) in enumerate(plans, start=1):
        print(f"\nRun {run_idx} -> {run_root}")
        shown = 0
        for src, dest in moves:
            print(f"  {src} -> {dest}")
            shown += 1
        if shown == 0:
            print("  (no items)")

    if apply:
        print("\nMove completed.")
    else:
        print("\nDry-run complete. Re-run with --apply to perform moves.")


if __name__ == "__main__":
    main()
