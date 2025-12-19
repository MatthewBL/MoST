#!/usr/bin/env python3
"""Plot best_req_min per experiment in 3D.

Recommended visualization: 3D scatter where
- X = input tokens (first number in experiment name)
- Y = output tokens (second number in experiment name)
- Z = best_req_min

Color and marker size encode the magnitude of best_req_min for readability.

Usage:
    python plot_best_reqmin.py --csv requests/best_reqmin.csv --out requests/best_reqmin_3d.png

If the CSV is missing, the script will try to run `find_best_reqmin.py` to create it.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from typing import List, Tuple


def read_csv_fallback(path: str) -> List[dict]:
        rows = []
        with open(path, newline='', encoding='utf-8') as fh:
                reader = csv.DictReader(fh)
                for r in reader:
                        rows.append({k.strip(): (v if v is not None else '') for k, v in r.items()})
        return rows


def load_data(csv_path: str) -> List[Tuple[int, int, float, str]]:
        """Return list of (input_tokens, output_tokens, best_req_min, experiment_name)"""
        if not os.path.isfile(csv_path):
                raise FileNotFoundError(csv_path)

        # Try pandas if available for convenience
        try:
                import pandas as pd

                df = pd.read_csv(csv_path)
                rows = df.to_dict(orient='records')
        except Exception:
                rows = read_csv_fallback(csv_path)

        out = []
        for r in rows:
                exp = r.get('experiment') or r.get('Experiment') or ''
                best = r.get('best_req_min') or r.get('best_req_min'.upper()) or r.get('best_req_min'.lower())
                if best is None or best == '':
                        continue
                try:
                        best_val = float(best)
                except Exception:
                        continue

                # parse experiment name like 128_256
                if '_' in exp:
                        left, right = exp.split('_', 1)
                else:
                        # can't parse, skip
                        continue
                try:
                        inp = int(left)
                        outt = int(right)
                except Exception:
                        # skip unparsable
                        continue

                out.append((inp, outt, best_val, exp))
        return out


def ensure_csv(csv_path: str, base_dir: str = 'requests') -> str:
        if os.path.isfile(csv_path):
                return csv_path
        # try to create it by running find_best_reqmin.py
        script = os.path.join(base_dir, 'find_best_reqmin.py')
        if not os.path.isfile(script):
                raise FileNotFoundError(f"CSV not found at {csv_path} and {script} is missing")

        print(f"CSV {csv_path} not found — running {script} to generate it")
        try:
                subprocess.check_call([sys.executable, script, '-d', base_dir, '-o', csv_path])
        except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to generate CSV by running {script}: {e}")
        return csv_path


def plot_3d(data: List[Tuple[int, int, float, str]], out_path: str, show: bool = False) -> None:
        try:
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
                import numpy as np
        except Exception as e:
                raise RuntimeError("matplotlib (and numpy) are required to plot. Install via 'pip install matplotlib numpy'")

        xs = [d[0] for d in data]
        ys = [d[1] for d in data]
        zs = [d[2] for d in data]
        labels = [d[3] for d in data]

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection='3d')

        sc = ax.scatter(xs, ys, zs, c=zs, cmap='viridis', s=np.clip(np.array(zs) * 5, 20, 300), depthshade=True)
        ax.set_xlabel('Input tokens')
        ax.set_ylabel('Output tokens')
        ax.set_zlabel('best_req_min')
        ax.set_title('Best requests per minute by (input_tokens, output_tokens)')

        # colorbar
        cbar = fig.colorbar(sc, ax=ax, shrink=0.6)
        cbar.set_label('best_req_min')

        # annotate each point with experiment label (optional: skip if too many points)
        if len(data) <= 40:
                for x, y, z, lab in data:
                        ax.text(x, y, z, lab, size=8)

        # save
        out_dir = os.path.dirname(out_path)
        if out_dir and not os.path.isdir(out_dir):
                os.makedirs(out_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        print(f"Saved plot to {out_path}")
        if show:
                plt.show()


def plot_heatmap(data: List[Tuple[int, int, float, str]], out_path: str, annotate: bool = True) -> None:
        """Create a 2D heatmap pivoted by input tokens (x) and output tokens (y).

        Missing cells are shown as white/NaN. The heatmap is saved to out_path.
        """
        try:
                import matplotlib.pyplot as plt
                import numpy as np
        except Exception:
                raise RuntimeError("matplotlib and numpy are required to plot. Install via 'pip install matplotlib numpy'")

        # collect unique sorted axes
        inputs = sorted({d[0] for d in data})
        outputs = sorted({d[1] for d in data})
        inp_index = {v: i for i, v in enumerate(inputs)}
        out_index = {v: i for i, v in enumerate(outputs)}

        grid = np.full((len(outputs), len(inputs)), np.nan, dtype=float)  # rows=outputs(y), cols=inputs(x)

        for x, y, z, _ in data:
                i = out_index[y]
                j = inp_index[x]
                grid[i, j] = z

        fig, ax = plt.subplots(figsize=(10, 8))
        cmap = plt.cm.viridis
        cmap.set_bad(color='white')

        im = ax.imshow(grid, aspect='auto', origin='lower', cmap=cmap)
        ax.set_xticks(range(len(inputs)))
        ax.set_xticklabels([str(v) for v in inputs], rotation=45)
        ax.set_yticks(range(len(outputs)))
        ax.set_yticklabels([str(v) for v in outputs])
        ax.set_xlabel('Input tokens')
        ax.set_ylabel('Output tokens')
        ax.set_title('Heatmap of best_req_min (input x output)')

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('best_req_min')

        if annotate:
                # choose a font size based on grid size so numbers are readable
                max_dim = max(1, max(grid.shape))
                # heuristic: larger grids get smaller text, small grids get larger text
                fontsize = int(max(8, min(20, 200 // max_dim)))

                # annotate each cell with value if not NaN
                for i in range(grid.shape[0]):
                        for j in range(grid.shape[1]):
                                v = grid[i, j]
                                if not np.isnan(v):
                                        ax.text(j, i, f"{v:.2f}", ha='center', va='center',
                                                        color='white' if v < (np.nanmax(grid) / 2) else 'black',
                                                        fontsize=fontsize)

        plt.tight_layout()
        out_dir = os.path.dirname(out_path)
        if out_dir and not os.path.isdir(out_dir):
                os.makedirs(out_dir, exist_ok=True)
        plt.savefig(out_path, dpi=200)
        print(f"Saved heatmap to {out_path}")
        plt.close(fig)


def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--csv', default=os.path.join('requests', 'best_reqmin.csv'), help='CSV path with best req min per experiment')
        parser.add_argument('--out', default=os.path.join('requests', 'best_reqmin_3d.png'), help='Output PNG path')
        parser.add_argument('--show', action='store_true', help='Show plot interactively')
        parser.add_argument('--type', choices=['3d', 'heatmap', 'both'], default='both', help='Type of plot to produce')
        args = parser.parse_args()

        csv_path = os.path.abspath(args.csv)
        try:
                csv_path = ensure_csv(csv_path, base_dir=os.path.dirname(args.csv) or 'requests')
        except Exception as e:
                print("Error ensuring CSV:", e)
                sys.exit(1)

        try:
                data = load_data(csv_path)
        except Exception as e:
                print("Failed to load CSV:", e)
                sys.exit(1)

        if not data:
                print("No valid data found in", csv_path)
                sys.exit(0)

        out_path = os.path.abspath(args.out)
        plot_type = args.type
        try:
                if plot_type in ('3d', 'both'):
                        # if both, adjust output filename
                        out3d = out_path if plot_type == '3d' else os.path.splitext(out_path)[0] + '_3d.png'
                        plot_3d(data, out3d, show=args.show)
                if plot_type in ('heatmap', 'both'):
                        outheat = out_path if plot_type == 'heatmap' else os.path.splitext(out_path)[0] + '_heatmap.png'
                        plot_heatmap(data, outheat, annotate=True)
        except Exception as e:
                print("Failed to plot:", e)
                sys.exit(1)


if __name__ == '__main__':
        main()
