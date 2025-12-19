#!/usr/bin/env python3
"""Find highest REQ_MIN with True EVALUATION per experiment folder.

Usage:
    python find_best_reqmin.py -d <requests_dir> [-o out.csv]

By default prints results to stdout. Optionally writes CSV with columns:
experiment,best_req_min,date_folder,results_csv_path
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
from typing import Dict, Optional, Tuple


def parse_bool(s: Optional[str]) -> bool:
        if s is None:
                return False
        v = s.strip().lower()
        return v in ("true", "1", "yes", "y", "t")


def parse_float(s: Optional[str]) -> Optional[float]:
        if s is None:
                return None
        try:
                return float(s)
        except Exception:
                return None


def find_best_reqmin_for_base(base_dir: str) -> Dict[str, Tuple[float, str, str]]:
        """Scan base_dir and return mapping experiment -> (best_req_min, date_folder, results_csv_path).

        Only considers rows where EVALUATION is True. If none found for an experiment it will be omitted.
        """
        results: Dict[str, Tuple[float, str, str]] = {}

        if not os.path.isdir(base_dir):
                raise ValueError(f"base_dir not found or not a directory: {base_dir}")

        for entry in sorted(os.listdir(base_dir)):
                exp_path = os.path.join(base_dir, entry)
                if not os.path.isdir(exp_path):
                        continue
                experiment = entry

                best_val: Optional[float] = None
                best_date: Optional[str] = None
                best_csv_path: Optional[str] = None

                for date_entry in sorted(os.listdir(exp_path)):
                        date_path = os.path.join(exp_path, date_entry)
                        if not os.path.isdir(date_path):
                                continue
                        csv_path = os.path.join(date_path, "results.csv")
                        if not os.path.isfile(csv_path):
                                continue

                        try:
                                with open(csv_path, newline='', encoding='utf-8') as fh:
                                        reader = csv.DictReader(fh)
                                        for raw_row in reader:
                                                # normalize keys
                                                row = {k.strip(): (v if v is not None else "") for k, v in raw_row.items()}
                                                if 'EVALUATION' not in row or 'REQ_MIN' not in row:
                                                        # try case-insensitive fallback
                                                        keys_low = {k.lower(): k for k in row.keys()}
                                                        if 'evaluation' in keys_low and 'req_min' in keys_low:
                                                                evaluation_key = keys_low['evaluation']
                                                                reqmin_key = keys_low['req_min']
                                                                evaluation = parse_bool(row.get(evaluation_key))
                                                                reqmin = parse_float(row.get(reqmin_key))
                                                        else:
                                                                logging.debug("Skipping %s: missing EVALUATION or REQ_MIN columns", csv_path)
                                                                continue
                                                else:
                                                        evaluation = parse_bool(row.get('EVALUATION'))
                                                        reqmin = parse_float(row.get('REQ_MIN'))

                                                if not evaluation:
                                                        continue
                                                if reqmin is None:
                                                        continue

                                                if best_val is None or reqmin > best_val:
                                                        best_val = reqmin
                                                        best_date = date_entry
                                                        best_csv_path = csv_path
                        except Exception as e:
                                logging.warning("Failed reading %s: %s", csv_path, e)

                if best_val is not None:
                        results[experiment] = (best_val, best_date or "", best_csv_path or "")

        return results


def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("-d", "--dir", dest="base_dir", default="requests",
                                                help="Path to requests directory (default: requests)")
        parser.add_argument("-o", "--out", dest="out_csv", help="Optional output CSV file")
        parser.add_argument("-v", "--verbose", action="store_true")
        args = parser.parse_args()

        logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                                                format="%(levelname)s: %(message)s")

        base_dir = args.base_dir
        results = find_best_reqmin_for_base(base_dir)

        if not results:
                print("No successful experiments (EVALUATION True) found in:", base_dir)
        # Print results to stdout
        print("experiment,best_req_min,date_folder,results_csv_path")
        for exp in sorted(results.keys()):
                best_val, best_date, best_csv = results[exp]
                print(f"{exp},{best_val},{best_date},{best_csv}")

        # Determine output CSV path: use provided -o or default to base_dir/best_reqmin.csv
        out_csv = args.out_csv or os.path.join(base_dir, "best_reqmin.csv")
        try:
                # Ensure output directory exists
                out_dir = os.path.dirname(out_csv)
                if out_dir and not os.path.isdir(out_dir):
                        os.makedirs(out_dir, exist_ok=True)

                with open(out_csv, 'w', newline='', encoding='utf-8') as outfh:
                        writer = csv.writer(outfh)
                        writer.writerow(['experiment', 'best_req_min', 'date_folder', 'results_csv_path'])
                        for exp in sorted(results.keys()):
                                best_val, best_date, best_csv = results[exp]
                                writer.writerow([exp, best_val, best_date, best_csv])
                logging.info("Wrote results to %s", out_csv)
        except Exception as e:
                logging.error("Failed writing output CSV %s: %s", out_csv, e)


if __name__ == '__main__':
        main()
