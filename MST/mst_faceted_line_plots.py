import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List

# Mapping from group folder names to friendly titles for heatmap headers
GROUP_TITLE_MAP: Dict[str, str] = {
    "a40": "Deepseek 7B - A40",
    "a40-2": "Llama 8B - A40",
    "a40-3": "Gemma 7B - A40",
    "a100": "Deepseek 7B - A100",
    "a100-2": "Llama 8B - A100",
}

def find_experiment_csvs(base_dir: Path) -> Dict[str, List[Path]]:
    """
    Find per-experiment MST CSV files under base_dir.
    Returns a mapping of experiment name -> list of CSV Paths.
    """
    experiments: Dict[str, List[Path]] = {}
    for exp_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        # Include only mst_*.csv files, excluding aggregate/outlier reports
        csvs = [
            p for p in exp_dir.glob("mst_*.csv")
            if p.name not in {"mst_average.csv", "mst_outliers_report.csv"}
        ]
        if not csvs:
            continue
        experiments[exp_dir.name] = csvs
    return experiments


def load_and_label(csv_paths: List[Path], experiment: str) -> pd.DataFrame:
    frames = []
    for p in csv_paths:
        df = pd.read_csv(p)
        # Normalize column names just in case
        df = df.rename(columns={
            "input_tokens": "input_tokens",
            "output_tokens": "output_tokens",
            "mst_req_min": "mst_req_min",
        })
        # Ensure numeric types
        df["input_tokens"] = pd.to_numeric(df["input_tokens"], errors="coerce")
        df["output_tokens"] = pd.to_numeric(df["output_tokens"], errors="coerce")
        df["mst_req_min"] = pd.to_numeric(df["mst_req_min"], errors="coerce")
        df["experiment"] = experiment
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def aggregate_by_tokens(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Returns two DataFrames:
    - avg_by_input: columns [experiment, input_tokens, avg_mst]
    - avg_by_output: columns [experiment, output_tokens, avg_mst]
    """
    avg_by_input = (
        df.groupby(["experiment", "input_tokens"], as_index=False)["mst_req_min"]
          .mean()
          .rename(columns={"mst_req_min": "avg_mst"})
    )
    avg_by_output = (
        df.groupby(["experiment", "output_tokens"], as_index=False)["mst_req_min"]
          .mean()
          .rename(columns={"mst_req_min": "avg_mst"})
    )
    # Sort for plotting
    avg_by_input = avg_by_input.sort_values(["experiment", "input_tokens"]) 
    avg_by_output = avg_by_output.sort_values(["experiment", "output_tokens"]) 
    return avg_by_input, avg_by_output


def plot_faceted_lines(
    avg_by_input: pd.DataFrame,
    avg_by_output: pd.DataFrame,
    out_path_input: Path,
    out_path_output: Path,
):
    """
    Create two separate line plots and save them to different files:
    - Average MST vs Input Tokens (legend inside plot, upper-right)
    - Average MST vs Output Tokens (legend inside plot, upper-right)
    """
    experiments = sorted(set(avg_by_input["experiment"]))

    # Plot 1: input tokens
    fig1, ax1 = plt.subplots(figsize=(7, 6))
    for exp in experiments:
        sub = avg_by_input[avg_by_input["experiment"] == exp]
        label = GROUP_TITLE_MAP.get(exp, exp)
        ax1.plot(sub["input_tokens"], sub["avg_mst"], marker="o", label=label)
    ax1.set_title("Average MST by Input Tokens")
    ax1.set_xlabel("Input Tokens")
    ax1.set_ylabel("Average MST (req/min)")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    ax1.legend(loc="upper right", frameon=False)
    out_path_input.parent.mkdir(parents=True, exist_ok=True)
    fig1.tight_layout()
    fig1.savefig(out_path_input, dpi=200)
    plt.close(fig1)
    print(f"Saved figure to: {out_path_input}")

    # Plot 2: output tokens
    # Ensure the experiments list covers output as well (in case of mismatch)
    experiments_out = sorted(set(avg_by_output["experiment"]))
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    for exp in experiments_out:
        sub = avg_by_output[avg_by_output["experiment"] == exp]
        label = GROUP_TITLE_MAP.get(exp, exp)
        ax2.plot(sub["output_tokens"], sub["avg_mst"], marker="o", label=label)
    ax2.set_title("Average MST by Output Tokens")
    ax2.set_xlabel("Output Tokens")
    ax2.set_ylabel("Average MST (req/min)")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    ax2.legend(loc="upper right", frameon=False)
    out_path_output.parent.mkdir(parents=True, exist_ok=True)
    fig2.tight_layout()
    fig2.savefig(out_path_output, dpi=200)
    plt.close(fig2)
    print(f"Saved figure to: {out_path_output}")


def main():
    base_dir = Path("./plots")
    experiments_map = find_experiment_csvs(base_dir)

    if not experiments_map:
        raise SystemExit("No experiment CSVs found under plots directory.")

    all_frames = []
    for exp, paths in experiments_map.items():
        df = load_and_label(paths, exp)
        all_frames.append(df)
    all_df = pd.concat(all_frames, ignore_index=True)

    avg_by_input, avg_by_output = aggregate_by_tokens(all_df)

    out_input = base_dir / "mst_avg_by_input_tokens.png"
    out_output = base_dir / "mst_avg_by_output_tokens.png"
    plot_faceted_lines(avg_by_input, avg_by_output, out_input, out_output)


if __name__ == "__main__":
    main()
