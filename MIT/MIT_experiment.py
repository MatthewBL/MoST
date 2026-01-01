import os
import subprocess
import time
from pathlib import Path

# Simpler experiment runner:
# - Iterates over a predefined list of INPUT/OUTPUT token pairs
# - For each pair, starts at REQ_MIN=1, runs loadgen
# - After each iteration, doubles REQ_MIN (exponential growth)
# - Stops when REQ_MIN reaches 128


def update_env(req_min: int, input_tokens: int, output_tokens: int) -> None:
    """Update or create .env with the required keys for this experiment.

    Preserves existing comments/keys where possible and updates only:
    - REQ_MIN
    - MIN_INPUT_TOKENS / MAX_INPUT_TOKENS
    - MIN_OUTPUT_TOKENS / MAX_OUTPUT_TOKENS
    """
    env_file = Path('.env')

    lines = []
    if env_file.exists():
        lines = env_file.read_text(encoding='utf-8').splitlines(True)  # keep line endings

    target_values = {
        'REQ_MIN': str(req_min),
        'MIN_INPUT_TOKENS': str(input_tokens),
        'MAX_INPUT_TOKENS': str(input_tokens),
        'MIN_OUTPUT_TOKENS': str(output_tokens),
        'MAX_OUTPUT_TOKENS': str(output_tokens),
    }

    found = {k: False for k in target_values.keys()}
    new_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#') or '=' not in stripped:
            new_lines.append(line)
            continue
        key, _ = stripped.split('=', 1)
        if key in target_values:
            new_lines.append(f"{key}={target_values[key]}\n")
            found[key] = True
        else:
            new_lines.append(line)

    # Append missing keys at the end if they were not found
    for key, value in target_values.items():
        if not found[key]:
            new_lines.append(f"{key}={value}\n")

    env_file.write_text(''.join(new_lines), encoding='utf-8')


def run_cmd(command: str, wait: bool = True) -> subprocess.Popen:
    print(f"Running: {command}")
    proc = subprocess.Popen(command, shell=True)
    if wait:
        proc.wait()
        if proc.returncode != 0:
            print(f"Warning: Command '{command}' returned non-zero exit code: {proc.returncode}")
    return proc


def _get_env_config() -> dict:
    """Read and return environment configuration used for storing results."""
    return {
        'MODEL': os.environ.get('MODEL', ''),
        'GPUS': os.environ.get('GPUS', ''),
        'CPUS': os.environ.get('CPUS', ''),
        'NODE': os.environ.get('NODE', ''),
    }


def _run_eval_and_store(stage: int, parent_dir: str, model: str, gpus: str, cpus: str, node: str) -> bool:
    """Run conversion/evaluation and store results.

    - Changes into the shared 'requests' directory to run conversion and evaluation scripts.
    - Then calls MST/store_results.py from the workspace root so results are persisted.
    Returns True if evaluation succeeded (exit code 0), else False.
    """
    workspace_root = Path(__file__).resolve().parent.parent
    original_dir = Path.cwd()

    # Run conversion/evaluation within the shared requests directory if it exists
    requests_dir = workspace_root / 'requests'
    evaluation_success = False
    try:
        if requests_dir.exists():
            os.chdir(requests_dir)
            run_cmd("python -u convert_to_csv.py", wait=True)
            run_cmd("python -u split_results.py", wait=True)
            eval_proc = run_cmd("python -u evaluate.py", wait=True)
            evaluation_success = (eval_proc.returncode == 0)
        else:
            print(f"Warning: requests directory not found at {requests_dir}. Skipping conversion/evaluation.")
    finally:
        os.chdir(workspace_root)

    # Always attempt to store results (mirroring MST logic)
    store_cmd = (
        f"python -u MST/store_results.py \"{model}\" \"{gpus}\" \"{cpus}\" \"{node}\" \"{stage}\" \"{parent_dir}\""
    )
    run_cmd(store_cmd, wait=True)

    # Return to original directory
    os.chdir(original_dir)
    return evaluation_success


def run_single_iteration(req_min: int, input_tokens: int, output_tokens: int, model: str, gpus: str, cpus: str, node: str, parent_dir: str, stage: int = 1) -> None:
    """Update env for the iteration, run load generator, then evaluate and store results."""
    update_env(req_min, input_tokens, output_tokens)

    # Regenerate inputs before running loadgen
    # Clear any stale sample_requests to ensure fresh generation
    workspace_root = Path(__file__).resolve().parent.parent
    requests_dir = workspace_root / 'requests'
    sample_file = requests_dir / 'sample_requests.json'
    try:
        if sample_file.exists():
            sample_file.unlink()
    except Exception as e:
        print(f"Warning: could not delete {sample_file}: {e}")

    run_cmd("python -u -m fmperf.loadgen.generate-input", wait=True)

    # Send traffic per the current REQ_MIN
    run_cmd("python -u -m fmperf.loadgen.run", wait=True)

    # Convert/evaluate and store
    _run_eval_and_store(stage=stage, parent_dir=parent_dir, model=model, gpus=gpus, cpus=cpus, node=node)


def run_simple_experiments() -> dict:
    """Run simple experiments across token pairs, doubling REQ_MIN until 128."""
    input_output_tokens = [
        [128, 128],
        [128, 256],
        [128, 512],
        [128, 1024],
        [128, 2048],
        [256, 128],
        [256, 256],
        [256, 512],
        [256, 1024],
        [256, 2048],
        [512, 128],
        [512, 256],
        [512, 512],
        [512, 1024],
        [512, 2048],
        [1024, 128],
        [1024, 256],
        [1024, 512],
        [1024, 1024],
        [1024, 2048],
        [2048, 128],
        [2048, 256],
        [2048, 512],
        [2048, 1024],
        [2048, 2048],
    ]

    results = {}
    cfg = _get_env_config()

    for tokens in input_output_tokens:
        input_tokens, output_tokens = tokens
        print("\n" + "=" * 60)
        print(f"Starting simple experiment for INPUT_TOKENS={input_tokens}, OUTPUT_TOKENS={output_tokens}")
        print("=" * 60)

        req_min = 1
        iterations = 0
        parent_dir = f"{input_tokens}_{output_tokens}"
        while True:
            iterations += 1
            print(f"\n--- Iteration {iterations}: REQ_MIN={req_min}, INPUT={input_tokens}, OUTPUT={output_tokens} ---")
            run_single_iteration(
                req_min,
                input_tokens,
                output_tokens,
                cfg['MODEL'],
                cfg['GPUS'],
                cfg['CPUS'],
                cfg['NODE'],
                parent_dir,
                stage=1,
            )

            # End experiment once REQ_MIN reaches 128 (include this iteration)
            if req_min == 128:
                print("Reached REQ_MIN=128. Ending experiment for this token pair.")
                break

            # Exponential increase (doubling)
            req_min *= 2

            # Small delay to avoid overwhelming the system
            time.sleep(1)

        results[f"{input_tokens}_{output_tokens}"] = 128  # Final REQ_MIN reached
        print(f"Completed token pair {input_tokens}_{output_tokens} at REQ_MIN=128")

    print("\n" + "=" * 60)
    print("ALL SIMPLE EXPERIMENTS COMPLETED")
    print("=" * 60)
    for token_combo, result in results.items():
        print(f"Tokens {token_combo}: Final REQ_MIN {result}")

    return results


if __name__ == "__main__":
    final_results = run_simple_experiments()
    print(f"Final results: {final_results}")
