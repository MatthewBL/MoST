#!/bin/bash
#SBATCH --cpus-per-gpu=4 --gpus=a30:1 -t 240:00:00

module purge
conda activate fmperf-env
pip install -r requirements.txt

# Env vars (adapt values to your cluster)
export TARGET=vllm
export URL=gpu06:9000
export REQUESTS_DIR=$PWD/requests
mkdir -p "$REQUESTS_DIR"
export REQUESTS_FILENAME=sample_requests.json
export RESULTS_FILENAME=results.json
export SAMPLE_SIZE=1
export FRAC_GREEDY=1.0
export NUM_USERS=10
export SWEEP_USERS=10,50,100,300,500,1000,2000,3000,5000,10000
export DURATION=120s
export BACKOFF=1s
export GRACE_PERIOD=5s

# Run 1
echo "Setting up experiment parameters for Run 1"

export REQUESTS_DIR=$PWD/16_16
mkdir -p "$REQUESTS_DIR"
export MIN_INPUT_TOKENS=16
export MAX_INPUT_TOKENS=16
export MIN_OUTPUT_TOKENS=16
export MAX_OUTPUT_TOKENS=16

# Run your experiment
echo "Generating requests"
python -m fmperf.loadgen.generate-input

echo "Start MIT experiment"
python -u  -m fmperf.loadgen.sweep


# Run 2
echo "Setting up experiment parameters for Run 2"

export REQUESTS_DIR=$PWD/32_16
mkdir -p "$REQUESTS_DIR"
export MIN_INPUT_TOKENS=32
export MAX_INPUT_TOKENS=32
export MIN_OUTPUT_TOKENS=16
export MAX_OUTPUT_TOKENS=16

# Run your experiment
echo "Generating requests"
python -m fmperf.loadgen.generate-input

echo "Start MIT experiment"
python -u  -m fmperf.loadgen.sweep


# Run 3
echo "Setting up experiment parameters for Run 3"

export REQUESTS_DIR=$PWD/64_16
mkdir -p "$REQUESTS_DIR"
export MIN_INPUT_TOKENS=64
export MAX_INPUT_TOKENS=64
export MIN_OUTPUT_TOKENS=16
export MAX_OUTPUT_TOKENS=16

# Run your experiment
echo "Generating requests"
python -m fmperf.loadgen.generate-input

echo "Start MIT experiment"
python -u  -m fmperf.loadgen.sweep


# Run 4
echo "Setting up experiment parameters for Run 4"

export REQUESTS_DIR=$PWD/128_16
mkdir -p "$REQUESTS_DIR"
export MIN_INPUT_TOKENS=128
export MAX_INPUT_TOKENS=128
export MIN_OUTPUT_TOKENS=16
export MAX_OUTPUT_TOKENS=16

# Run your experiment
echo "Generating requests"
python -m fmperf.loadgen.generate-input

echo "Start MIT experiment"
python -u  -m fmperf.loadgen.sweep


# Run 5
echo "Setting up experiment parameters for Run 5"

export REQUESTS_DIR=$PWD/256_16
mkdir -p "$REQUESTS_DIR"
export MIN_INPUT_TOKENS=256
export MAX_INPUT_TOKENS=256
export MIN_OUTPUT_TOKENS=16
export MAX_OUTPUT_TOKENS=16

# Run your experiment
echo "Generating requests"
python -m fmperf.loadgen.generate-input

echo "Start MIT experiment"
python -u  -m fmperf.loadgen.sweep


# Run 6
echo "Setting up experiment parameters for Run 6"

export REQUESTS_DIR=$PWD/16_32
mkdir -p "$REQUESTS_DIR"
export MIN_INPUT_TOKENS=16
export MAX_INPUT_TOKENS=16
export MIN_OUTPUT_TOKENS=32
export MAX_OUTPUT_TOKENS=32

# Run your experiment
echo "Generating requests"
python -m fmperf.loadgen.generate-input

echo "Start MIT experiment"
python -u  -m fmperf.loadgen.sweep


# Run 7
echo "Setting up experiment parameters for Run 7"
export REQUESTS_DIR=$PWD/32_32
mkdir -p "$REQUESTS_DIR"
export MIN_INPUT_TOKENS=32
export MAX_INPUT_TOKENS=32
export MIN_OUTPUT_TOKENS=32
export MAX_OUTPUT_TOKENS=32

# Run your experiment
echo "Generating requests"
python -m fmperf.loadgen.generate-input

echo "Start MIT experiment"
python -u  -m fmperf.loadgen.sweep


# Run 8
echo "Setting up experiment parameters for Run 8"
export REQUESTS_DIR=$PWD/64_32
mkdir -p "$REQUESTS_DIR"
export MIN_INPUT_TOKENS=64
export MAX_INPUT_TOKENS=64
export MIN_OUTPUT_TOKENS=32
export MAX_OUTPUT_TOKENS=32

# Run your experiment
echo "Generating requests"
python -m fmperf.loadgen.generate-input

echo "Start MIT experiment"
python -u  -m fmperf.loadgen.sweep


# Run 9
echo "Setting up experiment parameters for Run 9"
export REQUESTS_DIR=$PWD/128_32
mkdir -p "$REQUESTS_DIR"
export MIN_INPUT_TOKENS=128
export MAX_INPUT_TOKENS=128
export MIN_OUTPUT_TOKENS=32
export MAX_OUTPUT_TOKENS=32

# Run your experiment
echo "Generating requests"
python -m fmperf.loadgen.generate-input

echo "Start MIT experiment"
python -u  -m fmperf.loadgen.sweep


# Run 10
echo "Setting up experiment parameters for Run 10"
export REQUESTS_DIR=$PWD/256_32
mkdir -p "$REQUESTS_DIR"
export MIN_INPUT_TOKENS=256
export MAX_INPUT_TOKENS=256
export MIN_OUTPUT_TOKENS=32
export MAX_OUTPUT_TOKENS=32

# Run your experiment
echo "Generating requests"
python -m fmperf.loadgen.generate-input

echo "Start MIT experiment"
python -u  -m fmperf.loadgen.sweep


# Run 11
echo "Setting up experiment parameters for Run 11"
export REQUESTS_DIR=$PWD/16_64
mkdir -p "$REQUESTS_DIR"
export MIN_INPUT_TOKENS=16
export MAX_INPUT_TOKENS=16
export MIN_OUTPUT_TOKENS=64
export MAX_OUTPUT_TOKENS=64

# Run your experiment
echo "Generating requests"
python -m fmperf.loadgen.generate-input

echo "Start MIT experiment"
python -u  -m fmperf.loadgen.sweep


# Run 12
echo "Setting up experiment parameters for Run 12"
export REQUESTS_DIR=$PWD/32_64
mkdir -p "$REQUESTS_DIR"
export MIN_INPUT_TOKENS=32
export MAX_INPUT_TOKENS=32
export MIN_OUTPUT_TOKENS=64
export MAX_OUTPUT_TOKENS=64

# Run your experiment
echo "Generating requests"
python -m fmperf.loadgen.generate-input

echo "Start MIT experiment"
python -u  -m fmperf.loadgen.sweep


# Run 13
echo "Setting up experiment parameters for Run 13"
export REQUESTS_DIR=$PWD/64_64
mkdir -p "$REQUESTS_DIR"
export MIN_INPUT_TOKENS=64
export MAX_INPUT_TOKENS=64
export MIN_OUTPUT_TOKENS=64
export MAX_OUTPUT_TOKENS=64

# Run your experiment
echo "Generating requests"
python -m fmperf.loadgen.generate-input

echo "Start MIT experiment"
python -u  -m fmperf.loadgen.sweep


# Run 14
echo "Setting up experiment parameters for Run 14"
export REQUESTS_DIR=$PWD/128_64
mkdir -p "$REQUESTS_DIR"
export MIN_INPUT_TOKENS=128
export MAX_INPUT_TOKENS=128
export MIN_OUTPUT_TOKENS=64
export MAX_OUTPUT_TOKENS=64

# Run your experiment
echo "Generating requests"
python -m fmperf.loadgen.generate-input

echo "Start MIT experiment"
python -u  -m fmperf.loadgen.sweep


# Run 15
echo "Setting up experiment parameters for Run 15"
export REQUESTS_DIR=$PWD/256_64
mkdir -p "$REQUESTS_DIR"
export MIN_INPUT_TOKENS=256
export MAX_INPUT_TOKENS=256
export MIN_OUTPUT_TOKENS=64
export MAX_OUTPUT_TOKENS=64

# Run your experiment
echo "Generating requests"
python -m fmperf.loadgen.generate-input

echo "Start MIT experiment"
python -u  -m fmperf.loadgen.sweep


# Run 16
echo "Setting up experiment parameters for Run 16"
export REQUESTS_DIR=$PWD/16_128
mkdir -p "$REQUESTS_DIR"
export MIN_INPUT_TOKENS=16
export MAX_INPUT_TOKENS=16
export MIN_OUTPUT_TOKENS=128
export MAX_OUTPUT_TOKENS=128
# Run your experiment
echo "Generating requests"
python -m fmperf.loadgen.generate-input

echo "Start MIT experiment"
python -u  -m fmperf.loadgen.sweep


# Run 17
echo "Setting up experiment parameters for Run 17"
export REQUESTS_DIR=$PWD/32_128
mkdir -p "$REQUESTS_DIR"
export MIN_INPUT_TOKENS=32
export MAX_INPUT_TOKENS=32
export MIN_OUTPUT_TOKENS=128
export MAX_OUTPUT_TOKENS=128

# Run your experiment
echo "Generating requests"
python -m fmperf.loadgen.generate-input

echo "Start MIT experiment"
python -u  -m fmperf.loadgen.sweep


# Run 18
echo "Setting up experiment parameters for Run 18"
export REQUESTS_DIR=$PWD/64_128
mkdir -p "$REQUESTS_DIR"
export MIN_INPUT_TOKENS=64
export MAX_INPUT_TOKENS=64
export MIN_OUTPUT_TOKENS=128
export MAX_OUTPUT_TOKENS=128

# Run your experiment
echo "Generating requests"
python -m fmperf.loadgen.generate-input

echo "Start MIT experiment"
python -u  -m fmperf.loadgen.sweep


# Run 19
echo "Setting up experiment parameters for Run 19"
export REQUESTS_DIR=$PWD/128_128
mkdir -p "$REQUESTS_DIR"
export MIN_INPUT_TOKENS=128
export MAX_INPUT_TOKENS=128
export MIN_OUTPUT_TOKENS=128
export MAX_OUTPUT_TOKENS=128

# Run your experiment
echo "Generating requests"
python -m fmperf.loadgen.generate-input

echo "Start MIT experiment"
python -u  -m fmperf.loadgen.sweep


# Run 20
echo "Setting up experiment parameters for Run 20"
export REQUESTS_DIR=$PWD/256_128
mkdir -p "$REQUESTS_DIR"
export MIN_INPUT_TOKENS=256
export MAX_INPUT_TOKENS=256
export MIN_OUTPUT_TOKENS=128
export MAX_OUTPUT_TOKENS=128

# Run your experiment
echo "Generating requests"
python -m fmperf.loadgen.generate-input

echo "Start MIT experiment"
python -u  -m fmperf.loadgen.sweep


# Run 21
echo "Setting up experiment parameters for Run 21"
export REQUESTS_DIR=$PWD/16_256
mkdir -p "$REQUESTS_DIR"
export MIN_INPUT_TOKENS=16
export MAX_INPUT_TOKENS=16
export MIN_OUTPUT_TOKENS=256
export MAX_OUTPUT_TOKENS=256

# Run your experiment
echo "Generating requests"
python -m fmperf.loadgen.generate-input

echo "Start MIT experiment"
python -u  -m fmperf.loadgen.sweep


# Run 22
echo "Setting up experiment parameters for Run 22"
export REQUESTS_DIR=$PWD/32_256
mkdir -p "$REQUESTS_DIR"
export MIN_INPUT_TOKENS=32
export MAX_INPUT_TOKENS=32
export MIN_OUTPUT_TOKENS=256
export MAX_OUTPUT_TOKENS=256

# Run your experiment
echo "Generating requests"
python -m fmperf.loadgen.generate-input

echo "Start MIT experiment"
python -u  -m fmperf.loadgen.sweep


# Run 23
echo "Setting up experiment parameters for Run 23"
export REQUESTS_DIR=$PWD/64_256
mkdir -p "$REQUESTS_DIR"
export MIN_INPUT_TOKENS=64
export MAX_INPUT_TOKENS=64
export MIN_OUTPUT_TOKENS=256
export MAX_OUTPUT_TOKENS=256

# Run your experiment
echo "Generating requests"
python -m fmperf.loadgen.generate-input

echo "Start MIT experiment"
python -u  -m fmperf.loadgen.sweep


# Run 24
echo "Setting up experiment parameters for Run 24"
export REQUESTS_DIR=$PWD/128_256
mkdir -p "$REQUESTS_DIR"
export MIN_INPUT_TOKENS=128
export MAX_INPUT_TOKENS=128
export MIN_OUTPUT_TOKENS=256
export MAX_OUTPUT_TOKENS=256

# Run your experiment
echo "Generating requests"
python -m fmperf.loadgen.generate-input

echo "Start MIT experiment"
python -u  -m fmperf.loadgen.sweep


# Run 25
echo "Setting up experiment parameters for Run 25"
export REQUESTS_DIR=$PWD/256_256
mkdir -p "$REQUESTS_DIR"
export MIN_INPUT_TOKENS=256
export MAX_INPUT_TOKENS=256
export MIN_OUTPUT_TOKENS=256
export MAX_OUTPUT_TOKENS=256

# Run your experiment
echo "Generating requests"
python -m fmperf.loadgen.generate-input

echo "Start MIT experiment"
python -u  -m fmperf.loadgen.sweep
