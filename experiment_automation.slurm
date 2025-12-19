#!/bin/bash
#SBATCH --cpus-per-gpu=8 --gpus=a30:1 -t 08:00:00 --nodelist=gpu01

module purge
conda activate fmperf-env
pip install -r requirements.txt

# Run your experiment
echo "Start experiment"
python -u experiment_automation.py