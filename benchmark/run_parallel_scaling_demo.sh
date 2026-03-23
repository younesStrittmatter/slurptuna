#!/bin/bash
#SBATCH --job-name=slurptuna-scale-demo
#SBATCH --output=benchmark/logs/scaling_demo_%j.out
#SBATCH --error=benchmark/logs/scaling_demo_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail

cd /scratch/gpfs/JDC/younes/projects/slurptuna
source .venv/bin/activate
mkdir -p benchmark/logs

PYTHONPATH=/scratch/gpfs/JDC/younes/projects/slurptuna \
    python benchmark/run_parallel_scaling_demo.py \
    --n-trials 1 \
    --distributed-cpus-per-task 4 \
    --mem-per-cpu 2G \
    --slurm-timeout-minutes 180 \
    --loss-module benchmark.loss_definitions
