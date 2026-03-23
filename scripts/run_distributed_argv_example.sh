#!/bin/bash
#SBATCH --job-name=slurptuna-argv-example
#SBATCH --output=logs/slurptuna_argv_example_%j.out
#SBATCH --error=logs/slurptuna_argv_example_%j.err
#SBATCH --time=02:00:00
#SBATCH --qos=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail

cd /scratch/gpfs/JDC/younes/projects/slurptuna
source .venv/bin/activate

TASK="${1:-easy}"
/scratch/gpfs/JDC/younes/projects/slurptuna/.venv/bin/python examples/run_distributed_argv_example.py "$TASK"
