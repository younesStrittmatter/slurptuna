#!/bin/bash
#SBATCH --job-name=slurptuna-smoke
#SBATCH --output=logs/slurptuna_smoke_%j.out
#SBATCH --error=logs/slurptuna_smoke_%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail

cd /scratch/gpfs/JDC/younes/projects/slurptuna
source .venv/bin/activate
python examples/run_toy_loss.py