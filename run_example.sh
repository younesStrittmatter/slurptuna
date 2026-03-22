#!/bin/bash
#SBATCH --job-name=slurptuna-controller
#SBATCH --output=logs/controller_%j.out
#SBATCH --error=logs/controller_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

source .venv/bin/activate
python examples/run_toy_loss.py
