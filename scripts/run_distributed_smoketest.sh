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

python - <<'PY'
from __future__ import annotations

from datetime import timedelta
import json
import os
from pathlib import Path

from examples.run_toy_loss import TRUE, my_model
from slurptuna import execution_mode, optimize_run


run_name = f"toy_distributed_smoketest_job_{os.environ.get('SLURM_JOB_ID', 'manual')}"

result = optimize_run(
    my_model,
    mode=execution_mode("distributed"),
    n_trials=2,
    n_seeds=4,
    chunk_size=2,
    run_root="runs",
    run_name=run_name,
    loss_module="examples.run_toy_loss",
    random_seed=7,
    max_concurrent_trials=1,
    worker_parallelism=1,
    cpus_per_task=1,
    slurm_poll_seconds=2,
    slurm_timeout_minutes=5,
    worker_time_limit=timedelta(minutes=5),
    trial_retry_attempts=1,
)

run_dir = Path(result.run_dir or "")
summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))

print(json.dumps(
    {
        "mode": result.mode.value,
        "true": TRUE,
        "best_params": result.best_params,
        "best_value": result.best_value,
        "run_dir": result.run_dir,
        "summary": summary,
        "meta": {
            "run_name": meta["run_name"],
            "n_trials": meta["n_trials"],
            "n_seeds": meta["n_seeds"],
            "chunk_size": meta["chunk_size"],
            "num_chunks": meta["num_chunks"],
        },
    },
    indent=2,
))
PY