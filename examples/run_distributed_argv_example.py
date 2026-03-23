from __future__ import annotations

import os
import sys

from slurptuna import execution_mode, loss, optimize_run

TARGETS = {
    "easy": {"alpha": 0.3, "beta": 0.7},
    "hard": {"alpha": 0.6, "beta": 0.2},
}

TASK = sys.argv[1] if len(sys.argv) > 1 else "easy"
if TASK not in TARGETS:
    valid = ", ".join(sorted(TARGETS))
    raise ValueError(f"task must be one of: {valid}")

TRUE = TARGETS[TASK]


@loss(
    name=f"my_model_{TASK}",
    description=f"Recover toy parameters for task={TASK}",
    parameter_space={"alpha": (0.0, 1.0), "beta": (0.0, 1.0)},
)
def my_model(params, seed):
    base = abs(params["alpha"] - TRUE["alpha"]) + abs(params["beta"] - TRUE["beta"])
    jitter = (seed % 17) * 1e-4
    return base + jitter


if __name__ == "__main__":
    run_name = (
        f"toy_distributed_argv_{TASK}_job_{os.environ.get('SLURM_JOB_ID', 'manual')}"
    )

    result = optimize_run(
        my_model,
        mode=execution_mode("distributed"),
        n_trials=2,
        n_seeds=4,
        chunk_size=2,
        run_root="runs",
        run_name=run_name,
        random_seed=7,
        max_concurrent_trials=1,
        worker_parallelism=1,
        cpus_per_task=1,
        slurm_poll_seconds=2,
        slurm_timeout_minutes=5,
        trial_retry_attempts=1,
    )

    print("task:", TASK)
    print("mode:", result.mode.value)
    print("true:", TRUE)
    print("found:", result.best_params)
    print("best value:", result.best_value)
    print("run dir:", result.run_dir)
