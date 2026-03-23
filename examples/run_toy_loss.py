from __future__ import annotations

from datetime import timedelta
import os

from slurptuna import execution_mode, loss, optimize_run

TRUE = {"alpha": 0.3, "beta": 0.7}


@loss(
    name="my_model",
    description="Recover two parameters",
    parameter_space={"alpha": (0.0, 1.0), "beta": (0.0, 1.0)},
)
def my_model(params, seed):
    base = abs(params["alpha"] - TRUE["alpha"]) + abs(params["beta"] - TRUE["beta"])
    # Tiny seed-dependent term to mimic simulation noise and vary chunk summaries.
    jitter = (seed % 17) * 1e-4
    return base + jitter


if __name__ == "__main__":
    run_name = f"toy_distributed_smoketest_job_{os.environ.get('SLURM_JOB_ID', 'manual')}"

    # LOCAL / single-mode example.
    # result = optimize_run(
    #     my_model,
    #     mode=execution_mode("single"),
    #     n_trials=20,
    #     n_seeds=200,
    #     run_name="toy_single_local",
    # )

    # DISTRIBUTED example (Slurm array + reduce jobs).
    result = optimize_run(
        my_model,
        mode=execution_mode("distributed"),
        n_trials=3,
        n_seeds=6,
        chunk_size=2,
        run_name=run_name,
        loss_module="examples.run_toy_loss",
        max_concurrent_trials=1,
        worker_parallelism=1,
        cpus_per_task=1,
        slurm_poll_seconds=2,
        slurm_timeout_minutes=5,
        worker_time_limit=timedelta(minutes=5),
        trial_retry_attempts=1,
    )

    print("mode:", result.mode.value)
    print("true  :", TRUE)
    print("found :", result.best_params)
    print("best value:", result.best_value)
    print("run dir:", result.run_dir)

