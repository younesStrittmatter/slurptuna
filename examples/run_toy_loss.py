from __future__ import annotations

from datetime import timedelta

from slurptuna import ExecutionMode, loss, optimize_run

TRUE = {"alpha": 0.3, "beta": 0.7}


@loss(
    name="my_model",
    description="Recover two parameters",
    parameter_space={"alpha": (0.0, 1.0), "beta": (0.0, 1.0)},
)
def my_model(params, seed, context):
    _ = context
    base = abs(params["alpha"] - TRUE["alpha"]) + abs(params["beta"] - TRUE["beta"])
    # Tiny seed-dependent term to mimic simulation noise and vary chunk summaries.
    jitter = (seed % 17) * 1e-4
    return base + jitter


if __name__ == "__main__":
    # LOCAL / single-mode example.
    # result = optimize_run(
    #     my_model,
    #     mode=ExecutionMode.SINGLE,
    #     n_trials=20,
    #     n_seeds=200,
    #     run_name="toy_single_local",
    # )

    # DISTRIBUTED example (Slurm array + reduce jobs).
    result = optimize_run(
        my_model,
        mode=ExecutionMode.DISTRIBUTED,
        n_trials=6,
        n_seeds=120,
        chunk_size=20,
        run_name="toy_single_distributed",
        max_concurrent_trials=2,
        worker_parallelism=2,
        cpus_per_task=2,
        worker_time_limit=timedelta(minutes=30),
    )

    print("mode:", result.mode.value)
    print("true  :", TRUE)
    print("found :", result.best_params)
    print("best value:", result.best_value)
    print("run dir:", result.run_dir)

