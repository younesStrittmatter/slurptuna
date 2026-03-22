from datetime import timedelta

from slurptuna import ExecutionMode, loss, optimize_run

PARTICIPANTS = {
    "p01": {"alpha": 0.20, "beta": 0.80},
    "p02": {"alpha": 0.35, "beta": 0.55},
    "p03": {"alpha": 0.60, "beta": 0.25},
    "p04": {"alpha": 0.45, "beta": 0.65},
}


@loss(
    name="participant_parallel_demo",
    description="Fit shared alpha/beta across participants",
    parameter_space={"alpha": (0.0, 1.0), "beta": (0.0, 1.0)},
)
def participant_parallel_demo(params, seed):
    # Return dict => slurptuna averages participant losses per seed.
    # Seed-dependent term mimics per-seed simulation stochasticity.
    seed_term = (seed % 23) * 1e-4
    return {
        pid: abs(params["alpha"] - truth["alpha"]) + abs(params["beta"] - truth["beta"]) + seed_term
        for pid, truth in PARTICIPANTS.items()
    }


if __name__ == "__main__":
    # LOCAL / single-mode run:
    # result = optimize_run(
    #     participant_parallel_demo,
    #     mode=ExecutionMode.SINGLE,
    #     n_trials=20,
    #     n_seeds=200,
    # )

    # DISTRIBUTED run with explicit seed/chunk controls:
    result = optimize_run(
        participant_parallel_demo,
        mode=ExecutionMode.DISTRIBUTED,
        n_trials=20,
        n_seeds=800,      # total seeds evaluated per trial
        chunk_size=40,    # seeds handled by each array task
        worker_time_limit=timedelta(minutes=30),
        # num_chunks is optional; if omitted it is derived from n_seeds/chunk_size
        # num_chunks=20,
    )

    print("best value:", result.best_value)
    print("best params:", result.best_params)
    print("run dir:", result.run_dir)
